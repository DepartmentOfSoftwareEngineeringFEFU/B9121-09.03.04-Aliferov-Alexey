import numpy as np
import soundfile as sf
import mido
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReferenceGenerator:
    """
    Генератор эталонного аудио из MIDI-файла и WAV-семплов нот.
    
    Анализирует MIDI-файл, извлекает нотные события (note_on/note_off),
    загружает соответствующие семплы из папки samples и создает итоговый
    аудиофайл путем микширования обрезанных семплов.
    """
    
    def __init__(self, samples_dir: str = "samples", sample_rate: int = 44100):
        """
        Инициализация генератора.
        
        Args:
            samples_dir (str): Путь к папке с WAV-семплами нот
            sample_rate (int): Частота дискретизации выходного аудио
        """
        self.samples_dir = Path(samples_dir)
        self.sample_rate = sample_rate
        self.samples_cache = {}  # Кэш загруженных семплов
        
        # Проверяем существование папки с семплами
        if not self.samples_dir.exists():
            raise FileNotFoundError(f"Папка с семплами не найдена: {self.samples_dir}")
    
    def _load_sample(self, midi_note: int) -> Optional[np.ndarray]:
        """
        Загружает семпл для указанной MIDI-ноты.
        
        Args:
            midi_note (int): MIDI-номер ноты (48-84)
            
        Returns:
            np.ndarray или None: Аудиоданные семпла или None, если файл не найден
        """
        # Проверяем кэш
        if midi_note in self.samples_cache:
            return self.samples_cache[midi_note]
        
        # Формируем путь к файлу семпла
        sample_path = self.samples_dir / f"{midi_note}.wav"
        
        if not sample_path.exists():
            logger.warning(f"Семпл для ноты {midi_note} не найден: {sample_path}")
            return None
        
        try:
            # Загружаем аудиофайл
            audio_data, original_sr = sf.read(str(sample_path))
            
            # Приводим к моно, если стерео
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ресемплируем, если нужно
            if original_sr != self.sample_rate:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=self.sample_rate)
            
            # Сохраняем в кэш
            self.samples_cache[midi_note] = audio_data
            
            logger.debug(f"Загружен семпл для ноты {midi_note}, длительность: {len(audio_data)/self.sample_rate:.3f}с")
            return audio_data
            
        except Exception as e:
            logger.error(f"Ошибка загрузки семпла {sample_path}: {e}")
            return None
    
    def _extract_midi_events(self, midi_path: str) -> List[Tuple[int, float, float, int]]:
        """
        Извлекает нотные события из MIDI-файла.
        
        Args:
            midi_path (str): Путь к MIDI-файлу
            
        Returns:
            List[Tuple[int, float, float, int]]: Список событий (pitch, start_time, end_time, velocity)
        """
        try:
            mid = mido.MidiFile(midi_path)
        except Exception as e:
            raise ValueError(f"Не удалось загрузить MIDI-файл {midi_path}: {e}")
        
        # Извлекаем темп (по умолчанию 120 BPM = 500000 микросекунд на бит)
        tempo = 500000  # микросекунд на бит
        ticks_per_beat = mid.ticks_per_beat
        
        # Ищем первое сообщение set_tempo
        for track in mid.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                    break
            if tempo != 500000:
                break
        
        # Вычисляем секунд на тик
        seconds_per_tick = (tempo / 1_000_000) / ticks_per_beat
        
        # Извлекаем события note_on/note_off
        events = []
        active_notes = {}  # active_notes[pitch] = {'start_time': time, 'velocity': vel}
        current_time = 0.0
        
        # Объединяем все треки в один поток событий
        all_messages = []
        for track in mid.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
                if msg.type in ['note_on', 'note_off', 'set_tempo']:
                    all_messages.append((track_time, msg))
        
        # Сортируем по времени
        all_messages.sort(key=lambda x: x[0])
        
        # Обрабатываем события
        for abs_time, msg in all_messages:
            current_time = abs_time * seconds_per_tick
            
            if msg.type == 'set_tempo':
                # Обновляем темп (если встречается в середине композиции)
                tempo = msg.tempo
                seconds_per_tick = (tempo / 1_000_000) / ticks_per_beat
                
            elif msg.type == 'note_on' and msg.velocity > 0:
                # Начало ноты
                pitch = msg.note
                velocity = msg.velocity
                active_notes[pitch] = {'start_time': current_time, 'velocity': velocity}
                
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                # Окончание ноты
                pitch = msg.note
                if pitch in active_notes:
                    start_time = active_notes[pitch]['start_time']
                    velocity = active_notes[pitch]['velocity']
                    end_time = current_time
                    
                    # Добавляем событие только если длительность больше 0
                    if end_time > start_time:
                        events.append((pitch, start_time, end_time, velocity))
                    
                    del active_notes[pitch]
        
        # Закрываем оставшиеся активные ноты
        final_time = current_time
        for pitch, note_info in active_notes.items():
            start_time = note_info['start_time']
            velocity = note_info['velocity']
            if final_time > start_time:
                events.append((pitch, start_time, final_time, velocity))
        
        # Сортируем события по времени начала
        events.sort(key=lambda x: x[1])
        
        logger.info(f"Извлечено {len(events)} нотных событий из MIDI-файла")
        return events
    
    def _crop_sample(self, sample: np.ndarray, duration: float, velocity: int) -> np.ndarray:
        """
        Обрезает семпл до нужной длительности и применяет velocity.
        
        Args:
            sample (np.ndarray): Исходный семпл
            duration (float): Требуемая длительность в секундах
            velocity (int): Velocity ноты (0-127)
            
        Returns:
            np.ndarray: Обрезанный и обработанный семпл
        """
        # Вычисляем количество семплов для нужной длительности
        target_samples = int(duration * self.sample_rate)
        
        if target_samples <= 0:
            return np.array([])
        
        # Применяем velocity как масштабирующий коэффициент
        velocity_scale = velocity / 127.0
        
        if len(sample) >= target_samples:
            # Обрезаем семпл до нужной длины
            cropped = sample[:target_samples].copy()
        else:
            # Если семпл короче нужной длительности, дополняем тишиной
            cropped = np.zeros(target_samples)
            cropped[:len(sample)] = sample
        
        # Применяем velocity
        cropped *= velocity_scale
        
        # Добавляем плавное затухание в конце, чтобы избежать щелчков
        fade_samples = min(1000, len(cropped) // 10)  # 1000 семплов или 10% длины
        if fade_samples > 0:
            fade_curve = np.linspace(1.0, 0.0, fade_samples)
            cropped[-fade_samples:] *= fade_curve
        
        return cropped
    
    def generate_reference(self, midi_path: str, output_path: str) -> bool:
        """
        Генерирует эталонный аудиофайл из MIDI-файла.
        
        Args:
            midi_path (str): Путь к входному MIDI-файлу
            output_path (str): Путь для сохранения выходного WAV-файла
            
        Returns:
            bool: True при успешном выполнении, False при ошибке
        """
        try:
            logger.info(f"Начинаем генерацию эталона для {midi_path}")
            
            # Извлекаем события из MIDI
            events = self._extract_midi_events(midi_path)
            
            if not events:
                logger.warning("В MIDI-файле не найдено нотных событий")
                return False
            
            # Определяем общую длительность композиции
            total_duration = max(event[2] for event in events)  # максимальное end_time
            total_samples = int(total_duration * self.sample_rate)  # убрано + self.sample_rate

            # Создаем итоговый аудиобуфер
            output_audio = np.zeros(total_samples, dtype=np.float32)
            
            logger.info(f"Обрабатываем {len(events)} событий, общая длительность: {total_duration:.2f}с")
            
            # Обрабатываем каждое событие
            successful_events = 0
            for pitch, start_time, end_time, velocity in events:
                # Загружаем семпл для данной ноты
                sample = self._load_sample(pitch)
                if sample is None:
                    continue
                
                # Вычисляем длительность ноты
                duration = end_time - start_time
                
                # Обрезаем семпл до нужной длительности
                processed_sample = self._crop_sample(sample, duration, velocity)
                
                if len(processed_sample) == 0:
                    continue
                
                # Вычисляем позицию в выходном буфере
                start_sample = int(start_time * self.sample_rate)
                end_sample = start_sample + len(processed_sample)
                
                # Проверяем границы
                if start_sample >= total_samples:
                    continue
                
                if end_sample > total_samples:
                    # Обрезаем, если выходим за буфер
                    processed_sample = processed_sample[:total_samples - start_sample]
                    end_sample = total_samples
                
                # Добавляем обработанный семпл к выходному аудио
                output_audio[start_sample:end_sample] += processed_sample
                successful_events += 1
            
            logger.info(f"Успешно обработано {successful_events} из {len(events)} событий")
            
            # Нормализуем аудио для предотвращения клиппинга
            max_amplitude = np.max(np.abs(output_audio))
            if max_amplitude > 0.95:  # Если амплитуда близка к максимуму
                normalization_factor = 0.95 / max_amplitude
                output_audio *= normalization_factor
                logger.info(f"Аудио нормализовано с коэффициентом {normalization_factor:.3f}")
            
            # Сохраняем результат
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            sf.write(str(output_path), output_audio, self.sample_rate)
            
            logger.info(f"Эталонный аудиофайл сохранен: {output_path}")
            logger.info(f"Длительность: {len(output_audio)/self.sample_rate:.2f}с, "
                       f"частота: {self.sample_rate}Гц")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при генерации эталона: {e}")
            return False
    
    def clear_cache(self):
        """Очищает кэш загруженных семплов."""
        self.samples_cache.clear()
        logger.info("Кэш семплов очищен")


def generate_reference_audio(midi_path: str, output_path: str, 
                           samples_dir: str = "samples", 
                           sample_rate: int = 44100) -> bool:
    """
    Функция для генерации эталонного аудио.
    
    Args:
        midi_path (str): Путь к MIDI-файлу
        output_path (str): Путь для сохранения WAV-файла
        samples_dir (str): Папка с семплами нот
        sample_rate (int): Частота дискретизации
        
    Returns:
        bool: True при успехе, False при ошибке
    """
    generator = ReferenceGenerator(samples_dir, sample_rate)
    return generator.generate_reference(midi_path, output_path)


# Тестирование модуля
if __name__ == "__main__":
    midi_file = "exercises/ёлочка.mid"  # Заменить на реальный путь
    output_file = "reference_audio/example_reference.wav" # Заменить на реальный путь

    success = generate_reference_audio(midi_file, output_file)

    if success:
        print("Эталонный аудиофайл успешно создан!")
    else:
        print("Ошибка при создании эталонного аудиофайла.")
