from tkinter import messagebox
import numpy as np
import pypianoroll
import mido
import pygame
import pygame.midi
import time
import sounddevice as sd
import soundfile as sf
import threading
import queue
from pathlib import Path
from const import (MIN_NOTE, MAX_NOTE, WHITE_KEY_WIDTH, BLACK_KEY_WIDTH, 
                   WHITE_KEY_HEIGHT, BLACK_KEY_HEIGHT, KEY_HEIGHT, SCREEN_WIDTH, 
                   SCREEN_HEIGHT, LEAD_TIME, is_white_key, is_black_key)
from reference_generator import ReferenceGenerator
from performance_evaluator import PerformanceEvaluator

class AudioRecorder:
    """Класс для записи аудио во время визуализации."""
    
    def __init__(self, device_name=None, sample_rate=44100):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.audio_queue = queue.Queue()
        self.recording = False
        self.recorded_data = []
        
        # Определяем device_id из имени устройства
        self.device_id = None
        if device_name:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                if f"{dev['name']} (#{idx})" == device_name:
                    self.device_id = idx
                    break
    
    def _audio_callback(self, indata, frames, time, status):
        """Callback-функция для записи аудио."""
        if status:
            print(f"Audio callback status: {status}")
        if self.recording:
            self.audio_queue.put(indata.copy())
    
    def start_recording(self):
        """Начинает запись аудио."""
        self.recording = True
        self.recorded_data = []
        
        # Запускаем поток для обработки аудиоданных
        self.processing_thread = threading.Thread(target=self._process_audio_queue)
        self.processing_thread.start()
        
        # Запускаем запись
        self.stream = sd.InputStream(
            device=self.device_id,
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback
        )
        self.stream.start()
    
    def _process_audio_queue(self):
        """Обрабатывает очередь аудиоданных."""
        while self.recording or not self.audio_queue.empty():
            try:
                data = self.audio_queue.get(timeout=0.1)
                self.recorded_data.append(data.flatten())
            except queue.Empty:
                continue
    
    def stop_recording(self):
        """Останавливает запись и возвращает записанные данные."""
        self.recording = False
        
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()
        
        if self.recorded_data:
            return np.concatenate(self.recorded_data)
        else:
            return np.array([])

def parse_midi(path_to_midi):
    """
    Читает MIDI‑файл и возвращает NumPy‑массив событий для падающего пианоролла.
    """
    multitrack = pypianoroll.read(path_to_midi)
    resolution = multitrack.resolution

    merged = multitrack.blend(mode='max').astype(np.int16)

    mid = mido.MidiFile(path_to_midi)
    tempo = 500_000
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break
        if tempo != 500_000:
            break

    sec_per_tick = (tempo / 1_000_000) / resolution

    T, P = merged.shape
    prev = np.zeros((P,), dtype=np.int16)
    ongoing = {}
    events_list = []

    for t in range(T):
        for p in range(P):
            vel = int(merged[t, p])
            if vel > 0 and prev[p] == 0:
                ongoing[p] = {'t_on_tick': t, 'vel': vel}
            elif vel == 0 and prev[p] > 0:
                if p in ongoing:
                    t0 = ongoing[p]['t_on_tick']
                    v0 = ongoing[p]['vel']
                    t_on_sec = t0 * sec_per_tick
                    t_off_sec = t * sec_per_tick
                    events_list.append((p, t_on_sec, t_off_sec, v0))
                    del ongoing[p]
        prev[:] = merged[t, :]

    for p, info in ongoing.items():
        t0 = info['t_on_tick']
        v0 = info['vel']
        t_on_sec = t0 * sec_per_tick
        t_off_sec = T * sec_per_tick
        events_list.append((p, t_on_sec, t_off_sec, v0))

    events_list.sort(key=lambda x: x[1])

    dtype = np.dtype([
        ('pitch',    np.int16),
        ('t_on',     np.float64),
        ('t_off',    np.float64),
        ('velocity', np.int16)
    ])
    events_array = np.array(events_list, dtype=dtype)
    return events_array

def pitch_to_key_position(pitch):
    """Возвращает позицию и размеры клавиши для заданной MIDI-ноты."""
    if pitch < MIN_NOTE or pitch > MAX_NOTE:
        return None
    
    if is_white_key(pitch):
        white_keys_before = 0
        for note in range(MIN_NOTE, pitch):
            if is_white_key(note):
                white_keys_before += 1
        
        x = white_keys_before * WHITE_KEY_WIDTH
        return {
            'x': x,
            'width': WHITE_KEY_WIDTH,
            'height': WHITE_KEY_HEIGHT,
            'is_black': False
        }
    else:
        left_white_pitch = pitch - 1
        while left_white_pitch >= MIN_NOTE and is_black_key(left_white_pitch):
            left_white_pitch -= 1
        
        if left_white_pitch < MIN_NOTE:
            return None
            
        white_keys_before_left_white = 0
        for note in range(MIN_NOTE, left_white_pitch):
            if is_white_key(note):
                white_keys_before_left_white += 1
        
        left_white_x = white_keys_before_left_white * WHITE_KEY_WIDTH
        x = left_white_x + WHITE_KEY_WIDTH - BLACK_KEY_WIDTH // 2
        return {
            'x': x,
            'width': BLACK_KEY_WIDTH,
            'height': BLACK_KEY_HEIGHT,
            'is_black': True
        }

def pitch_to_color(pitch):
    """Возвращает цвет для заданной MIDI-ноты."""
    import colorsys
    
    if pitch < MIN_NOTE or pitch > MAX_NOTE:
        return (128, 128, 128)
    
    note = pitch % 12
    hue = note / 12.0
    saturation = 0.9
    value = 0.95
    
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))

class NoteRect:
    """Описывает ноту в пианоролле."""
    def __init__(self, pitch, t_on, t_off, velocity, pixels_per_second):
        self.pitch = pitch
        self.t_on = t_on
        self.t_off = t_off
        self.velocity = velocity
        self.pixels_per_second = pixels_per_second

        key_pos = pitch_to_key_position(pitch)
        if key_pos:
            self.x = key_pos['x']
            self.width = key_pos['width']
            self.is_black = key_pos['is_black']
        else:
            self.x = None
            self.width = None
            self.is_black = False
        
        dur = t_off - t_on
        self.height = max(int(dur * pixels_per_second), 8)
        self.color = pitch_to_color(pitch)

    def draw(self, surface, y_offset, key_top_y, current_time):
        """Рисует ноту с учетом текущего смещения временного слоя."""
        if self.x is None:
            return
            
        note_y_in_layer = -self.height
        note_y_on_screen = note_y_in_layer + y_offset
        
        if (note_y_on_screen < SCREEN_HEIGHT and 
            note_y_on_screen + self.height > 0):
            
            rect = pygame.Rect(self.x, int(note_y_on_screen), self.width, self.height)
            pygame.draw.rect(surface, self.color, rect)

def draw_keyboard(screen):
    """Рисует клавиатуру пианино."""
    key_top_y = SCREEN_HEIGHT - KEY_HEIGHT
    
    for note in range(MIN_NOTE, MAX_NOTE + 1):
        if is_white_key(note):
            key_pos = pitch_to_key_position(note)
            if key_pos:
                rect = pygame.Rect(key_pos['x'], key_top_y, key_pos['width'], key_pos['height'])
                pygame.draw.rect(screen, (255, 255, 255), rect)
                pygame.draw.rect(screen, (50, 50, 50), rect, 1)
    
    for note in range(MIN_NOTE, MAX_NOTE + 1):
        if is_black_key(note):
            key_pos = pitch_to_key_position(note)
            if key_pos:
                rect = pygame.Rect(key_pos['x'], key_top_y, key_pos['width'], key_pos['height'])
                pygame.draw.rect(screen, (30, 30, 30), rect)
                pygame.draw.rect(screen, (100, 100, 100), rect, 1)

def highlight_active_keys(screen, active_keys):
    """Подсвечивает активные клавиши поверх клавиатуры."""
    key_top_y = SCREEN_HEIGHT - KEY_HEIGHT
    
    for pitch, color in active_keys:
        key_pos = pitch_to_key_position(pitch)
        if key_pos:
            rect = pygame.Rect(key_pos['x'], key_top_y, key_pos['width'], key_pos['height'])
            pygame.draw.rect(screen, color, rect, 3)

def extract_device_id(device_name):
    """Извлекает ID устройства из строки формата 'Имя устройства (#ID)'."""
    if not device_name:
        return None
    
    import re
    match = re.search(r'#(\d+)\)$', device_name)
    if match:
        return int(match.group(1))
    return None

def run_visualization(midi_path, enable_sound=False, device_name=None):
    """Запускает визуализацию с записью звука и оценкой исполнения."""
    
    # Получаем события из MIDI
    events = parse_midi(midi_path)
    if len(events) == 0:
        print("Не удалось найти note_on/note_off в MIDI.")
        return

    # Находим временные границы
    earliest_t_on = min(ev['t_on'] for ev in events)
    last_t_off = max(ev['t_off'] for ev in events)

    # Параметры визуализации
    lead_time = LEAD_TIME
    key_top_y = SCREEN_HEIGHT - KEY_HEIGHT
    travel_distance = key_top_y + 50
    pixels_per_second = travel_distance / lead_time

    # Создаём список NoteRect
    note_rects = []
    for ev in events:
        if ev['pitch'] < MIN_NOTE or ev['pitch'] > MAX_NOTE:
            continue
        nr = NoteRect(
            pitch=ev['pitch'],
            t_on=ev['t_on'],
            t_off=ev['t_off'],
            velocity=ev['velocity'],
            pixels_per_second=pixels_per_second
        )
        note_rects.append(nr)

    # Инициализируем Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Пианоролл-анимация")
    clock = pygame.time.Clock()
    FPS = 60

    # Инициализация MIDI синтезатора
    midi_out = None
    if enable_sound:
        pygame.midi.init()
        try:
            if pygame.midi.get_count() > 0:
                midi_out = pygame.midi.Output(pygame.midi.get_default_output_id())
            else:
                print("Нет доступных MIDI-устройств для вывода.")
        except Exception as e:
            print("Ошибка инициализации MIDI:", e)
            midi_out = None

    # Инициализация записи звука
    recorder = None
    if device_name:
        try:
            recorder = AudioRecorder(device_name)
        except Exception as e:
            print(f"Ошибка инициализации записи звука: {e}")
            recorder = None

    active_notes = set()
    midi_channel = 0

    # Время начала программы
    program_start_time = time.time()
    visualization_start_time = earliest_t_on - lead_time
    
    # Флаг начала записи
    recording_started = False

    running = True
    while running:
        current_time = (time.time() - program_start_time) + visualization_start_time

        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Очистка экрана
        screen.fill((30, 30, 30))

        # Вычисляем смещение временного слоя
        elapsed_time = current_time - visualization_start_time
        active_keys = []

        # Запускаем запись в момент начала первой ноты
        if not recording_started and recorder and current_time >= earliest_t_on:
            recorder.start_recording()
            recording_started = True
            print("Начата запись звука")

        # MIDI: управление звучанием нот
        notes_to_remove = set()
        for pitch in active_notes:
            still_active = any(
                nr.pitch == pitch and nr.t_on <= current_time <= nr.t_off
                for nr in note_rects
            )
            if not still_active and midi_out:
                try:
                    midi_out.note_off(int(pitch), 0)
                except TypeError:
                    midi_out.note_off(int(pitch), 0)
                notes_to_remove.add(pitch)
        active_notes -= notes_to_remove

        if elapsed_time >= 0:
            for nr in note_rects:
                time_until_touch = nr.t_on - current_time
                y_offset = -lead_time * pixels_per_second + (lead_time - time_until_touch) * pixels_per_second + key_top_y
                if time_until_touch <= lead_time and time_until_touch >= -2.0:
                    nr.draw(screen, y_offset, key_top_y, current_time)
                if nr.t_on <= current_time <= nr.t_off:
                    active_keys.append((nr.pitch, nr.color))
                    if midi_out and nr.pitch not in active_notes:
                        try:
                            midi_out.note_on(int(nr.pitch), int(max(1, min(nr.velocity, 127))), midi_channel)
                        except TypeError:
                            midi_out.note_on(int(nr.pitch), int(max(1, min(nr.velocity, 127))))
                        active_notes.add(nr.pitch)

        # Рисуем клавиатуру и подсветку
        draw_keyboard(screen)
        highlight_active_keys(screen, active_keys)

        # Завершение, если прошло достаточно времени после последней ноты
        if current_time > last_t_off:
            running = False

        pygame.display.flip()
        clock.tick(FPS)

    # Завершение: выключаем все ноты и закрываем MIDI
    if midi_out:
        for pitch in active_notes:
            try:
                midi_out.note_off(int(pitch), 0)
            except TypeError:
                midi_out.note_off(int(pitch), 0)
        midi_out.close()
        
    if enable_sound:
        pygame.midi.quit()

    # Останавливаем запись и получаем данные
    recorded_audio = None
    if recorder and recording_started:
        recorded_audio = recorder.stop_recording()
        print("Запись звука остановлена")

    pygame.quit()

    # Обработка записанного аудио и оценка исполнения
    if recorded_audio is not None and len(recorded_audio) > 0:
        try:
            # Создаём выделенные папки, если их нет
            recording_dir = Path("recorded_audio")
            recording_dir.mkdir(exist_ok=True)
            reference_dir = Path("reference_audio")
            reference_dir.mkdir(exist_ok=True)

            # Нормализация записанного аудио
            max_ampl = np.max(np.abs(recorded_audio))
            if max_ampl > 0.95:
                normalization_factor = 0.95 / max_ampl
                recorded_audio = recorded_audio * normalization_factor
                print(f"Запись нормализована с коэффициентом {normalization_factor:.3f}")

            # Сохраняем записанное аудио
            performance_path = recording_dir / f"{Path(midi_path).stem}_performance.wav"
            sf.write(str(performance_path), recorded_audio, recorder.sample_rate)
            print(f"Запись сохранена: {performance_path}")

            # Генерируем эталонный аудиофайл
            reference_path = reference_dir / f"{Path(midi_path).stem}_reference.wav"
            generator = ReferenceGenerator(samples_dir="samples", sample_rate=recorder.sample_rate)
            
            if generator.generate_reference(midi_path, str(reference_path)):
                print(f"Эталонный аудиофайл создан: {reference_path}")
                
                # Оцениваем исполнение
                evaluator = PerformanceEvaluator(sr=recorder.sample_rate)
                results = evaluator.evaluate(str(reference_path), str(performance_path))
                
                message = ''
                print("\n=== РЕЗУЛЬТАТЫ ОЦЕНКИ ИСПОЛНЕНИЯ ===")
                print(f"DTW расстояние: {results['dtw_distance']:.2f}")
                message += f"DTW расстояние: {results['dtw_distance']:.2f}\n"
                print(f"Оценка сходства: {results['similarity']:.3f} ({results['similarity']*100:.1f}%)")
                message += f"Оценка сходства: {results['similarity']:.3f} ({results['similarity']*100:.1f}%)\n"
                
                # Интерпретация результатов
                if results['similarity'] >= 0.78:
                    print("Отличное исполнение! 🎉")
                    message += "Отличное исполнение! 🎉\n"
                elif results['similarity'] >= 0.7:
                    print("Хорошее исполнение! 👍")
                    message += "Хорошее исполнение! 👍\n"
                elif results['similarity'] >= 0.6:
                    print("Удовлетворительное исполнение. Есть над чем поработать.")
                    message += "Удовлетворительное исполнение. Есть над чем поработать.\n"
                else:
                    print("Нужно больше практики. Попробуйте еще раз!")
                    message += "Нужно больше практики. Попробуйте еще раз!\n"
                messagebox.showwarning("=== РЕЗУЛЬТАТЫ ОЦЕНКИ ИСПОЛНЕНИЯ ===", message)
            else:
                print("Ошибка при создании эталонного аудиофайла")
                
        except Exception as e:
            print(f"Ошибка при обработке записи: {e}")
    else:
        print("Запись звука не была произведена или пуста")

if __name__ == "__main__":
    # Пример использования
    midi_file = "exercises/ёлочка.mid"  # Заменить на реальный путь к MIDI-файлу
    device_name = 'Realtek High Definition Audio (#1)'  # Заменить на нужное имя устройства
    # Если задан параметр device_name, то будет использоваться запись звука и производиться оценка исполнения
    run_visualization(midi_file, enable_sound=False, device_name=device_name)
