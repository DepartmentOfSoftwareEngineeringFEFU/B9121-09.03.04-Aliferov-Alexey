import os
import shutil
from pathlib import Path
import mido
import sounddevice as sd
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from const import MIN_NOTE, MAX_NOTE
from visualizer import run_visualization
# Диапазон MIDI-нот от C4 (60) до C7 (96) включительно


# Пути к папкам (относительно этого файла)
PROJECT_DIR = Path(__file__).parent.resolve()
EXERCISES_DIR = PROJECT_DIR / "exercises"
SAMPLES_DIR = PROJECT_DIR / "samples"


class PianoTrainerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Тренажёр пианино")
        self.resizable(False, False)
        self.geometry("550x500")

        # Создаём папки exercises и samples (если их нет)
        EXERCISES_DIR.mkdir(exist_ok=True)
        SAMPLES_DIR.mkdir(exist_ok=True)

        # Будем хранить список устройств и выбираемый девайс
        self.input_devices = self.query_input_devices()
        self.selected_device = tk.StringVar(value=self.input_devices[0] if self.input_devices else "")
        self.sound_enabled = tk.BooleanVar(value=False)  # по умолчанию звук выключен
        # Инициализируем UI
        self.create_widgets()
        self.refresh_exercise_list()
        self.update_samples_status()

    def query_input_devices(self) -> list[str]:
        """
        Возвращает список имён устройств, у которых есть входные каналы (микрофоны).
        """
        devices = sd.query_devices()
        input_list = []
        for idx, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                # Формат: "Имя (индекс)"
                name = f"{dev['name']} (#{idx})"
                input_list.append(name)
        return input_list

    def create_widgets(self):
        frame_device = tk.LabelFrame(self, text="Выбор устройства записи", padx=10, pady=5)
        frame_device.pack(fill="x", padx=10, pady=(10, 5))

        if not self.input_devices:
            lbl_no_dev = tk.Label(frame_device, text="Нет доступных устройств ввода (микрофонов).")
            lbl_no_dev.pack(anchor="w")
        else:
            lbl_dev = tk.Label(frame_device, text="Выберите микрофон:")
            lbl_dev.pack(side="left", padx=(0, 5))

            self.combo_devices = ttk.Combobox(
                frame_device,
                values=self.input_devices,
                state="readonly",
                textvariable=self.selected_device,
                width=50
            )
            self.combo_devices.pack(side="left", fill="x", expand=True)

        frame_sound = tk.LabelFrame(self, text="Настройки звука", padx=10, pady=5)
        frame_sound.pack(fill="x", padx=10, pady=(5, 5))

        self.sound_checkbox = tk.Checkbutton(
            frame_sound,
            text="Включить проигрывание звука во время упражнения",
            variable=self.sound_enabled
        )
        self.sound_checkbox.pack(anchor="w")

        lbl = tk.Label(self, text="Список упражнений:")
        lbl.pack(anchor="w", padx=10, pady=(10, 0))

        self.listbox = tk.Listbox(self, width=70, height=10)
        self.listbox.pack(padx=10, pady=(0, 5))


        frame_buttons = tk.Frame(self)
        frame_buttons.pack(fill="x", padx=10, pady=(5, 5))

        btn_add = tk.Button(frame_buttons, text="Добавить упражнение", command=self.on_add_exercise)
        btn_add.pack(side="left")

        btn_delete = tk.Button(frame_buttons, text="Удалить упражнение", command=self.on_delete_exercise)
        btn_delete.pack(side="left", padx=(5, 0))

        self.btn_run = tk.Button(frame_buttons, text="Запустить упражнение", command=self.on_run_exercise)
        self.btn_run.pack(side="left", padx=(5, 0))
        self.btn_run.configure(state="disabled")


        frame_status = tk.LabelFrame(self, text="Состояние семплов", padx=10, pady=5)
        frame_status.pack(fill="both", padx=10, pady=(5, 5))

        self.samples_status = tk.Text(
            frame_status,
            height=4,
            wrap="word",
            bg=self.cget("bg"),
            relief="flat",
            state="disabled"
        )
        self.samples_status.pack(fill="both", padx=5, pady=5)

    def validate_midi(self, path: Path) -> bool:
        """
        Проверяет, что MIDI-файл содержит только ноты в диапазоне [MIN_NOTE, MAX_NOTE].
        """
        try:
            mid = mido.MidiFile(str(path))
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось прочитать MIDI:\n{e}")
            return False

        for track in mid.tracks:
            for msg in track:
                if msg.type in ("note_on", "note_off"):
                    if msg.note < MIN_NOTE or msg.note > MAX_NOTE:
                        return False
        return True

    def get_valid_exercises(self) -> list[str]:
        """
        Возвращает имена всех корректных MIDI-файлов из папки exercises.
        """
        result = []
        for entry in EXERCISES_DIR.iterdir():
            if entry.is_file() and entry.suffix.lower() in (".mid", ".midi"):
                if self.validate_midi(entry):
                    result.append(entry.name)
        return sorted(result)

    def refresh_exercise_list(self):
        """
        Обновляет Listbox в соответствии с текущим содержимым папки exercises.
        """
        self.listbox.delete(0, tk.END)
        for name in self.get_valid_exercises():
            self.listbox.insert(tk.END, name)

    def on_add_exercise(self):
        """
        Обработчик кнопки «Добавить упражнение» — выбор и валидация MIDI-файла.
        """
        filepath = filedialog.askopenfilename(
            title="Выберите MIDI-файл",
            filetypes=[("MIDI файлы", "*.mid *.midi")],
        )
        if not filepath:
            return

        src = Path(filepath)
        if src.suffix.lower() not in (".mid", ".midi"):
            messagebox.showwarning("Неверный формат", "Выбранный файл не является MIDI.")
            return

        if not self.validate_midi(src):
            messagebox.showwarning(
                "Невалидный диапазон",
                f"MIDI-файл содержит ноты вне диапазона (номера нот {MIN_NOTE}–{MAX_NOTE})."
            )
            return

        dst = EXERCISES_DIR / src.name
        try:
            shutil.copy(src, dst)
        except Exception as e:
            messagebox.showerror("Ошибка копирования", f"Не удалось скопировать файл:\n{e}")
            return

        self.refresh_exercise_list()

    def on_delete_exercise(self):
        """
        Обработчик кнопки «Удалить упражнение» — удаляет выбранный файл из exercises.
        """
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Удаление", "Пожалуйста, выберите упражнение для удаления.")
            return

        index = selection[0]
        midi_name = self.listbox.get(index)
        full_path = EXERCISES_DIR / midi_name

        answer = messagebox.askyesno("Подтверждение удаления",
                                     f"Действительно удалить '{midi_name}' из списка упражнений?")
        if not answer:
            return

        try:
            os.remove(full_path)
        except Exception as e:
            messagebox.showerror("Ошибка удаления", f"Не удалось удалить файл:\n{e}")
            return

        self.refresh_exercise_list()

    def get_expected_sample_names(self) -> list[str]:
        """
        Возвращает список требуемых сэмплов от MIN_NOTE до MAX_NOTE включительно.
        """
        return [str(note) for note in range(MIN_NOTE, MAX_NOTE + 1)]

    def get_missing_samples(self) -> list[str]:
        """
        Возвращает список ожидаемых, но отсутствующих нот-сэмплов (по именам файлов).
        """
        expected = set(self.get_expected_sample_names())
        found = set()

        if SAMPLES_DIR.exists():
            for file in SAMPLES_DIR.iterdir():
                if file.is_file() and file.suffix.lower() == ".wav":
                    name = file.stem.lower()
                    if name.isdigit() and MIN_NOTE <= int(name) <= MAX_NOTE:
                        found.add(name)

        # Отсортируем по порядку чисел
        return sorted(expected - found)

    def update_samples_status(self):
        """
        Обновляет многострочное поле с отсутствующими сэмплами и состояние кнопки запуска.
        """
        missing = self.get_missing_samples()

        self.samples_status.config(state="normal")
        self.samples_status.delete("1.0", tk.END)

        if not missing:
            self.samples_status.insert(tk.END, "Семплы: присутствуют ✔")
            self.btn_run.config(state="normal")
        else:
            text = "Семплы отсутствуют:\n" + ", ".join(missing)
            self.samples_status.insert(tk.END, text)
            self.btn_run.config(state="disabled")

        self.samples_status.config(state="disabled")

    def on_run_exercise(self):
        """
        Обработчик кнопки «Запустить упражнение» — запускает логику выполнения.
        """
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showinfo("Выбор упражнения", "Пожалуйста, выберите упражнение из списка.")
            return

        midi_name = self.listbox.get(selection[0])
        device = self.selected_device.get()
        sound_enabled = self.sound_enabled.get()

        # Проверяем, что устройство выбрано
        if not device:
            messagebox.showwarning("Устройство не выбрано", "Пожалуйста, выберите устройство записи.")
            return

        print(f"Запуск упражнения: {midi_name}")
        print(f"Устройство записи: {device}")
        print(f"Звук: {'включен' if sound_enabled else 'выключен'}")
        
        # Запускаем визуализацию с записью звука
        try:
            run_visualization(f'exercises/{midi_name}', enable_sound=sound_enabled, device_name=device)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Произошла ошибка при запуске упражнения:\n{e}")

if __name__ == "__main__":
    app = PianoTrainerApp()
    app.mainloop()
