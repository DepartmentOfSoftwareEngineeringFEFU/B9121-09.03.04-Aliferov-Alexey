MIN_NOTE = 48
MAX_NOTE = 84
NUM_KEYS = MAX_NOTE - MIN_NOTE + 1

# Размеры клавиш
WHITE_KEY_WIDTH = 32  # ширина белой клавиши
BLACK_KEY_WIDTH = 20  # ширина черной клавиши (меньше белой)
WHITE_KEY_HEIGHT = 100  # высота белых клавиш
BLACK_KEY_HEIGHT = 65   # высота черных клавиш (короче белых)

KEY_HEIGHT = WHITE_KEY_HEIGHT  # общая высота области клавиатуры
LEAD_TIME = 4

# Паттерн белых и черных клавиш в октаве
# True = белая, False = черная
KEY_PATTERN = [True, False, True, False, True, True, False, True, False, True, False, True]
# Соответствует: C, C#, D, D#, E, F, F#, G, G#, A, A#, B

def is_white_key(midi_note):
    """Возвращает True, если нота соответствует белой клавише."""
    return KEY_PATTERN[midi_note % 12]

def is_black_key(midi_note):
    """Возвращает True, если нота соответствует черной клавише."""
    return not KEY_PATTERN[midi_note % 12]

def get_white_key_count():
    """Возвращает количество белых клавиш в диапазоне [MIN_NOTE, MAX_NOTE]."""
    count = 0
    for note in range(MIN_NOTE, MAX_NOTE + 1):
        if is_white_key(note):
            count += 1
    return count

# Подсчитываем количество белых клавиш для определения ширины экрана
WHITE_KEY_COUNT = get_white_key_count()
SCREEN_WIDTH = WHITE_KEY_COUNT * WHITE_KEY_WIDTH
SCREEN_HEIGHT = 900
