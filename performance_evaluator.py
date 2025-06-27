import numpy as np
import librosa
from scipy.spatial.distance import cdist


class PerformanceEvaluator:
    """
    Класс для оценки корректности исполнения упражнения на фортепиано.
    """

    def __init__(self, sr: int = 22050, hop_length: int = 512, alpha: float = 800.0):
        """
        Инициализация параметров оценщика.

        Args:
            sr (int): Целевая частота дискретизации при загрузке аудио (Гц).
            hop_length (int): Шаг окна для STFT/хрома.
            alpha (float): Параметр масштабирования для экспоненциальной нормализации.
        """
        self.sr = sr
        self.hop_length = hop_length
        self.alpha = alpha

        # Сюда можно сохранять загруженные аудиодорожки (если нужно)
        self._y_ref = None
        self._y_perf = None

    def _load_audio(self, path: str) -> np.ndarray:
        """
        Загрузка аудиофайла с ресемплированием до self.sr.

        Args:
            path (str): Путь к аудиофайлу (.wav, .mp3 и т.п.).

        Returns:
            np.ndarray: Одномерный массив аудиосигнала.
        """
        y, _ = librosa.load(path, sr=self.sr)
        return y

    def _extract_chroma(self, y: np.ndarray) -> np.ndarray:
        """
        Извлечение хрома-признаков (12-полосной хромаграммы) из аудиосигнала.

        Args:
            y (np.ndarray): Временной ряд аудиосигнала.

        Returns:
            np.ndarray: Хрома-матрица размера (12, T), где T — число фреймов.
        """
        S = np.abs(librosa.stft(y, hop_length=self.hop_length))
        chroma = librosa.feature.chroma_stft(S=S, sr=self.sr, hop_length=self.hop_length)
        return chroma

    def _compute_dtw(self, chroma_ref: np.ndarray, chroma_perf: np.ndarray) -> tuple:
        """
        Вычисление DTW-выравнивания между двумя хрома-матрицами.

        Args:
            chroma_ref (np.ndarray): Хрома-матрица эталона (12, T_ref).
            chroma_perf (np.ndarray): Хрома-матрица исполнения (12, T_perf).

        Returns:
            tuple: (dist, cost_matrix) - итоговая стоимость и матрица затрат.
        """
        D = cdist(chroma_ref.T, chroma_perf.T, metric='euclidean')
        cost_matrix, _ = librosa.sequence.dtw(C=D)
        dist = cost_matrix[-1, -1]
        return dist, cost_matrix

    def _compute_similarity(self, dtw_dist: float) -> float:
        """
        Преобразование DTW-расстояния в нормированное значение сходства [0, 1].

        Args:
            dtw_dist (float): Итоговое DTW-расстояние.

        Returns:
            float: Оценка сходства (1.0 = идеально, ближе к 0 = сильно различаются).
        """
        return float(np.exp(-dtw_dist / self.alpha))

    def evaluate(self, ref_path: str, perf_path: str) -> dict:
        """
        Базовая оценка на основе хрома-признаков и DTW.

        Args:
            ref_path (str): Путь к аудиофайлу-эталону.
            perf_path (str): Путь к аудиофайлу исполнения.

        Returns:
            dict: Словарь с результатами базовой оценки.
        """
        self._y_ref = self._load_audio(ref_path)
        self._y_perf = self._load_audio(perf_path)

        chroma_ref = self._extract_chroma(self._y_ref)
        chroma_perf = self._extract_chroma(self._y_perf)

        dist, cost_matrix = self._compute_dtw(chroma_ref, chroma_perf)
        sim = self._compute_similarity(dist)

        return {
            'dtw_distance': float(dist),
            'similarity': sim,
            'cost_matrix': cost_matrix
        }

# Тестирование модуля
if __name__ == "__main__":
    # Пример использования
    evaluator = PerformanceEvaluator()
    # Пути заменить на нужные
    results = evaluator.evaluate("reference_audio/example_reference.wav", "recorded_audio/example_performance.wav")

    print("DTW Distance:", results['dtw_distance'])
    print("Similarity Score:", results['similarity'])
