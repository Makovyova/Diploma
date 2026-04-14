"""
Спектральный анализ ошибок навигации (FFT)

Используется для выявления характерных частот сбоев:
- Низкие частоты (<0.5 Гц) → дрейф оценки
- Высокие частоты (>5 Гц) → вибрации, джиттер дескрипторов
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, Tuple, Optional, List
import json


class FFTAnalysis:
    """
    Спектральный анализ ошибок визуальной навигации
    """

    def __init__(self, sampling_rate: float = 30.0):
        """
        Параметры:
        -----------
        sampling_rate : float
            Частота дискретизации (Гц), обычно FPS камеры
        """
        self.sampling_rate = sampling_rate

    def compute_welch_psd(self, error_signal: np.ndarray,
                         nperseg: int = 128) -> Tuple[np.ndarray, np.ndarray]:
        """
        Оценка спектральной плотности мощности (PSD) методом Уэлча

        Возвращает:
        -----------
        frequencies : np.ndarray
            Частоты (Гц)
        psd : np.ndarray
            Спектральная плотность мощности
        """
        frequencies, psd = signal.welch(
            error_signal,
            fs=self.sampling_rate,
            nperseg=min(nperseg, len(error_signal)),
            scaling='density'
        )
        return frequencies, psd

    def compute_fft(self, error_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Быстрое преобразование Фурье

        Возвращает:
        -----------
        frequencies : np.ndarray
            Частоты (Гц)
        magnitude : np.ndarray
            Амплитудный спектр
        """
        n = len(error_signal)
        yf = fft(error_signal)
        xf = fftfreq(n, 1 / self.sampling_rate)[:n//2]

        magnitude = 2.0/n * np.abs(yf[:n//2])
        return xf, magnitude

    def find_dominant_frequencies(self, frequencies: np.ndarray,
                                 psd: np.ndarray,
                                 top_n: int = 3,
                                 min_freq: float = 0.1,
                                 max_freq: float = 15.0) -> List[Dict]:
        """
        Поиск доминирующих частот в спектре

        Возвращает:
        -----------
        List[Dict] с информацией о доминирующих частотах
        """
        # Фильтрация по диапазону
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        freq_filtered = frequencies[mask]
        psd_filtered = psd[mask]

        # Поиск пиков
        peak_indices = signal.find_peaks(psd_filtered, height=np.max(psd_filtered)*0.1)[0]

        if len(peak_indices) == 0:
            return []

        # Сортировка по мощности
        peak_powers = psd_filtered[peak_indices]
        sorted_indices = np.argsort(peak_powers)[::-1]

        results = []
        for idx in sorted_indices[:top_n]:
            peak_idx = peak_indices[idx]
            results.append({
                'frequency': float(freq_filtered[peak_idx]),
                'power': float(psd_filtered[peak_idx]),
                'power_db': float(10 * np.log10(psd_filtered[peak_idx] + 1e-10))
            })

        return results

    def classify_error_type(self, frequencies: np.ndarray,
                           psd: np.ndarray) -> Dict:
        """
        Классификация типа ошибок по спектру

        Возвращает:
        -----------
        Dict с классификацией доминирующих ошибок
        """
        # Диапазоны частот
        low_freq_mask = frequencies < 0.5
        mid_freq_mask = (frequencies >= 0.5) & (frequencies < 5.0)
        high_freq_mask = frequencies >= 5.0

        # Энергия в диапазонах
        low_energy = np.trapz(psd[low_freq_mask], frequencies[low_freq_mask])
        mid_energy = np.trapz(psd[mid_freq_mask], frequencies[mid_freq_mask])
        high_energy = np.trapz(psd[high_freq_mask], frequencies[high_freq_mask])

        total_energy = low_energy + mid_energy + high_energy

        # Доли энергии
        low_ratio = low_energy / total_energy if total_energy > 0 else 0
        mid_ratio = mid_energy / total_energy if total_energy > 0 else 0
        high_ratio = high_energy / total_energy if total_energy > 0 else 0

        # Классификация
        dominant_type = "unknown"
        if low_ratio > 0.5:
            dominant_type = "drift"  # Дрейф оценки
        elif high_ratio > 0.4:
            dominant_type = "vibration"  # Вибрации/джиттер
        elif mid_ratio > 0.4:
            dominant_type = "periodic"  # Периодические ошибки

        return {
            'dominant_type': dominant_type,
            'low_freq_ratio': float(low_ratio),
            'mid_freq_ratio': float(mid_ratio),
            'high_freq_ratio': float(high_ratio),
            'total_energy': float(total_energy),
            'interpretation': self._interpret_type(dominant_type)
        }

    def _interpret_type(self, error_type: str) -> str:
        """Интерпретация типа ошибок для отчёта"""
        interpretations = {
            'drift': "Низкочастотный дрейф оценки. Возможные причины: калибровка, масштабная неопределённость",
            'vibration': "Высокочастотные вибрации. Возможные причины: джиттер дескрипторов, быстрое движение камеры",
            'periodic': "Периодические ошибки. Возможные причины: циклические паттерны в окружении",
            'unknown': "Не удалось классифицировать тип ошибок"
        }
        return interpretations.get(error_type, "Неизвестный тип")

    def analyze(self, df: pd.DataFrame,
               error_column: str = 'error') -> Dict:
        """
        Полный спектральный анализ ошибок

        Параметры:
        -----------
        df : pd.DataFrame
            DataFrame с колонкой ошибки
        error_column : str
            Название колонки с ошибкой

        Возвращает:
        -----------
        Dict с результатами анализа
        """
        if error_column not in df.columns:
            raise ValueError(f"Колонка '{error_column}' не найдена в DataFrame")

        error_signal = df[error_column].values

        # Удаление тренда
        error_signal = signal.detrend(error_signal)

        # Welch PSD
        frequencies, psd = self.compute_welch_psd(error_signal)

        # FFT
        fft_freq, fft_mag = self.compute_fft(error_signal)

        # Доминирующие частоты
        dominant_freqs = self.find_dominant_frequencies(frequencies, psd)

        # Классификация
        classification = self.classify_error_type(frequencies, psd)

        return {
            'frequencies': frequencies,
            'psd': psd,
            'fft_frequencies': fft_freq,
            'fft_magnitude': fft_mag,
            'dominant_frequencies': dominant_freqs,
            'classification': classification,
            'sampling_rate': self.sampling_rate
        }

    def analyze_axes_separately(self, df: pd.DataFrame) -> Dict:
        """
        Раздельный анализ ошибок по осям X, Y, Z
        """
        results = {}

        for axis in ['x', 'y', 'z']:
            error_col = f'est_{axis}'
            gt_col = f'gt_{axis}'

            if error_col in df.columns and gt_col in df.columns:
                axis_error = (df[error_col] - df[gt_col]).values
                axis_error = signal.detrend(axis_error)

                frequencies, psd = self.compute_welch_psd(axis_error)
                classification = self.classify_error_type(frequencies, psd)

                results[axis] = {
                    'frequencies': frequencies,
                    'psd': psd,
                    'classification': classification
                }

        return results


def compute_spectrogram(df: pd.DataFrame,
                       sampling_rate: float = 30.0,
                       window_size: int = 256,
                       step_size: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Расчёт спектрограммы (время-частота-мощность) для визуализации heatmap

    Возвращает:
    -----------
    f : np.ndarray
        Частоты (Гц)
    t : np.ndarray
        Время (с)
    Sxx : np.ndarray
        Спектрограмма (мощность)
    """
    if 'error' not in df.columns:
        raise ValueError("DataFrame должен содержать колонку 'error'")

    error_signal = df['error'].values
    error_signal = signal.detrend(error_signal)

    f, t, Sxx = signal.spectrogram(
        error_signal,
        fs=sampling_rate,
        nperseg=window_size,
        noverlap=window_size - step_size,
        scaling='density'
    )

    return f, t, Sxx


def generate_fft_report(df: pd.DataFrame,
                       sampling_rate: float = 30.0,
                       output_path: Optional[str] = None) -> Dict:
    """
    Генерация полного отчёта спектрального анализа

    Возвращает:
    -----------
    Dict с метриками и классификацией
    """
    analyzer = FFTAnalysis(sampling_rate=sampling_rate)

    # Общий анализ
    overall = analyzer.analyze(df, error_column='error')

    # Анализ по осям
    axes = analyzer.analyze_axes_separately(df)

    report = {
        'overall': {
            'dominant_frequencies': overall['dominant_frequencies'],
            'classification': overall['classification']
        },
        'axes': {axis: data['classification'] for axis, data in axes.items()},
        'sampling_rate': sampling_rate
    }

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    return report
