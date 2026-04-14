"""
Расчёт интегрального показателя качества навигации

Авторская метрика: J = α·ATE + β·jitter + γ·latency
Учитывает точность, гладкость траектории и вычислительные ресурсы
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


class IntegralMetric:
    """
    Интегральный показатель качества визуальной навигации

    Параметры:
    -----------
    alpha : float
        Вес точности (ATE)
    beta : float
        Вес гладкости (jitter/гладкость траектории)
    gamma : float
        Вес вычислительных ресурсов (латентность)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        # Нормализация весов
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total

    def calculate_ate(self, gt_positions: np.ndarray,
                     est_positions: np.ndarray) -> float:
        """
        Absolute Trajectory Error (ATE)
        Среднеквадратичная ошибка позиционирования
        """
        errors = np.linalg.norm(est_positions - gt_positions, axis=1)
        return np.sqrt(np.mean(errors**2))

    def calculate_rpe(self, gt_positions: np.ndarray,
                     est_positions: np.ndarray,
                     delta: int = 1) -> float:
        """
        Relative Pose Error (RPE)
        Относительная ошибка на шаге delta
        """
        gt_diff = np.diff(gt_positions, axis=0)[::delta]
        est_diff = np.diff(est_positions, axis=0)[::delta]
        rel_errors = np.linalg.norm(est_diff - gt_diff, axis=1)
        return np.mean(rel_errors)

    def calculate_jitter(self, est_positions: np.ndarray) -> float:
        """
        Оценка джиттера/гладкости траектории
        Через вторую производную (ускорение)
        """
        # Вторая производная позиции = ускорение
        acceleration = np.gradient(np.gradient(est_positions, axis=0), axis=0)
        jitter = np.mean(np.linalg.norm(acceleration, axis=1))
        return jitter

    def calculate_smoothness(self, est_positions: np.ndarray) -> float:
        """
        Альтернативная оценка гладкости
        Через дисперсию первой производной (скорости)
        """
        velocity = np.gradient(est_positions, axis=0)
        # Меньшая дисперсия = более плавная траектория
        smoothness = np.mean(np.var(velocity, axis=0))
        return smoothness

    def calculate_latency_penalty(self, processing_times: Optional[np.ndarray] = None,
                                  target_fps: float = 30.0) -> float:
        """
        Штраф за вычислительную сложность
        Если fps ниже target, увеличивается штраф
        """
        if processing_times is None:
            return 0.0  # Нет данных — не штрафуем

        actual_fps = 1.0 / np.mean(processing_times)
        if actual_fps >= target_fps:
            return 0.0

        # Линейный штраф за недостаток FPS
        penalty = (target_fps - actual_fps) / target_fps
        return penalty

    def calculate_integral_metric(self, df: pd.DataFrame,
                                  processing_times: Optional[np.ndarray] = None,
                                  target_fps: float = 30.0) -> Tuple[float, Dict]:
        """
        Расчёт интегрального показателя качества

        J = α·ATE_norm + β·jitter_norm + γ·latency_penalty

        Возвращает:
        -----------
        J : float
            Интегральный показатель (меньше = лучше)
        components : Dict
            Отдельные компоненты для анализа
        """
        # Извлечение позиций
        gt_positions = df[['gt_x', 'gt_y', 'gt_z']].values
        est_positions = df[['est_x', 'est_y', 'est_z']].values

        # 1. ATE
        ate = self.calculate_ate(gt_positions, est_positions)

        # 2. Jitter
        jitter = self.calculate_jitter(est_positions)

        # 3. Latency penalty
        latency_penalty = self.calculate_latency_penalty(processing_times, target_fps)

        # Нормализация компонент (приведение к [0, 1])
        # Для ATE: типичные значения 0.1 - 5.0 м
        ate_norm = min(ate / 5.0, 1.0)

        # Для jitter: типичные значения 0.01 - 2.0
        jitter_norm = min(jitter / 2.0, 1.0)

        # Latency уже в [0, 1]

        # Итоговая метрика
        J = (self.alpha * ate_norm +
             self.beta * jitter_norm +
             self.gamma * latency_penalty)

        components = {
            'ate': ate,
            'ate_normalized': ate_norm,
            'jitter': jitter,
            'jitter_normalized': jitter_norm,
            'latency_penalty': latency_penalty,
            'weights': {
                'alpha': self.alpha,
                'beta': self.beta,
                'gamma': self.gamma
            }
        }

        return J, components

    def calculate_for_dataframe(self, df: pd.DataFrame) -> pd.Series:
        """
        Покомпонентный расчёт для каждой строки DataFrame
        Добавляет колонки: error, smoothness, integral_score
        """
        # Ошибка позиционирования
        error = np.sqrt(
            (df['est_x'] - df['gt_x'])**2 +
            (df['est_y'] - df['gt_y'])**2 +
            (df['est_z'] - df['gt_z'])**2
        )

        # Гладкость (вторая производная)
        smoothness = np.abs(np.gradient(np.gradient(df['est_x']))) + \
                     np.abs(np.gradient(np.gradient(df['est_y']))) + \
                     np.abs(np.gradient(np.gradient(df['est_z'])))

        # Интегральный скоринг по точкам
        integral_score = (self.alpha * error +
                         self.beta * smoothness +
                         self.gamma * 0.0)  # Latency пока 0

        return pd.DataFrame({
            'error': error,
            'smoothness': smoothness,
            'integral_score': integral_score
        })


def compute_metrics_for_experiment(df: pd.DataFrame,
                                   alpha: float = 0.5,
                                   beta: float = 0.3,
                                   gamma: float = 0.2,
                                   processing_times: Optional[np.ndarray] = None,
                                   target_fps: float = 30.0) -> Dict:
    """
    Обёртка для быстрого расчёта всех метрик

    Параметры:
    -----------
    df : pd.DataFrame
        Должен содержать колонки: gt_x, gt_y, gt_z, est_x, est_y, est_z
    alpha, beta, gamma : float
        Веса для интегральной метрики
    processing_times : np.ndarray
        Время обработки каждого кадра (сек)
    target_fps : float
        Целевая частота обновления

    Возвращает:
    -----------
    Dict с метриками: ATE, RPE, Jitter, Integral Metric
    """
    metric = IntegralMetric(alpha=alpha, beta=beta, gamma=gamma)
    J, components = metric.calculate_integral_metric(
        df, processing_times, target_fps
    )

    gt_positions = df[['gt_x', 'gt_y', 'gt_z']].values
    est_positions = df[['est_x', 'est_y', 'est_z']].values

    results = {
        'ate': components['ate'],
        'rpe': metric.calculate_rpe(gt_positions, est_positions),
        'jitter': components['jitter'],
        'smoothness': metric.calculate_smoothness(est_positions),
        'integral_metric': J,
        'max_error': float(np.max(components['ate'])),
        'components': components
    }

    return results
