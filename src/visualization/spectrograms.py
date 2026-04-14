"""
Визуализация спектрального анализа (спектрограммы, heatmap)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal
from typing import Optional, Tuple


def plot_spectrum_psd(frequencies: np.ndarray,
                     psd: np.ndarray,
                     title: str = "Спектральная плотность мощности") -> go.Figure:
    """
    График спектральной плотности мощности (PSD)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frequencies, y=10 * np.log10(psd + 1e-10),
        mode='lines', name='PSD',
        line=dict(color='orange', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))

    # Линии-разделители диапазонов
    fig.add_vline(x=0.5, line_dash="dash", line_color="blue", opacity=0.5,
                 annotation_text="Дрейф")
    fig.add_vline(x=5.0, line_dash="dash", line_color="red", opacity=0.5,
                 annotation_text="Вибрации")

    fig.update_layout(
        title=title,
        xaxis_title='Частота (Гц)',
        yaxis_title='Мощность (dB/Гц)',
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def plot_fft_magnitude(frequencies: np.ndarray,
                      magnitude: np.ndarray) -> go.Figure:
    """
    Амплитудный спектр FFT
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frequencies, y=magnitude,
        mode='lines', name='FFT',
        line=dict(color='blue', width=2)
    ))

    fig.update_layout(
        title='Амплитудный спектр FFT',
        xaxis_title='Частота (Гц)',
        yaxis_title='Амплитуда',
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def plot_spectrogram(f: np.ndarray, t: np.ndarray, Sxx: np.ndarray,
                    title: str = "Спектрограмма ошибки") -> go.Figure:
    """
    Спектрограмма (heatmap время-частота-мощность)
    """
    # Конвертация в dB
    Sxx_db = 10 * np.log10(Sxx + 1e-10)

    fig = go.Figure(data=go.Heatmap(
        z=Sxx_db,
        x=t,
        y=f,
        colorscale='YlOrRd',
        colorbar=dict(title='Мощность (dB)'),
        hovertemplate='Время: %{x:.2f} с<br>Частота: %{y:.2f} Гц<br>Мощность: %{z:.1f} dB<extra></extra>'
    ))

    fig.update_layout(
        title=title,
        xaxis_title='Время (с)',
        yaxis_title='Частота (Гц)',
        height=450,
        template='plotly_white'
    )

    return fig


def plot_dominant_frequencies(frequencies: np.ndarray,
                             psd: np.ndarray,
                             dominant_freqs: list) -> go.Figure:
    """
    График PSD с выделенными доминирующими частотами
    """
    fig = go.Figure()

    # Основной спектр
    fig.add_trace(go.Scatter(
        x=frequencies, y=10 * np.log10(psd + 1e-10),
        mode='lines', name='PSD',
        line=dict(color='gray', width=2)
    ))

    # Доминирующие частоты
    for idx, freq_info in enumerate(dominant_freqs):
        freq = freq_info['frequency']
        power = freq_info['power_db']

        fig.add_vline(
            x=freq,
            line_dash="dash",
            line_width=2,
            line_color='red'
        )

        fig.add_annotation(
            x=freq,
            y=power,
            text=f"{freq:.2f} Гц",
            showarrow=True,
            arrowhead=2,
            bgcolor='rgba(255, 0, 0, 0.1)'
        )

    fig.update_layout(
        title='Доминирующие частоты в спектре ошибок',
        xaxis_title='Частота (Гц)',
        yaxis_title='Мощность (dB/Гц)',
        height=400,
        template='plotly_white',
        showlegend=False
    )

    return fig


def plot_error_classification(classification: dict) -> go.Figure:
    """
    Визуализация классификации типов ошибок (pie chart)
    """
    fig = go.Figure(data=[go.Pie(
        labels=['Низкие частоты (дрейф)', 'Средние частоты', 'Высокие частоты (вибрации)'],
        values=[
            classification['low_freq_ratio'],
            classification['mid_freq_ratio'],
            classification['high_freq_ratio']
        ],
        hole=.3,
        marker_colors=['blue', 'green', 'red']
    )])

    fig.update_layout(
        title=f"Классификация ошибок: {classification['dominant_type']}",
        annotations=[dict(
            text=classification['interpretation'][:50] + "...",
            x=0.5, y=0.5,
            font_size=12,
            showarrow=False
        )],
        height=400
    )

    return fig


def create_fft_dashboard(df: pd.DataFrame,
                        sampling_rate: float = 30.0) -> go.Figure:
    """
    Комплексный дашборд FFT анализа
    """
    from src.metrics.fft_analysis import FFTAnalysis, compute_spectrogram

    analyzer = FFTAnalysis(sampling_rate=sampling_rate)

    # Расчёты
    error_signal = signal.detrend(df['error'].values)
    frequencies, psd = analyzer.compute_welch_psd(error_signal)
    dominant_freqs = analyzer.find_dominant_frequencies(frequencies, psd)
    classification = analyzer.classify_error_type(frequencies, psd)

    f_spec, t_spec, Sxx = compute_spectrogram(df, sampling_rate)

    # Создание subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Ошибка во времени',
            'Спектральная плотность мощности',
            'Спектрограмма',
            'Классификация ошибок'
        ),
        specs=[[{"type": "xy"}, {"type": "xy"}],
               [{"type": "heatmap"}, {"type": "domain"}]]
    )

    # 1. Ошибка во времени
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'], y=df['error'],
            mode='lines', name='Ошибка',
            line=dict(color='red', width=2)
        ),
        row=1, col=1
    )

    # 2. PSD
    fig.add_trace(
        go.Scatter(
            x=frequencies, y=10 * np.log10(psd + 1e-10),
            mode='lines', name='PSD',
            line=dict(color='orange', width=2)
        ),
        row=1, col=2
    )

    # 3. Спектрограмма
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    fig.add_trace(
        go.Heatmap(
            z=Sxx_db, x=t_spec, y=f_spec,
            colorscale='YlOrRd',
            showscale=False
        ),
        row=2, col=1
    )

    # 4. Классификация
    fig.add_trace(
        go.Pie(
            labels=['Дрейф', 'Средние', 'Вибрации'],
            values=[
                classification['low_freq_ratio'],
                classification['mid_freq_ratio'],
                classification['high_freq_ratio']
            ],
            hole=.3,
            marker_colors=['blue', 'green', 'red'],
            showlegend=False
        ),
        row=2, col=2
    )

    fig.update_layout(
        title='Полный спектральный анализ ошибок',
        height=800,
        template='plotly_white'
    )

    return fig
