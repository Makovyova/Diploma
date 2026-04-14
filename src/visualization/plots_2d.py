"""
2D визуализация результатов навигации
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple


def plot_error_over_time(df: pd.DataFrame,
                        title: str = "Ошибка позиционирования") -> go.Figure:
    """
    График ошибки во времени
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['error'],
        mode='lines', name='Ошибка позиционирования',
        line=dict(color='red', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 0, 0, 0.1)'
    ))

    # Среднее значение
    mean_error = df['error'].mean()
    fig.add_hline(
        y=mean_error,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Средняя: {mean_error:.3f} м"
    )

    fig.update_layout(
        title=title,
        xaxis_title='Время (с)',
        yaxis_title='Ошибка (м)',
        height=400,
        showlegend=True,
        template='plotly_white'
    )

    return fig


def plot_integral_metric(df: pd.DataFrame,
                         column: str = 'integral_score') -> go.Figure:
    """
    График интегрального показателя во времени
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df[column],
        mode='lines', name='Интегральный показатель',
        line=dict(color='purple', width=2),
        fill='tozeroy',
        fillcolor='rgba(128, 0, 128, 0.1)'
    ))

    fig.update_layout(
        title='Интегральный показатель качества',
        xaxis_title='Время (с)',
        yaxis_title='Значение метрики',
        height=400,
        showlegend=True,
        template='plotly_white'
    )

    return fig


def compare_algorithms(results_dict: dict,
                      metric: str = 'error') -> go.Figure:
    """
    Сравнение ошибок нескольких алгоритмов на одном графике

    Параметры:
    -----------
    results_dict : dict
        {'Algorithm Name': DataFrame, ...}
    metric : str
        Название колонки для сравнения
    """
    fig = go.Figure()

    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

    for idx, (name, df) in enumerate(results_dict.items()):
        if metric in df.columns:
            color = colors[idx % len(colors)]
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df[metric],
                mode='lines', name=name,
                line=dict(color=color, width=2)
            ))

    fig.update_layout(
        title='Сравнение алгоритмов',
        xaxis_title='Время (с)',
        yaxis_title=metric,
        height=450,
        showlegend=True,
        template='plotly_white'
    )

    return fig


def plot_position_errors(df: pd.DataFrame) -> go.Figure:
    """
    Раздельные графики ошибок по осям X, Y, Z
    """
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=('Ошибка по X', 'Ошибка по Y', 'Ошибка по Z')
    )

    for idx, axis in enumerate(['x', 'y', 'z'], 1):
        error_col = f'est_{axis}'
        gt_col = f'gt_{axis}'

        if error_col in df.columns and gt_col in df.columns:
            error = df[error_col] - df[gt_col]
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'], y=error,
                    mode='lines', name=f'{axis.upper()}',
                    line=dict(width=2)
                ),
                row=idx, col=1
            )

    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_white'
    )

    return fig


def plot_trajectory_2d_topview(df: pd.DataFrame) -> go.Figure:
    """
    2D проекция траектории (вид сверху, XY плоскость)
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['gt_x'], y=df['gt_y'],
        mode='lines', name='Ground Truth',
        line=dict(width=3, color='green')
    ))

    fig.add_trace(go.Scatter(
        x=df['est_x'], y=df['est_y'],
        mode='lines', name='Оценка',
        line=dict(width=2, color='red', dash='dot')
    ))

    # Начальные точки
    fig.add_trace(go.Scatter(
        x=[df['gt_x'].iloc[0]], y=[df['gt_y'].iloc[0]],
        mode='markers', name='Старт',
        marker=dict(size=10, color='green', symbol='circle')
    ))

    # Конечные точки
    fig.add_trace(go.Scatter(
        x=[df['gt_x'].iloc[-1]], y=[df['gt_y'].iloc[-1]],
        mode='markers', name='Финиш',
        marker=dict(size=10, color='green', symbol='x')
    ))

    fig.update_layout(
        title='Траектория (вид сверху)',
        xaxis_title='X (м)',
        yaxis_title='Y (м)',
        height=500,
        showlegend=True,
        template='plotly_white',
        scaleanchor="x",
        scaleratio=1
    )

    return fig


def plot_cumulative_error(df: pd.DataFrame) -> go.Figure:
    """
    Накопленная ошибка (кумулятивная сумма)
    """
    cumulative = np.cumsum(df['error'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=cumulative,
        mode='lines', name='Накопленная ошибка',
        line=dict(color='darkred', width=3)
    ))

    fig.update_layout(
        title='Накопленная ошибка',
        xaxis_title='Время (с)',
        yaxis_title='Суммарная ошибка (м)',
        height=400,
        template='plotly_white'
    )

    return fig
