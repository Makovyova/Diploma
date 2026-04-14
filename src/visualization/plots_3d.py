"""
3D визуализация траекторий навигации
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple, Dict


def plot_trajectory_3d(df: pd.DataFrame,
                      title: str = "3D Траектория полёта") -> go.Figure:
    """
    3D визуализация траектории (GT vs оценка)
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=df['gt_x'], y=df['gt_y'], z=df['gt_z'],
            mode='lines', name='Ground Truth',
            line=dict(width=4, color='green'),
            opacity=0.8
        ),
        go.Scatter3d(
            x=df['est_x'], y=df['est_y'], z=df['est_z'],
            mode='lines', name='Оценка алгоритма',
            line=dict(width=3, color='red', dash='dot'),
            opacity=0.8
        )
    ])

    # Точка старта
    fig.add_trace(go.Scatter3d(
        x=[df['gt_x'].iloc[0]], y=[df['gt_y'].iloc[0]], z=[df['gt_z'].iloc[0]],
        mode='markers+text', name='Старт',
        marker=dict(size=5, color='green'),
        text=['Старт'],
        textposition='top center'
    ))

    # Точка финиша
    fig.add_trace(go.Scatter3d(
        x=[df['gt_x'].iloc[-1]], y=[df['gt_y'].iloc[-1]], z=[df['gt_z'].iloc[-1]],
        mode='markers+text', name='Финиш',
        marker=dict(size=5, color='red'),
        text=['Финиш'],
        textposition='bottom center'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (м)',
            yaxis_title='Y (м)',
            zaxis_title='Z (м)',
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )

    return fig


def plot_trajectory_with_error_coloring(df: pd.DataFrame) -> go.Figure:
    """
    3D траектория с цветовой кодировкой ошибки
    (зелёный = малая ошибка, красный = большая)
    """
    error = df['error'].values

    # Нормализация ошибки для цветовой шкалы
    error_norm = (error - error.min()) / (error.max() - error.min() + 1e-10)

    fig = go.Figure()

    # Траектория с цветовой кодировкой
    fig.add_trace(go.Scatter3d(
        x=df['est_x'], y=df['est_y'], z=df['est_z'],
        mode='lines', name='Траектория',
        line=dict(
            width=4,
            color=error_norm,
            colorscale='RdYlGn',
            colorbar=dict(title='Ошибка (м)')
        ),
        opacity=0.9
    ))

    # GT траектория
    fig.add_trace(go.Scatter3d(
        x=df['gt_x'], y=df['gt_y'], z=df['gt_z'],
        mode='lines', name='Ground Truth',
        line=dict(width=3, color='gray', dash='dash'),
        opacity=0.5
    ))

    fig.update_layout(
        title='3D траектория с картой ошибок',
        scene=dict(
            xaxis_title='X (м)',
            yaxis_title='Y (м)',
            zaxis_title='Z (м)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return fig


def plot_multiple_trajectories_3d(traj_dict: dict) -> go.Figure:
    """
    Сравнение нескольких траекторий в 3D

    Параметры:
    -----------
    traj_dict : dict
        {'Algorithm Name': DataFrame, ...}
    """
    fig = go.Figure()

    colors = ['green', 'red', 'blue', 'orange', 'purple', 'brown']

    # GT (если есть в первом DataFrame)
    first_df = list(traj_dict.values())[0]
    fig.add_trace(go.Scatter3d(
        x=first_df['gt_x'], y=first_df['gt_y'], z=first_df['gt_z'],
        mode='lines', name='Ground Truth',
        line=dict(width=5, color='gray'),
        opacity=0.4
    ))

    for idx, (name, df) in enumerate(traj_dict.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatter3d(
            x=df['est_x'], y=df['est_y'], z=df['est_z'],
            mode='lines', name=name,
            line=dict(width=3, color=color),
            opacity=0.8
        ))

    fig.update_layout(
        title='Сравнение траекторий (3D)',
        scene=dict(
            xaxis_title='X (м)',
            yaxis_title='Y (м)',
            zaxis_title='Z (м)',
            aspectmode='data'
        ),
        height=600,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=True
    )

    return fig


def plot_altitude_profile(df: pd.DataFrame) -> go.Figure:
    """
    Профиль высоты (Z) вдоль траектории
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.sqrt(df['gt_x']**2 + df['gt_y']**2),
        y=df['gt_z'],
        mode='lines', name='GT',
        line=dict(width=3, color='green')
    ))

    fig.add_trace(go.Scatter(
        x=np.sqrt(df['est_x']**2 + df['est_y']**2),
        y=df['est_z'],
        mode='lines', name='Оценка',
        line=dict(width=2, color='red', dash='dot')
    ))

    fig.update_layout(
        title='Профиль высоты',
        xaxis_title='Расстояние от начала (м)',
        yaxis_title='Высота Z (м)',
        height=400,
        template='plotly_white'
    )

    return fig


def animate_trajectory(df: pd.DataFrame,
                      step: int = 10) -> go.Figure:
    """
    Анимация прохождения траектории
    """
    fig = go.Figure()

    # Статичная GT траектория
    fig.add_trace(go.Scatter3d(
        x=df['gt_x'], y=df['gt_y'], z=df['gt_z'],
        mode='lines', name='Ground Truth',
        line=dict(width=3, color='green'),
        opacity=0.3
    ))

    # Анимированная оценка
    fig.add_trace(go.Scatter3d(
        x=df['est_x'].iloc[0:1],
        y=df['est_y'].iloc[0:1],
        z=df['est_z'].iloc[0:1],
        mode='lines+markers', name='Оценка',
        line=dict(width=4, color='red'),
        marker=dict(size=4, color='red')
    ))

    # Кадры анимации
    frames = []
    for i in range(0, len(df), step):
        frame = go.Frame(
            data=[
                go.Scatter3d(
                    x=df['est_x'].iloc[0:i+1],
                    y=df['est_y'].iloc[0:i+1],
                    z=df['est_z'].iloc[0:i+1],
                    mode='lines+markers',
                    line=dict(width=4, color='red'),
                    marker=dict(size=4, color='red')
                )
            ],
            name=str(i)
        )
        frames.append(frame)

    fig.frames = frames

    # Кнопки управления
    fig.update_layout(
        title='Анимация траектории',
        scene=dict(
            xaxis_title='X (м)',
            yaxis_title='Y (м)',
            zaxis_title='Z (м)',
            aspectmode='data'
        ),
        height=600,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶ Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 50, 'redraw': True},
                                   'fromcurrent': True}]
                },
                {
                    'label': '⏸ Stop',
                    'method': 'animate',
                    'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                     'mode': 'immediate'}]
                }
            ]
        }]
    )

    return fig
