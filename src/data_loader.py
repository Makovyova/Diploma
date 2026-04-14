"""
Модуль загрузки данных из ROS bag файлов и CSV
Поддержка EuRoC MAV формата
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
import json


class DataLoader:
    """Загрузка и кэширование данных траекторий"""

    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self._cache = {}

    def load_csv_trajectory(self, filepath: str) -> pd.DataFrame:
        """
        Загрузка траектории из CSV файла
        Ожидаемые колонки: timestamp, gt_x, gt_y, gt_z, est_x, est_y, est_z
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Файл не найден: {filepath}")

        # Кэширование
        cache_key = str(filepath)
        if cache_key in self._cache:
            return self._cache[cache_key]

        df = pd.read_csv(filepath)

        # Валидация колонок
        required_cols = ['timestamp', 'gt_x', 'gt_y', 'gt_z', 'est_x', 'est_y', 'est_z']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing}")

        # Расчёт ошибки позиционирования
        if 'error' not in df.columns:
            df['error'] = np.sqrt(
                (df['est_x'] - df['gt_x'])**2 +
                (df['est_y'] - df['gt_y'])**2 +
                (df['est_z'] - df['gt_z'])**2
            )

        self._cache[cache_key] = df
        return df

    def load_bag_file(self, bag_path: str, topics: Optional[List[str]] = None) -> Dict:
        """
        Загрузка ROS bag файла (требует rosbag)
        Конвертация в DataFrame для последующего использования
        """
        try:
            import rosbag
            from sensor_msgs.msg import Image, Imu
            from nav_msgs.msg import Odometry
        except ImportError:
            raise ImportError(
                "Для работы с ROS bag требуется библиотека rosbag.\n"
                "Установите: pip install rosbag-tools\n"
                "Или используйте ROS environment"
            )

        bag_path = Path(bag_path)
        if not bag_path.exists():
            raise FileNotFoundError(f"ROS bag файл не найден: {bag_path}")

        result = {
            'timestamps': [],
            'gt_data': [],
            'est_data': [],
            'imu_data': [],
        }

        bag = rosbag.Bag(str(bag_path))

        # Чтение топиков
        available_topics = bag.get_type_and_topic_info().keys()
        print(f"Доступные топики: {list(available_topics)}")

        if topics is None:
            topics = list(available_topics)

        for topic, msg, t in bag.read_messages(topics=topics):
            timestamp = msg.header.stamp.to_sec()

            # Ground Truth (обычно из Vicon/OptiTrack)
            if 'vicon' in topic.lower() or 'gt' in topic.lower():
                result['gt_data'].append({
                    'timestamp': timestamp,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z,
                    'qx': msg.pose.pose.orientation.x,
                    'qy': msg.pose.pose.orientation.y,
                    'qz': msg.pose.pose.orientation.z,
                    'qw': msg.pose.pose.orientation.w,
                })

            # Оценка одометрии
            elif 'odom' in topic.lower() or 'estimate' in topic.lower():
                result['est_data'].append({
                    'timestamp': timestamp,
                    'x': msg.pose.pose.position.x,
                    'y': msg.pose.pose.position.y,
                    'z': msg.pose.pose.position.z,
                })

            # IMU данные
            elif 'imu' in topic.lower():
                result['imu_data'].append({
                    'timestamp': timestamp,
                    'ax': msg.linear_acceleration.x,
                    'ay': msg.linear_acceleration.y,
                    'az': msg.linear_acceleration.z,
                    'wx': msg.angular_velocity.x,
                    'wy': msg.angular_velocity.y,
                    'wz': msg.angular_velocity.z,
                })

        bag.close()

        # Конвертация в DataFrame
        if result['gt_data']:
            result['gt_df'] = pd.DataFrame(result['gt_data'])
        if result['est_data']:
            result['est_df'] = pd.DataFrame(result['est_data'])
        if result['imu_data']:
            result['imu_df'] = pd.DataFrame(result['imu_data'])

        return result

    def merge_gt_est(self, gt_df: pd.DataFrame, est_df: pd.DataFrame,
                     time_sync_tol: float = 0.01) -> pd.DataFrame:
        """
        Синхронизация GT и оценки по времени
        """
        merged = pd.merge_asof(
            est_df.sort_values('timestamp'),
            gt_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=time_sync_tol
        )

        merged = merged.rename(columns={
            'x_x': 'est_x', 'y_x': 'est_y', 'z_x': 'est_z',
            'x_y': 'gt_x', 'y_y': 'gt_y', 'z_y': 'gt_z'
        })

        # Расчёт ошибки
        if all(c in merged.columns for c in ['est_x', 'est_y', 'est_z', 'gt_x', 'gt_y', 'gt_z']):
            merged['error'] = np.sqrt(
                (merged['est_x'] - merged['gt_x'])**2 +
                (merged['est_y'] - merged['gt_y'])**2 +
                (merged['est_z'] - merged['gt_z'])**2
            )

        return merged

    def save_trajectory_csv(self, df: pd.DataFrame, filename: str,
                           subdir: str = "trajectories"):
        """Сохранение траектории в CSV"""
        output_path = self.results_dir / subdir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Сохранено: {output_path}")
        return output_path

    def save_metrics_json(self, metrics: Dict, filename: str,
                         subdir: str = "metrics"):
        """Сохранение метрик в JSON"""
        output_path = self.results_dir / subdir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Сохранено: {output_path}")
        return output_path

    def load_all_results(self, pattern: str = "*.csv") -> List[pd.DataFrame]:
        """Загрузка всех результатов по маске"""
        trajectory_dir = self.results_dir / "trajectories"
        if not trajectory_dir.exists():
            return []

        files = list(trajectory_dir.glob(pattern))
        results = []
        for f in files:
            try:
                df = self.load_csv_trajectory(f)
                df['source_file'] = f.name
                results.append(df)
            except Exception as e:
                print(f"Ошибка загрузки {f}: {e}")

        return results
