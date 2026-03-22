
import os
import json
import random
from datetime import datetime
import numpy as np

import constants as CONST


class DataLoader:
    def __init__(self):
        self.data_dir = CONST.DATA_DIR
        self.trajectory_num_steps = CONST.TRAJECTORY_STEPS
        self.observation_window_size = CONST.WINDOW_SIZE
        self.price_max = 700.0
        self.price_min = 7.0
        self.max_gap_percent = 30.0
        self.trajectory_symbol = None
        self.trajectory_data = None
        self.trajectory_index = None
        self.reset()


    def _get_json_data_v2(self, file_path, datetime_format='%Y-%m-%d'):
        file1 = open(file_path)
        json_data = json.load(file1)
        file1.close()
        key_dt__val_ohlcv = {}
        time_series = json_data['Time Series (Daily)']
        for date_str in time_series:
            date_dt = datetime.strptime(date_str, datetime_format)
            key_dt__val_ohlcv[date_dt] = {
                'o': float(time_series[date_str]['1. open']),
                'h': float(time_series[date_str]['2. high']),
                'l': float(time_series[date_str]['3. low']),
                'c': float(time_series[date_str]['4. close']),
                'v': float(time_series[date_str]['5. volume']),
            }
        key_dt__val_ohlcv = {k: v for k, v in sorted(key_dt__val_ohlcv.items())}
        return key_dt__val_ohlcv

    def _percent_diff(self, v_initial, v_final):
        v_initial = 1.0 if v_initial == 0.0 else v_initial
        v_final = 1.0 if v_final == 0.0 else v_final
        pd = (v_final - v_initial) / v_initial
        return pd * 100.0

    def _get_max_gap_percent(self, dt_trajectory, json_data):
        percent_differences = []
        dt_previous = None
        for dt_current in dt_trajectory:
            if dt_previous is None:
                dt_previous = dt_current
                continue
            percent_differences.append(self._percent_diff(
                json_data[dt_previous]['c'],
                json_data[dt_current]['o']
            ))
            dt_previous = dt_current
        return max(np.abs(percent_differences))

    def reset(self):
        while True:
            file_name = random.choice(os.listdir(self.data_dir))
            file_path = os.path.join(self.data_dir, file_name)
            json_data = self._get_json_data_v2(file_path)
            dt_all = list(json_data.keys())
            if len(dt_all) < (self.trajectory_num_steps + self.observation_window_size + 3):
                continue
            max_start = len(dt_all) - (self.trajectory_num_steps + self.observation_window_size + 1)
            start_idx = random.randint(0, max_start)
            dt_trajectory = dt_all[start_idx: start_idx + self.trajectory_num_steps + self.observation_window_size + 1]
            val_max = max([json_data[k]['h'] for k in dt_trajectory])
            val_min = min([json_data[k]['l'] for k in dt_trajectory])
            val_gap = self._get_max_gap_percent(dt_trajectory, json_data)
            if val_max < self.price_max and val_min > self.price_min and val_gap < self.max_gap_percent:
                self.trajectory_symbol = file_name.rsplit('.', 1)[0]
                self.trajectory_data = {k: json_data[k] for k in dt_trajectory}
                self.trajectory_index = 0
                break

    def _normalize_volume(self, data_np):
        median = np.median(data_np)
        q1 = np.percentile(data_np, 25)
        q3 = np.percentile(data_np, 75)
        iqr = q3 - q1
        return (data_np - median) / (iqr + 1e-8)

    def _normalize_price(self, data_np):
        mean = data_np.mean()
        std = data_np.std()
        return (data_np - mean) / (std + 1e-8)

    def _format_observation_data_v2(self, observation_data):
        data_price_ohc = []
        data_price_olc = []
        data_volume = []
        for dt in observation_data:
            data_price_ohc = data_price_ohc + [
                observation_data[dt]['o'],
                observation_data[dt]['h'],
                observation_data[dt]['c']
            ]
            data_price_olc = data_price_olc + [
                observation_data[dt]['o'],
                observation_data[dt]['l'],
                observation_data[dt]['c']
            ]
            volume = observation_data[dt]['v']
            data_volume = data_volume + [volume, volume, volume]
        return np.stack([
            self._normalize_price(np.array(data_price_ohc).astype(np.float32)),
            self._normalize_price(np.array(data_price_olc).astype(np.float32)),
            self._normalize_volume(np.array(data_volume).astype(np.float32))
        ], axis=0).astype(np.float32)

    def get_state(self):
        dt_trajectory = list(self.trajectory_data.keys())
        dt_observation = dt_trajectory[self.trajectory_index: self.trajectory_index + self.observation_window_size]
        observation_data = {k: self.trajectory_data[k] for k in dt_observation}
        self.trajectory_index = self.trajectory_index + 1
        return self._format_observation_data_v2(observation_data), observation_data

