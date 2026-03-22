
import numpy as np
import pandas as pd


class RewardCalculator:
    def __init__(self):
        self.trajectory_data = {}
        # self.ma_min = 3
        # self.ma_max = 12
        self.position_max_usd = 1000.0
        self.position_shares = None
        self.previous_signal = None
        self.entry_price = None

    def _get_signal(self, trajectory_data, ma_1, ma_2):

        df = pd.DataFrame.from_dict(trajectory_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df['SMA_1'] = df['c'].rolling(window=ma_1).mean()
        df['SMA_2'] = df['c'].rolling(window=ma_2).mean()

        ma_1_c = df['SMA_1'].iloc[-1]
        ma_2_c = df['SMA_2'].iloc[-1]

        return 1 if ma_1_c > ma_2_c else -1, df['c'].iloc[-1]

    def _get_pl(self, signal, price):
        # ====================================================================================
        if self.position_shares is None: # no position
            if signal == 1:
                self.position_shares = int(self.position_max_usd / price)
                self.previous_signal = signal
                self.entry_price = price
            elif signal == -1:
                self.position_shares = int(self.position_max_usd / price) * -1
                self.previous_signal = signal
                self.entry_price = price
            else:
                raise RuntimeError('bad signal')
            return 0.0
        # ====================================================================================
        elif self.previous_signal == 1 and signal == 1: # staying long
            return 0.0
        # ====================================================================================
        elif self.previous_signal == -1 and signal == -1: # staying short
            return 0.0
        # ====================================================================================
        elif self.previous_signal == 1 and signal == -1: # switching from long to short
            pl = (price - self.entry_price) * abs(self.position_shares)
            self.position_shares = int(self.position_max_usd / price) * -1
            self.previous_signal = signal
            self.entry_price = price
            return pl
        # ====================================================================================
        elif self.previous_signal == -1 and signal == 1: # switching from short to long
            pl = (self.entry_price - price) * abs(self.position_shares)
            self.position_shares = int(self.position_max_usd / price)
            self.previous_signal = signal
            self.entry_price = price
            return pl
        # ====================================================================================
        else:
            raise RuntimeError('bad signal')

    def _reward_ma1_ma2(self, ma_1_pred, ma_2_pred):

        ma_1_rescaled = int(np.rint(np.clip((9.0 * ma_1_pred) + 3.0, a_min=3.0, a_max=12.0)))
        ma_2_rescaled = int(np.rint(np.clip((9.0 * ma_2_pred) + 3.0, a_min=3.0, a_max=12.0)))

        signal, price = self._get_signal(self.trajectory_data, ma_1_rescaled, ma_2_rescaled)
        pl = self._get_pl(signal, price)
        return pl / 100.0

    def _reward_channel_spread(self, ma_1_pred, ma_2_pred):
        median_rescaled = np.clip((9.0 * ma_1_pred) + 3.0, a_min=3.0, a_max=12.0)
        spread_rescaled = np.clip(((5.0 * ma_2_pred) + 2.0) / 2.0, a_min=2.0, a_max=7.0)

        ma_1_rescaled = int(np.rint(np.clip(median_rescaled + spread_rescaled, a_min=3.0, a_max=12.0)))
        ma_2_rescaled = int(np.rint(np.clip(median_rescaled - spread_rescaled, a_min=3.0, a_max=12.0)))

        # print(median_rescaled, spread_rescaled , '|', ma_1_rescaled, ma_2_rescaled)

        signal, price = self._get_signal(self.trajectory_data, ma_1_rescaled, ma_2_rescaled)
        pl = self._get_pl(signal, price)
        return pl / 100.0

    def get_reward(self, action, observation_data):
        self.trajectory_data = self.trajectory_data | observation_data
        action_1_pred = action[0]
        action_2_pred = action[1]

        # return self._reward_ma1_ma2(action_1_pred, action_2_pred)
        return self._reward_channel_spread(action_1_pred, action_2_pred)
