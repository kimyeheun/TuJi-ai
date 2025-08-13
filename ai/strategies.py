import numpy as np
import pandas as pd
import torch

from Config import Config


def get_feature_mask(mask, all_features, indicators):
    use_features = Config.PRICE_FEATURES + indicators
    for idx, feat in enumerate(all_features):
        if feat not in use_features:
            mask[:, idx] = 0.0
    return mask

# NOTE: 중급자
class IntermediateStrategy():
    @staticmethod
    def _prep_window_tensor(df_window: pd.DataFrame, scaler, indicators: list[str]):
        """(T, F) -> (1, T, F) 스케일링 + 마스킹"""
        X = df_window[Config.ALL_FEATURES].astype(np.float32).fillna(0.0).values
        X_scaled = scaler.transform(X)
        mask = np.ones_like(X_scaled, dtype=np.float32)
        for j, feat in enumerate(Config.ALL_FEATURES):
            if feat not in Config.PRICE_FEATURES and feat not in indicators:
                mask[:, j] = 0.0
        return X_scaled[None, ...], mask[None, ...]

    @staticmethod
    def _slice_valid_range(valid_idx: np.ndarray, start_i: int, end_i: int):
        left = np.searchsorted(valid_idx, start_i, side="left")
        right = np.searchsorted(valid_idx, end_i, side="right") - 1
        if left > right:
            return None
        return left, right

    def run(self, stock_data, model, scaler, buy, sell, window_size=30, indicators=None, **kwargs):
        indicators = [] if indicators is None else indicators
        valid_idx = []  # 각 윈도우의 "종료 시점" 원본 인덱스
        X, mask = [], []

        # 고정 마스크 (윈도우 공통)
        window_mask = get_feature_mask(
            np.ones((window_size, len(Config.ALL_FEATURES)), dtype=np.float32),
            Config.ALL_FEATURES, indicators
        )

        # 롤링 윈도우 구성
        for i in range(window_size, len(stock_data)):
            window = stock_data[Config.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            mask.append(window_mask)
            valid_idx.append(i)

        valid_idx = np.asarray(valid_idx, dtype=np.int64)
        X = np.asarray(X, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)

        # NaN 방어 + 스케일
        X[np.isnan(X)] = 0.0
        X = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        # 1회 추론 (배치)
        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32, device=device)
            masks = torch.tensor(mask, dtype=torch.float32, device=device)
            logits = model(inputs, masks)  # (B, C) 또는 (B, T, C)
            if logits.ndim == 3:
                logits = logits[:, -1, :]  # 마지막 타임스텝만
            probs_all = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, 3)

        # 출력 배열 (원본 길이 기준)
        n = len(stock_data)
        predictions_list = [0] * n
        MAX_HORIZON = 20

        # 공통: 구간 보정 함수
        def clamp_range(start_row, end_row):
            start_row = max(start_row, window_size - 1)  # 첫 유효 인덱스 보정
            end_row = min(end_row, n - 1)
            if end_row < start_row:
                return None
            return start_row, end_row

        # 매수 구간 처리
        for s in buy:
            # 구간 끝: 다음 sell 시작 전 또는 horizon
            next_sell_after_s = next((i for i in sell if i > s), None)
            end_row = s + MAX_HORIZON if next_sell_after_s is None else min(next_sell_after_s, s + MAX_HORIZON)
            rng = clamp_range(s, end_row)
            if rng is None:
                continue
            start_row, end_row = rng
            vr = self._slice_valid_range(valid_idx, start_row, end_row)
            if vr is None:
                continue
            l, r = vr  # 배치 인덱스 범위
            local = probs_all[l:r + 1, 1]  # class 1 = Buy
            best_b = l + int(np.argmax(local))
            predictions_list[int(valid_idx[best_b])] = 1

        # 매도 구간 처리
        for s in sell:
            next_buy_after_s = next((i for i in buy if i > s), None)
            end_row = s + MAX_HORIZON if next_buy_after_s is None else min(next_buy_after_s, s + MAX_HORIZON)
            rng = clamp_range(s, end_row)
            if rng is None:
                continue
            start_row, end_row = rng
            vr = self._slice_valid_range(valid_idx, start_row, end_row)
            if vr is None:
                continue
            l, r = vr
            local = probs_all[l:r + 1, 2]  # class 2 = Sell
            best_b = l + int(np.argmax(local))
            predictions_list[int(valid_idx[best_b])] = 2

        return predictions_list


# NOTE: 초급자
class LowerStrategy():
    def run(self, stock_data, model, scaler, window_size=30, indicators=None, **kwargs):
        indicators = [] if indicators is None else indicators
        valid_idx = []
        X, mask = [], []

        window_mask = get_feature_mask(
            np.ones((window_size, len(Config.ALL_FEATURES)), dtype=np.float32),
            Config.ALL_FEATURES, indicators)

        for i in range(window_size, len(stock_data)):
            window = stock_data[Config.ALL_FEATURES].iloc[i - window_size:i].values
            X.append(window)
            mask.append(window_mask)
            valid_idx.append(i)

        mask = np.array(mask)

        X = np.array(X)
        X[np.isnan(X)] = 0
        X = scaler.transform(X.reshape(-1, X.shape[2])).reshape(X.shape)

        model.eval()
        device = next(model.parameters()).device
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(device)
            masks = torch.tensor(mask, dtype=torch.float32).to(device)
            outputs = model(inputs, masks)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

        predictions_list = [0] * len(stock_data)
        for idx, i in enumerate(valid_idx):
            pred = preds[idx]
            if pred == 1:
                predictions_list[i] = 1  # 매수
            elif pred == 2:
                predictions_list[i] = 2  # 매도

        return predictions_list