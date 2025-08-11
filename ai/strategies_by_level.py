import os
from typing import List

import joblib
import numpy as np
import pandas as pd
import torch
from langchain.chains import SequentialChain

from Config import Config
from ai.Model import MaskAwareAttentionLSTM
from ai.get_data_from_prompt import get_indicators
from ai.strategies import LowerStrategy, IntermediateStrategy
from llm.dsl_interpreter import dsl_to_code
from llm.prompt import generate_dsl_ollama
from utils.mylog import logger


# NOTE:프롬프트 분기점
async def prompt_bifurcation(difficulty : int, prompt_text:str, stock_df, client) -> List:
    if difficulty == 2:
        action = await upper_level(prompt_text, stock_df, client)
    elif difficulty == 1:
        action = await intermediate_level(prompt_text, stock_df, client)
    elif difficulty == 0:
        action = await lower_level(prompt_text, stock_df)
    else:
        raise NotImplementedError("난이도가 잘못 출력되었습니다. 복구 로직을 실행하세요.")
    return action[-60:]

# NOTE: 상
async def upper_level(prompt_text:str, stock_df:pd.DataFrame, openai_client) -> List:
    # 1. LLM → DSL 파싱
    dsl = await generate_dsl_ollama(prompt_text)

    # 2. DSL → 코드 변환
    # code = dsl_to_code(dsl, df_var="df")
    df_lc = stock_df.copy(deep=False)
    df_lc.columns = df_lc.columns.str.lower()
    code = dsl_to_code(dsl, df_var="df")

    logger.print(f"Code: \n{code}")

    # 3. 코드 실행 환경 준비 및 신호 추론
    local_env = {
        "df": df_lc,
        "np": np,
        "pd": pd,
        "talib": __import__("talib")  # TA-Lib 파이썬 래퍼
    }

    exec(code, local_env)

    # 4. Buy/Sell 시그널 불리언 시리즈를 int로 변환 (0: 유지, 1: 매수, 2: 매도)
    buy_signal = local_env.get("final_buy_signal", pd.Series([False]*len(stock_df)))
    sell_signal = local_env.get("final_sell_signal", pd.Series([False]*len(stock_df)))

    # TODO: 예외 처리 어떻게 할까...
    # [0,1,2] 리스트 생성 (우선 순위: 매수>매도>유지, 혹은 동시=매수)
    action = []
    for b, s in zip(buy_signal, sell_signal):
        if b and not s:
            action.append(1)  # 매수
        elif not b and s:
            action.append(2)  # 매도
        elif b and s:
            action.append(1)  # 동시시 매수 우선(예시)
        else:
            action.append(0)  # 유지
    return action

# NOTE: 중
async def intermediate_level(prompt_text:str, stock_df, openai_client) -> List:
    # 1. LLM → DSL 파싱
    dsl = await generate_dsl_ollama(prompt_text)

    # 2. DSL → 코드 변환
    # code = dsl_to_code(dsl, df_var="df")

    df_lc = stock_df.copy(deep=False)
    df_lc.columns = df_lc.columns.str.lower()
    code = dsl_to_code(dsl, df_var="df")

    # print("============code=============")
    # print(code)

#     code = """macd_line, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
# entry_signal_0 = (pd.Series([False]*len(df), index=df.index))
# exit_signal_0 = (((macd_line.shift(1) >= macd_signal.shift(1)) & (macd_line < macd_signal)))
# final_buy_signal = entry_signal_0
# final_sell_signal = exit_signal_0"""

    # 3. 코드 실행 환경 준비 및 신호 추론
    local_env = {
        "df": df_lc,
        "np": np,
        "pd": pd,
        "talib": __import__("talib")  # TA-Lib 파이썬 래퍼
    }
    exec(code, local_env)

    # 4. Buy/Sell 시그널 불리언 시리즈를 int로 변환 (0: 유지, 1: 매수, 2: 매도)
    buy_signal = local_env.get("final_buy_signal", pd.Series([True]*len(stock_df)))
    sell_signal = local_env.get("final_sell_signal", pd.Series([True]*len(stock_df)))

    # 2) NaN 방어 + bool 캐스팅
    buy_signal = buy_signal.reindex(stock_df.index).fillna(False).astype(bool)
    sell_signal = sell_signal.reindex(stock_df.index).fillna(False).astype(bool)

    # 3) fail-safe: 둘 중 해당 신호가 올 False면 전부 True로 승격
    if not buy_signal.any():
        buy_signal[:] = True
    if not sell_signal.any():
        sell_signal[:] = True

    use_indicators = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    current_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskAwareAttentionLSTM(input_dim=12, hidden_dim=64, output_dim=3, num_layers=2, dropout=0.3).to(device)

    model_path = os.path.join(current_dir, '..', 'models', 'lstm_classifier.pt')
    scaler_path = os.path.join(current_dir, '..', 'models', 'scaler.pkl')
    state_dict = torch.load(model_path, map_location=device)
    scaler = joblib.load(scaler_path)
    model.load_state_dict(state_dict)

    buy_signal = np.where(buy_signal.values)[0].tolist()
    sell_signal = np.where(sell_signal.values)[0].tolist()
    print("buy: " , buy_signal)
    print("sell: " , sell_signal)

    intermediate_level = IntermediateStrategy()
    action = intermediate_level.run(stock_df, model, scaler,
                                        buy=buy_signal,
                                        sell=sell_signal,
                                        window_size=30,
                                        indicators=use_indicators)

    print(action)
    return action

# NOTE: 하
async def lower_level(prompt_text:str, stock_df) -> List:
    # use_indicators = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    use_indicators = await get_indicators(prompt_text)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskAwareAttentionLSTM(input_dim=12, hidden_dim=64, output_dim=3, num_layers=2, dropout=0.3).to(device)

    # state_dict와 scaler 경로 확인!
    state_dict = torch.load(Config.MODEL_PATH , map_location=device)
    scaler = joblib.load(Config.SCALER_PATH)
    model.load_state_dict(state_dict)

    # 모델 추론
    lower_level = LowerStrategy()
    action = lower_level.run(stock_df, model, scaler,
                                   window_size=30,
                                   indicators=use_indicators)

    print(action)
    return action