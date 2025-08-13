import json
import os
from http.client import HTTPException
from time import time

import numpy as np
import pandas as pd
from fastapi import APIRouter
from openai import AsyncOpenAI
from starlette.responses import HTMLResponse, FileResponse, JSONResponse

from Config import Config
from ai.get_data_from_prompt import classify_difficulty, get_propensity
from ai.strategies_by_level import prompt_bifurcation
from app.schemas import StockInitRequest, StockInitResponse, PromptRequest, PromptResponse, ActionResult
from app.store import STOCK_DATA_STORE, ACTIONS_STORE, get_actions, save_actions
# 지표 계산
from utils.calc_indicator import add_technical_indicators
from utils.mylog import logger
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()
client = AsyncOpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1")

@router.get("/ai/good")
def alright():
    return {"ok": True}


@router.post("/ai/init", response_model=StockInitResponse)
async def stock_init(request: StockInitRequest):
    room_id = request.roomId
    if room_id in STOCK_DATA_STORE.keys():
        logger.print(f"{room_id}'s Storages : new user enter")
        return StockInitResponse(result="ok")

    # 보조 지표 계산 후 저장.
    df = pd.DataFrame({
        "Open": request.ohlcv.open,
        "High": request.ohlcv.high,
        "Low": request.ohlcv.low,
        "Close": request.ohlcv.close,
        "Volume": request.ohlcv.volume,
    })
    df = add_technical_indicators(df)

    # 계산된 지표를 roomId 별로 저장
    STOCK_DATA_STORE[room_id] = df

    logger.print(f"{room_id}'s Storage created")
    return StockInitResponse(result="ok")


@router.post("/ai/prompt", response_model=PromptResponse)
async def stock_prompt(request: PromptRequest):
    room_id = request.roomId
    stock_df = STOCK_DATA_STORE.get(room_id)
    results = []

    user_id = request.prompts[0].userId
    prompt = request.prompts[0].prompt

    start = time()
    buy, sell = await get_propensity(prompt)
    difficulty = await classify_difficulty(prompt)
    end = time()

    logger.print(f"{user_id}'s Prompt - buy: {buy}, sell: {sell}, level: {difficulty}")
    logger.print(f"Delay for prompt : {end - start}")

    start = time()
    actions = await prompt_bifurcation(difficulty, prompt, stock_df, client)
    end = time()
    logger.print(f"Delay for actions : {end - start}")
    logger.print(f"{user_id}'s Actions : {str(actions)}")

    # NOTE: 시각화를 위한 user actions 저장
    save_actions(room_id, user_id, actions)

    results.append(ActionResult(userId=user_id, buy=buy, sell=sell, action=actions))
    return PromptResponse(results=results)


@router.get("/ai/visualization")
async def visualization(roomId: int):
    """
    정적 HTML 파일 반환 (데이터는 /ai/visualization/data 로 fetch)
    """
    html_path = Config.HTML_PATH
    if not html_path.exists():
        raise HTTPException(500, f"HTML not found: {html_path}")
    # 파일만 반환. roomId는 쿼리스트링으로 HTML 측에서 그대로 사용
    return FileResponse(str(html_path))

@router.get("/ai/visualization/data")
async def visualization_data(roomId: int):
    df = STOCK_DATA_STORE.get(roomId)
    if df is None:
        return JSONResponse({"error": f"No data for roomId={roomId}. Call /ai/init first."}, status_code=404)

    if isinstance(df, pd.DataFrame):
        df: pd.DataFrame = df
    else:
        try:
            df = pd.DataFrame(df)
        except Exception as e:
            raise HTTPException(500, f"Data for roomId={roomId} is not a DataFrame: {type(df)}")

        # 2) NaN/Inf 정리 (여기서 0으로 채우고 싶으면 .fillna(0) 사용)
    # df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    df = df[-60:]
    x = list(range(len(df)))

    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    base = {c: (df[c].tolist() if c in df.columns else []) for c in base_cols}

    indicators = {c: df[c].tolist() for c in Config.INDICATOR_FEATURES if c in df.columns}

    actions_map = ACTIONS_STORE.get(roomId, {})  # { userId: [actions] }

    payload = {"x": x, "base": base, "indicators": indicators, "actionsMap": actions_map}
    return JSONResponse(payload)

@router.get("/ai/data")
async def get_data():
    return str(STOCK_DATA_STORE)
