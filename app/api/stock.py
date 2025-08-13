import asyncio
import os
from time import time
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter
from openai import AsyncOpenAI

from utils.mylog import logger
# 지표 계산
from utils.calc_indicator import add_technical_indicators
from ai.get_data_from_prompt import classify_difficulty, get_propensity
from ai.strategies_by_level import prompt_bifurcation, upper_level, intermediate_level, lower_level
from app.schemas import StockInitRequest, StockInitResponse, PromptRequest, PromptResponse, ActionResult
from app.store import STOCK_DATA_STORE

load_dotenv()
router = APIRouter()
# client = None
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'),
    base_url="https://gms.ssafy.io/gmsapi/api.openai.com/v1")


@router.get("/ai/good")
def alright():
    return {"ok": True}


@router.post("/ai/init", response_model=StockInitResponse)
async def stock_init(request: StockInitRequest):
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
    STOCK_DATA_STORE[request.roomId] = df
    print(STOCK_DATA_STORE.keys())
    print(df)
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
    logger.print(f"Delay for prompt : {end - start}")

    start = time()
    actions = await prompt_bifurcation(difficulty, prompt, stock_df, client)
    end = time()
    logger.print(f"Delay for actions : {end - start}")

    logger.print(f"Actions : {str(actions)}")

    results.append(ActionResult(userId=user_id, buy=buy, sell=sell, action=actions))
    return PromptResponse(results=results)


@router.get("/ai/data")
async def get_data():
    return str(STOCK_DATA_STORE)
