from pydantic import BaseModel
from typing import List

'''
    처음 받아 오는 주식 데이터 
'''
class OHLCV(BaseModel):
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[float]

class StockInitRequest(BaseModel):
    roomId: int
    ohlcv: OHLCV

class StockInitResponse(BaseModel):
    result: str = "ok"


'''
    유저 프롬프트 
'''
class UserPrompt(BaseModel):
    userId: int
    prompt: str

class PromptRequest(BaseModel):
    roomId: int
    prompts: List[UserPrompt]

class ActionResult(BaseModel):
    userId: int
    buy: float
    sell: float
    action: List[int]

class PromptResponse(BaseModel):
    results: List[ActionResult]