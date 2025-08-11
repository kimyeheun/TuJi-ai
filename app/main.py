from fastapi import FastAPI
from app.api import stock

app = FastAPI()
app.include_router(stock.router)
