from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

env_path = Path(__file__).resolve().parent.parent / '.env'

class ApiSettings(BaseSettings):
    MODEL: str = "exaone3.5"
    OLLAMA_HOST: str = "http://127.0.0.1:11434"
    PORT: str = "8000"

class LoggingSettings(BaseSettings):
    LEVEL: str = "INFO"

class Settings(BaseSettings):
    # .env 파일을 읽도록 설정
    model_config = SettingsConfigDict(env_file=env_path, env_file_encoding='utf-8')

    # 위에서 정의한 설정 클래스들을 속성으로 포함
    api: ApiSettings = ApiSettings()
    log: LoggingSettings = LoggingSettings()

settings = Settings()

# NOTE: ai Config
class Config:
    BASE_DIR = Path(__file__).resolve().parent

    MODEL_PATH = BASE_DIR / "ai" / "models" / "attention_lstm_classifier.pt"
    SCALER_PATH = BASE_DIR / "ai" / "models" / "scaler.pkl"
    HTML_PATH = BASE_DIR / "app" / "html" / "index.html"
    PRICE_FEATURES = ["Open", "High", "Low", "Close", "Volume"]
    INDICATOR_FEATURES = ["RSI", "MACD", "MACD_SIGNAL", "BB_UPPER", "BB_LOWER", "MOM", "CCI"]
    ALL_FEATURES = PRICE_FEATURES + INDICATOR_FEATURES
