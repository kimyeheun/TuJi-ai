import logging
import sys

# 1. 새 레벨 이름과 숫자 지정
CUSTOM_LEVEL = 25
logging.addLevelName(CUSTOM_LEVEL, "PRINT")


# 2. Logger 클래스 확장
class CustomLogger(logging.getLoggerClass()):
    def print(self, message, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_LEVEL):
            self._log(CUSTOM_LEVEL, message, args, **kwargs)


# 3. 새로운 클래스를 기본 로거로 설정
logging.setLoggerClass(CustomLogger)


# 4. 커스텀 포맷터 클래스 정의
class ColoredFormatter(logging.Formatter):
    """
    A custom formatter that adds color to log levels.
    """
    COLORS = {
        'WARNING': "\033[93m",  # Yellow
        'INFO': "\033[94m",  # Blue
        'DEBUG': "\033[92m",  # Green
        'CRITICAL': "\033[91m",  # Red
        'ERROR': "\033[91m",  # Red
        'PRINT': "\033[38;5;208m",  # Orange
        'RESET': "\033[0m"  # Reset color
    }

    def format(self, record):
        log_fmt = f"%(levelname)s:       %(message)s"
        formatter = logging.Formatter(
            self.COLORS.get(record.levelname, self.COLORS['RESET']) + log_fmt + self.COLORS['RESET'])
        return formatter.format(record)


# 5. 로거 인스턴스 생성 및 핸들러 설정
logger = logging.getLogger(__name__)

# 기존 핸들러가 있다면 제거
if logger.hasHandlers():
    logger.handlers.clear()

# StreamHandler 생성 및 커스텀 포맷터 설정
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(ColoredFormatter())
logger.addHandler(handler)

# 6. 로거 레벨 설정
logger.setLevel(logging.INFO)