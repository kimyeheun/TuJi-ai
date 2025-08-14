# =============== Stage 0: build TA-Lib C ===============
FROM python:3.12-slim AS talib_c

ENV TA_PREFIX="/opt/ta-lib"
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates gfortran \
    libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

RUN curl -L https://github.com/ta-lib/ta-lib/releases/download/v0.6.4/ta-lib-0.6.4-src.tar.gz \
    -o ta-lib.tar.gz && \
    tar -xzf ta-lib.tar.gz && \
    cd ta-lib-0.6.4 && \
    ./configure --prefix="${TA_PREFIX}" && \
    make -j"$(nproc)" && \
    make install && \
    ls -alh ${TA_PREFIX}/lib


# =============== Stage 1: build Python wheels ===============
FROM python:3.12-slim AS wheels

# numpy/pandas 등 wheel 빌드용 툴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev \
 && rm -rf /var/lib/apt/lists/*

# TA-Lib C를 연결(파이썬 talib 빌드 시 필요)
COPY --from=talib_c /opt/ta-lib /opt/ta-lib

ENV LD_LIBRARY_PATH="/opt/ta-lib/lib" \
    TA_LIBRARY_PATH=/opt/ta-lib/lib \
    TA_INCLUDE_PATH=/opt/ta-lib/include \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .

# 가능한 한 wheel로 선컴파일(실패해도 런타임에서 설치하도록 허용)
RUN pip wheel --wheel-dir=/wheels -r requirements.txt || true

# =============== Stage 2: runtime ===============
FROM python:3.12-slim AS runtime

# 런타임에 필요한 최소 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 liblapack3 curl \
 && rm -rf /var/lib/apt/lists/*

# TA-Lib C 복사 및 환경변수
COPY --from=talib_c /opt/ta-lib /opt/ta-lib
ENV LD_LIBRARY_PATH=/opt/ta-lib/lib:$LD_LIBRARY_PATH \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# 앱 작업 디렉토리
WORKDIR /app

# 미리 빌드한 wheel 우선 설치, 없으면 PyPI에서 설치
COPY --from=wheels /wheels /wheels
COPY requirements.txt .
RUN if [ -d /wheels ] && ls /wheels/*.whl >/dev/null 2>&1; then \
      pip install --no-index --find-links=/wheels -r requirements.txt ; \
    else \
      pip install -r requirements.txt ; \
    fi

# 소스 코드
COPY . .

# 네트워킹/헬스체크
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://localhost:8000/ai/good || exit 1

# FastAPI 진입점 (트리 기준)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
