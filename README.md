# TuJi-ai

**프롬프트 기반 자동 주식 트레이딩 AI**  

사용자가 자연어로 작성한 트레이딩 전략 프롬프트를 해석하여, AI 모델을 통해 매수·매도 시점을 예측하고  
FastAPI 서버를 통해 손쉽게 실행할 수 있는 알고리즘 트레이딩 시스템입니다.

---

## 📌 프로젝트 개요

### 배경
- 기존 규칙 기반(stock indicator only) 트레이딩의 한계 → 데이터 기반 학습 모델(LSTM, Attention LSTM 등) 도입
- 자연어 입력을 받아 **LLM**(대규모 언어 모델)이 이를 구조화된 전략 코드(DSL)로 변환
- 변환된 전략을 기반으로 AI 모델이 예측한 매매 신호를 실행
- FastAPI를 통해 REST API로 호출 가능 → 다른 서비스나 UI와 쉽게 연동

### 주요 특징
1. **자연어 전략 파싱**
   - 사용자가 `"RSI가 30 이하일 때 매수, 70 이상일 때 매도"`처럼 입력하면 이를 DSL로 변환
2. **AI 모델 기반 신호 예측**
   - OHLCV + 보조 지표(RSI, MACD, 볼린저밴드 등)를 입력받아 매수/매도 확률 예측
3. **규칙 + 모델 하이브리드 전략**
   - 규칙 기반 필터로 진입 조건을 걸러내고, 그 이후 모델 예측으로 타이밍 결정
4. **FastAPI 서버**
   - `/ai/init`, `/ai/prompt`, `/ai/data` 등의 엔드포인트 제공

---

## 📂 디렉토리 구조 & 코드 설명
```
TuJi-ai/
├── app/ # FastAPI 애플리케이션
│ ├── main.py # API 엔트리포인트, 서버 실행
│ ├── routers/ # API 라우팅 모듈
│ └── services/ # 비즈니스 로직 계층
│
├── ai/ # AI 모델 관련 코드
│ ├── models/ # 학습된 모델 파일(LSTM, Attention LSTM 등)
│ ├── train_model.py # 모델 학습 스크립트
│ ├── predict_model.py # 예측 실행 로직
│ └── strategy_model_based.py # 모델 기반 매매 전략 구현
│
├── llm/ # LLM 기반 전략 파서
│ ├── prompt_templates.py # LLM 프롬프트 템플릿
│ ├── dsl_parser.py # 자연어 → DSL 변환
│ └── ollama_client.py # Ollama API 연동
│
├── utils/ # 공통 유틸리티
│ ├── data_loader.py # CSV/DB 데이터 로딩
│ ├── feature_engineering.py# 기술적 지표 생성
│ ├── mask_utils.py # 피처 마스킹 로직
│ └── backtest.py # 백테스트 엔진
│
├── Config.py # 프로젝트 환경설정
├── requirements.txt # Python 패키지 목록
└── README.md # 프로젝트 설명 문서
```

## ⚙️ 설치 및 실행
### 1. 저장소 클론
```bash
git clone https://github.com/kimyeheun/TuJi-ai.git
cd TuJi-ai
```
### 2. 가상환경 생성 및 활성화
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
### 3. 패키지 설치
```pip install -r requirements.txt```
### 4. 환경 변수 설정
.env 파일 또는 Config.py에서 API 키, 모델 경로 등을 설정합니다.
### 5. 서버 실행
```uvicorn app.main:app --reload```

---
## 📊 API 예시
### 1. 전략 생성
```
POST /strategy
{
  "prompt": "RSI가 30 이하일 때 매수, 70 이상일 때 매도"
}
```
+ LLM이 DSL로 변환하여 내부에 저장

### 2. 예측 실행
```
POST /predict
{
  "symbol": "AAPL",
  "start_date": "2024-01-01",
  "end_date": "2024-06-30"
}
```
+ AI 모델이 매수/매도/홀드 신호 반환

### 🧠 모델 설명
+ **MaskAAttentionSTM**
  + 기술적 지표별 사용 여부에 따라 마스크를 적용하여 학습
  + 결측값 처리 + 선택적 지표 학습
  + 시계열 내 중요한 시점 가중치 부여
+ **전략 유형**
  1) **순수 모델 기반 ** : 프롬프트에 나온 조건 검증부터 진입·청산 타이밍 결정까지 모든 판단을 AI 모델이 수행합니다.
  2) **규칙 기반** : ta-lib 라이브러리를 활용해 프롬프트에 명시된 모든 매매 조건을 충족했을 때만 매수·매도를 실행합니다.
  3) **규칙 + 모델 하이브리드** : 프롬프트에 명시된 최소한의 조건만 필터로 적용하고, 그 외의 매매 시점은 AI 모델이 자율적으로 판단하여 결정합니다.
