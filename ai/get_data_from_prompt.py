# NOTE: 프롬프트 분기점
import logging
import random
from typing import Any

from langchain.chains import LLMChain, SequentialChain
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_ollama import ChatOllama

from Config import settings as cfg, Config
from utils.mylog import logger


# NOTE: 상 중 하 분류 후 각 코드로 분기
async def classify_difficulty(prompt_text: str) -> int:
    llm = ChatOllama(
        model=cfg.api.MODEL,
        base_url=cfg.api.OLLAMA_HOST,
        temperature=0,
        format="json",
    )

    PROMPT = PromptTemplate(
        input_variables=["prompt_text"],
        template=(
            """너는 지시문을 0/1/2로 분류한다.
- 0: 기준 없음/감각적/위임
- 1: 단일 기준(지표 1개/규칙 1개)
- 2: 복합 기준(2개↑ 조합, 손절/익절 포함)
반드시 JSON 하나만 출력: {{"label": 0|1|2}}
텍스트: {prompt_text}
JSON:"""
        ),
    )

    chain = PROMPT | llm | JsonOutputParser()
    result = await chain.ainvoke({"prompt_text": prompt_text})
    result = result.get("label", 0)

    if isinstance(result, int) and result in (0, 1, 2):
        return result
    if isinstance(result, str) and result.strip() in ("0", "1", "2"):
        return int(result.strip())
    return 0

# NOTE: 투자 성향 분석하기
async def get_propensity(prompt_text: str) -> tuple[Any, Any]:
    llm = ChatOllama(
        model=cfg.api.MODEL,
        base_url=cfg.api.OLLAMA_HOST,
        temperature=0,
        format="json",
    )

    template = PromptTemplate(
        input_variables=["prompt_text"],
        template=
        """너는 한국어 투자 지시문을 읽고 매수/매도 비율을 0~1 사이 숫자로 변환하는 변환기다.
반드시 아래 하나의 JSON 객체만 출력한다. 부가 텍스트/설명/코드블록 금지.

출력 스키마(정확히 이 키만):
{{"buy": <0~1 number>, "sell": <0~1 number>}}
- 소수점 둘째 자리까지 표기(예: 0.20)
- 문자열/퍼센트 기호 금지

판정 규칙:
1) 숫자 인식: "40%", "40 퍼센트", "사십 퍼센트" → 0.40 / "절반, 반, 하프" → 0.50
2) 강도어 → 수치 매핑(명시 수치 없을 때 사용):
   - 전부/전액/풀/몰빵/올인/가득: 1.00
   - 대부분/거의 전부/크게/강하게: 0.80
   - 과감히/공격적/도전적: 1.00
   - 일부: 0.30
   - 조금/소량/소폭/약간: 0.20
   - 관망/보류/기다림/하지 않음: 0.00
   - 중립/헤지/리밸런싱: 0.50
3) 문장에 buy/sell 둘 다 있으면 각각 독립적으로 판단(서로 영향 X).
   - 동일 측에 여러 표현이 있으면 **가장 강한(최대) 강도**를 채택.
   - 수치와 강도어가 함께 있으면 **수치 우선**.
4) 한 측만 언급되면, 다른 측은 0.00으로 둔다.
5) 아무 정보도 없으면 둘 다 0.00.
6) 항상 0~1 사이로 클램프. 1보다 큰 수치는 100으로 나누어 해석.

예시:
"전부 다 매수" → {{"buy": 1.00, "sell": 0.00}}
"전액 매도 조금만 매수" → {{"buy": 0.20, "sell": 1.00}}
"40퍼센트 사고 많이 팔래" → {{"buy": 0.40, "sell": 0.80}}
"도전적으로 사고 팔아보자" → {{"buy": 1.00, "sell": 1.00}}
"오늘은 관망" → {{"buy": 0.00, "sell": 0.00}}
"절반 매도" → {{"buy": 0.00, "sell": 0.50}}
"일부 매수" → {{"buy": 0.30, "sell": 0.00}}
"중립적으로 대응" → {{"buy": 0.50, "sell": 0.50}}

문장: {prompt_text}
JSON만 출력:
    """
    )

    chain = template | llm | JsonOutputParser()
    try:
        result = await chain.ainvoke({"prompt_text": prompt_text})
        buy_ratio = result.get("buy")
        sell_ratio = result.get("sell")

        if buy_ratio is not None and sell_ratio is not None:
            if buy_ratio > 1.0:
                buy_ratio = buy_ratio / 100
            if sell_ratio > 1.0:
                sell_ratio = sell_ratio / 100
        buy_ratio = buy_ratio or 1.0
        sell_ratio = sell_ratio or 1.0

        return buy_ratio, sell_ratio
    except Exception as e:
        logging.exception(f"정상적인 json 답변이 아니거나, 예상치 못한 오류 발생: {e}")
        return 0.3, 0.3

# NOTE: 프롬프트로부터 지표 뽑아내기
async def get_indicators(prompt_text: str) -> list:
    llm = ChatOllama(
        model=cfg.api.MODEL,
        base_url=cfg.api.OLLAMA_HOST,
        temperature=0,
        format="json",
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 텍스트에서 아래 허용 지표들만 찾아내는 추출기다.
허용 지표 외의 단어는 절대 포함하지 마라. 중복 제거.
허용 지표: {allowed}

출력은 반드시 JSON 객체 한 줄: {{\"indicators\": [\"RSI\", ...]}}
지표가 하나도 없으면 {{\"indicators\": []}} 만 출력하라. 
추가 설명, 코드블록, 문장은 금지."""),
        ("human", "text: {text}\nJSON만 출력:")
    ])
    chain = prompt | llm | JsonOutputParser()

    try:
        parsed = await chain.ainvoke({"text": prompt_text, "allowed": Config.ALL_FEATURES})
        result = parsed.get("indicators", [])
    except Exception as e:
        logging.exception(f"프롬프트로부터 지표를 정상적으로 추출하지 못함: {e}")
        result = []
    finally:
        if not result:
            k = min(5, len(Config.ALL_FEATURES))
            result = random.sample(Config.ALL_FEATURES, k=k)

    logger.print(f"Indicators : {result}")
    return result

