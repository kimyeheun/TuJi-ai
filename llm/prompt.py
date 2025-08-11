import json

from openai import AsyncOpenAI

from Config import settings as cfg
from llm.dsl import *

SYSTEM_PROMPT = """
당신은 금융 및 알고리즘 트레이딩 전략 전문가입니다.
사용자가 입력하는 주식 매매 전략을 아래 JSON 구조의 DSL로 변환하세요.
반드시 아래의 JSON 규칙을 따르세요.
- 조건은 논리적으로 중첩(AND/OR)할 수 있습니다.

[예시 입력]
RSI가 30 밑에서 반등하거나, SMA50이 SMA200을 상향 돌파하면 50% 매수. RSI가 70 이상이면 전량 매도.
[예시 출력]
{
  "strategy_title": "RSI+SMA 복합 매매",
  "entries": [
    {
      "logic": "OR",
      "conditions": [
        {
          "logic": "AND",
          "conditions": [
            {
              "indicator": "RSI",
              "operator": "<",
              "value": 30,
              "trend": "up"
            }
          ]
        },
        {
          "logic": "AND",
          "conditions": [
            {
              "indicator": "SMA",
              "params": {"timeperiod": 50},
              "operator": "crosses_above",
              "compare_to": "SMA200"
            }
          ]
        }
      ],
      "action": "buy",
    }
  ],
  "exits": [
    {
      "logic": "AND",
      "conditions": [
        {
          "indicator": "RSI",
          "operator": ">=",
          "value": 70
        }
      ],
      "action": "sell",
    }
  ]
}
[예시 입력]
"볼린저밴드 하단을 터치하고 RSI가 20 이하에서 상승 전환하며 MACD가 시그널선을 상향돌파하면 매수. 볼린저밴드 상단 돌파하거나 RSI 80 이상이면 매도."
[예시 출력]
{
  "strategy_title": "볼린저밴드+RSI+MACD 삼중필터 매매",
  "entries": [
    {
      "logic": "AND",
      "conditions": [
        {
          "indicator": "BBANDS_LOWER",
          "params": {"timeperiod": 20, "nbdevdn": 2},
          "operator": "<=",
          "value": 0,
          "trend": "up"
        },
        {
          "indicator": "RSI",
          "params": {"timeperiod": 14},
          "operator": "<=",
          "value": 20,
          "trend": "up"
        },
        {
          "indicator": "MACD",
          "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
          "operator": "crosses_above",
          "compare_to": {
            "indicator": "MACD_SIGNAL",
            "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9}
          }
        }
      ],
      "action": "buy",
    }
  ],
  "exits": [
    {
      "logic": "OR",
      "conditions": [
        {
          "indicator": "BBANDS_UPPER",
          "params": {"timeperiod": 20, "nbdevup": 2},
          "operator": ">=",
          "value": 0
        },
        {
          "indicator": "RSI",
          "params": {"timeperiod": 14},
          "operator": ">=",
          "value": 80
        }
      ],
      "action": "sell",
    }
  ]
}
"""

function_schema = {
    "name": "parse_trading_strategy",
    "description": "자연어 매매 전략을 복합 논리 및 운용 비중을 포함한 DSL로 변환",
    "parameters": {
        "type": "object",
        "properties": {
            "strategy_title": {"type": "string"},
            "entries": {
                "type": "array",
                "items": {"$ref": "#/definitions/EntryOrExit"}
            },
            "exits": {
                "type": "array",
                "items": {"$ref": "#/definitions/EntryOrExit"}
            }
        },
        "required": ["strategy_title", "entries", "exits"],
        "definitions": {
            "EntryOrExit": {
                "type": "object",
                "properties": {
                    "logic": {
                        "type": "string",
                        "enum": ["AND", "OR"]
                    },
                    "conditions": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"$ref": "#/definitions/Condition"},
                                {"$ref": "#/definitions/EntryOrExit"}  # 복합 논리 재귀
                            ]
                        }
                    },
                    "action": {"type": "string", "enum": ["buy", "sell", "buy_partial", "sell_partial", "hold"]},
                },
                "required": ["logic", "conditions", "action"]
            },
            "Condition": {
                "type": "object",
                "properties": {
                    "indicator": {"type": "string"},
                    "operator": {"type": "string"},
                    "value": {"type": ["number", "string", "null"]},
                    "compare_to": {"type": ["string", "null"]},
                    "params": {"type": "object"},
                    "trend": {"type": ["string", "null"]},
                    "lag": {"type": "integer"}
                },
                "required": ["indicator", "operator"]
            }
        }
    }
}


def parse_condition(cond_json: Dict[str, Any]) -> Condition:
    """
    단일 Condition 또는 AND/OR 복합 조건(재귀) JSON을 파싱하여 Condition 객체로 변환.
    """
    # 복합 논리 조건인지 확인 (logic 필드 유무)
    if 'logic' in cond_json and 'conditions' in cond_json:
        # 하위 conditions도 재귀적으로 변환
        sub_conditions = [parse_condition(sub) for sub in cond_json['conditions']]
        return Condition(
            logic=cond_json['logic'],
            conditions=sub_conditions
        )
    else:
        # 단일 leaf 조건
        return Condition(
            indicator=cond_json.get('indicator'),
            operator=cond_json.get('operator'),
            value=cond_json.get('value'),
            compare_to=cond_json.get('compare_to'),
            params=cond_json.get('params'),
            trend=cond_json.get('trend'),
            lag=cond_json.get('lag', 0)
        )

def parse_entry_or_exit(entry_json: Dict[str, Any]) -> EntryOrExit:
    """
    Entry/Exit 조건(복합 논리 + 액션) JSON을 파싱하여 EntryOrExit 객체로 변환.
    """
    # 조건부 복합 논리 root_condition 파싱
    root_condition = parse_condition({
        "logic": entry_json.get("logic"),
        "conditions": entry_json.get("conditions", [])
    })
    return EntryOrExit(
        root_condition=root_condition,
        action=entry_json["action"],
        comment=entry_json.get("comment")
    )

def parse_strategy_dsl(dsl_json: Dict[str, Any]) -> StrategyDSL:
    """
    LLM이 반환한 최상위 DSL JSON을 전체 StrategyDSL 객체로 변환.
    """
    entries = [parse_entry_or_exit(entry) for entry in dsl_json.get("entries", [])]
    exits = [parse_entry_or_exit(exit_) for exit_ in dsl_json.get("exits", [])]
    return StrategyDSL(
        strategy_title=dsl_json.get("strategy_title", ""),
        entries=entries,
        exits=exits,
        custom_logic_required=dsl_json.get("custom_logic_required", False),
        custom_logic_description=dsl_json.get("custom_logic_description")
    )


async def generate_dsl(client, natural_text: str) -> StrategyDSL:
    response = await client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": f"{SYSTEM_PROMPT}"},
            {"role": "user", "content": f"{natural_text}"}
        ],
        functions=[function_schema],
        function_call={"name": "parse_trading_strategy"}
    )

    function_args_str = response.choices[0].message.function_call.arguments
    dsl_json = json.loads(function_args_str)
    dsl_obj = parse_strategy_dsl(dsl_json)
    return dsl_obj


# async def generate_dsl_ollama(natural_text: str) -> StrategyDSL:
#     llm = ChatOllama(
#         model="qwen2.5:14b-instruct-q4_K_M",
#         base_url="http://localhost:11434",
#         temperature=0,
#         format="json",
#     )
#     llm = llm.bind_tools([function_schema])
#
#     PROMPT = PromptTemplate(
#         input_variables=["SYSTEM_PROMPT"],
#         template=(
#             """{SYSTEM_PROMPT}"""
#         ),
#     )
#     chain = PROMPT | llm
#     response = await chain.ainvoke(
#         {"SYSTEM_PROMPT": SYSTEM_PROMPT},
#     )
#     raw = response.content
#     try:
#         dsl_json = json.loads(raw)
#     except Exception:
#         # 첫 번째 {...} 블록만 뽑아 파싱 시도 (모델이 앞/뒤에 텍스트 붙였을 때)
#         import re
#         m = re.search(r"\{[\s\S]*\}$", raw.strip())
#         if not m:
#             raise
#         dsl_json = json.loads(m.group(0))
#
#     print("+++++json++++++")
#     print(dsl_json)
#
#
#     return parse_strategy_dsl(dsl_json)


# OLLAMA = AsyncOpenAI(base_url=cfg.api.HOST+":"+cfg.api.LLAMA_PORT+"/v1", api_key="ollama")  # api_key는 더미면 됨
#
# async def generate_dsl_ollama(natural_text: str) -> StrategyDSL:
#     tools = [{
#         "type": "function",
#         "function": function_schema  # {"name","description","parameters"} 그대로
#     }]
#
#     resp = await OLLAMA.chat.completions.create(
#         model="qwen3:8b", #"qwen2.5:14b-instruct-q4_K_M",  # 또는 14b-instruct-q4_K_M (VRAM 여유에 따라)
#         messages=[
#             {"role": "system", "content": f"{SYSTEM_PROMPT}"},
#             {"role": "user", "content": f"{natural_text}"},
#         ],
#         tools=tools,
#         tool_choice={
#             "type": "function",
#             "function": {"name": "parse_trading_strategy"}
#         },
#         temperature=0,
#     )
#     print("=====================")
#     print(resp)
#     msg = resp.choices[0].message
#     print("=====================")
#     print(msg)
#     print("=====================")
#     tool_call = resp.choices[0].message.tool_calls[0]
#     args = json.loads(tool_call.function.arguments)
#     print(args)
#     print("~~~~~~~~~~~~~~~~~~~")
#     # 1) 정상: tool_calls 로 넘어온 arguments 사용
#     args_str = None
#     if getattr(msg, "tool_calls", None):
#         for tc in msg.tool_calls:
#             if tc.function.name == "parse_trading_strategy":
#                 args_str = tc.function.arguments
#                 break
#
#     # 2) 예외: content에 JSON이 직접 온 경우(모델 편차) 대비
#     if args_str is None:
#         raw = msg.content or ""
#         try:
#             dsl_json = json.loads(raw)
#         except Exception:
#             import re
#             m = re.search(r"\{[\s\S]*\}$", raw.strip())
#             if not m:
#                 # TODO: 예외 처리
#                 raise RuntimeError(f"DSL JSON을 파싱할 수 없습니다: {raw[:200]}...")
#             dsl_json = json.loads(m.group(0))
#     else:
#         dsl_json = json.loads(args_str)
#     return parse_strategy_dsl(dsl_json)

