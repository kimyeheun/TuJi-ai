from typing import List, Optional, Union, Dict, Any

class Condition:
    """
    단일 조건 또는 복합 조건(AND/OR)을 재귀적으로 표현하는 클래스.
    """
    def __init__(
        self,
        # 'AND', 'OR' 중 하나일 경우 하위 conditions 리스트 필요. (논리 조합)
        logic: Optional[str] = None,
        # 하위 조건 리스트. 단일 조건일 경우 None.
        conditions: Optional[List["Condition"]] = None,
        # 아래 4개는 단일 조건(leaf)에만 사용
        indicator: Optional[str] = None,        # 예: "RSI", "MACD"
        operator: Optional[str] = None,         # 예: "<", ">", "crosses_above", ...
        value: Optional[Union[float, int]] = None,     # 값(숫자) 비교일 경우
        compare_to: Optional[str] = None,       # 다른 indicator와 비교할 경우
        params: Optional[Dict[str, Any]] = None,# indicator 파라미터 예: {"timeperiod": 14}
        trend: Optional[str] = None,            # "up", "down" (트렌드, 옵션)
        lag: Optional[int] = 0,                 # 랙(며칠 전 값 비교, 옵션)
    ):
        self.logic = logic
        self.conditions = conditions
        self.indicator = indicator
        self.operator = operator
        self.value = value
        self.compare_to = compare_to
        self.params = params or {}
        self.trend = trend
        self.lag = lag

class EntryOrExit:
    """
    매수/매도 신호(진입/청산)의 조건 그룹, 액션, 운용비중을 표현.
    """
    def __init__(
        self,
        # 조건 루트 (Condition 인스턴스, 복합 논리 가능)
        root_condition: Condition,
        # buy, sell, buy_partial, sell_partial 등
        action: str,
        # 옵션: 기타 코멘트, 설명
        comment: Optional[str] = None,
    ):
        self.root_condition = root_condition
        self.action = action
        self.comment = comment

class StrategyDSL:
    """
    하나의 전략 전체를 표현.
    """
    def __init__(
        self,
        # 전략 이름, 설명
        strategy_title: str,
        # 매수 조건들 (EntryOrExit 객체 리스트)
        entries: List[EntryOrExit],
        # 매도 조건들 (EntryOrExit 객체 리스트)
        exits: List[EntryOrExit],
        # 옵션: 백테스트/실전 구현에 필요한 추가 설명
        custom_logic_required: bool = False,
        custom_logic_description: Optional[str] = None,
    ):
        self.strategy_title = strategy_title
        self.entries = entries
        self.exits = exits
        self.custom_logic_required = custom_logic_required
        self.custom_logic_description = custom_logic_description

    def __repr__(self):
        return f"StrategyDSL({self.strategy_title}, entries={len(self.entries)}, exits={len(self.exits)})"
