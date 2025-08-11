from __future__ import annotations

import re
from typing import Dict, Any, Tuple, List, Optional

from llm.dsl import StrategyDSL, Condition

# =========================
# 1) TALib 지표 메타 정보
# =========================
# 입력 컬럼은 프로젝트 전반에서 소문자 사용(df['close'])를 가정합니다.
INDICATOR_FUNC_MAP: Dict[str, Dict[str, Any]] = {
    "RSI":  {"func": "talib.RSI", "params": {"timeperiod": 14}, "input": ["close"]},
    "MOM":  {"func": "talib.MOM", "params": {"timeperiod": 10}, "input": ["close"]},
    "CCI":  {"func": "talib.CCI", "params": {"timeperiod": 14}, "input": ["high", "low", "close"]},
    "SMA":  {"func": "talib.SMA", "params": {"timeperiod": 20}, "input": ["close"]},
    "EMA":  {"func": "talib.EMA", "params": {"timeperiod": 20}, "input": ["close"]},
    "BBANDS": {
        "func": "talib.BBANDS",
        "params": {"timeperiod": 20, "nbdevup": 2, "nbdevdn": 2},
        "input": ["close"]
    },
    "MACD": {
        "func": "talib.MACD",
        "params": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
        "input": ["close"]
    },
    # 캔들 패턴(정수 시그널 반환; 보통 0, 100, -100)
    "CDLDOJI":         {"func": "talib.CDLDOJI",         "input": ["open", "high", "low", "close"]},
    "CDLHAMMER":       {"func": "talib.CDLHAMMER",       "input": ["open", "high", "low", "close"]},
    "CDLENGULFING":    {"func": "talib.CDLENGULFING",    "input": ["open", "high", "low", "close"]},
    "CDLSHOOTINGSTAR": {"func": "talib.CDLSHOOTINGSTAR", "input": ["open", "high", "low", "close"]},
    "CDLSPINNINGTOP":  {"func": "talib.CDLSPINNINGTOP",  "input": ["open", "high", "low", "close"]},
    "CDLMORNINGSTAR":  {"func": "talib.CDLMORNINGSTAR",  "input": ["open", "high", "low", "close"]},
    "CDLEVENINGSTAR":  {"func": "talib.CDLEVENINGSTAR",  "input": ["open", "high", "low", "close"]},
}

# ======================================
# 2) 연산자/컴포넌트/별칭 정규화 테이블
# ======================================
OP_ALIASES = {
    # 교차
    "will_cross_above": "crosses_above",
    "will_cross_below": "crosses_below",
    "cross_above": "crosses_above",
    "cross_below": "crosses_below",
    # 비교
    "above": ">", "below": "<",
    "gt": ">", "greater": ">", "greater_than": ">", "gte": ">=", "ge": ">=",
    "lt": "<", "less": "<", "less_than": "<", "lte": "<=", "le": "<=",
    "eq": "==", "equals": "==", "equal": "==", "neq": "!=", "ne": "!=",
    # 추세
    "rising": "is_trending_up",
    "falling": "is_trending_down",
}

# MACD/BBANDS 서브 컴포넌트 별칭
COMPONENT_ALIAS: Dict[str, Tuple[str, str]] = {
    "MACD_LINE":   ("MACD", "macd_line"),
    "MACD_SIGNAL": ("MACD", "macd_signal"),
    "MACD_HIST":   ("MACD", "macd_hist"),
    "BBANDS_UPPER": ("BBANDS", "bb_upper"),
    "BBANDS_MIDDLE": ("BBANDS", "bb_mid"),
    "BBANDS_LOWER": ("BBANDS", "bb_lower"),
    # 흔한 변형
    "BB_UPPER": ("BBANDS", "bb_upper"),
    "BB_MIDDLE": ("BBANDS", "bb_mid"),
    "BB_MID":    ("BBANDS", "bb_mid"),
    "BB_LOWER":  ("BBANDS", "bb_lower"),
}

# =========================
# 3) 내부 유틸
# =========================
def _norm_indicator(name: Optional[str]) -> str:
    return (name or "").strip().upper()

def _norm_operator(op: Optional[str]) -> str:
    if not op:
        return ""
    opn = op.strip().lower()
    return OP_ALIASES.get(opn, opn)

def _to_float_safe(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None

def _macd_suffix(p: Dict[str, Any]) -> str:
    f = int(p.get("fastperiod", 12)); s = int(p.get("slowperiod", 26)); g = int(p.get("signalperiod", 9))
    return "" if (f, s, g) == (12, 26, 9) else f"_f{f}_s{s}_sig{g}"

def _ma_suffix(p: Dict[str, Any]) -> str:
    tp = int(p.get("timeperiod", 20))
    return f"{tp}"

def _bb_suffix(p: Dict[str, Any]) -> str:
    tp = int(p.get("timeperiod", 20))
    up = float(p.get("nbdevup", 2)); dn = float(p.get("nbdevdn", 2))
    return "" if (tp, up, dn) == (20, 2.0, 2.0) else f"_{tp}_{int(up)}_{int(dn)}"

# ============================================
# 4) 인디케이터 코드 생성 (중복계산 캐시 포함)
# ============================================
def indicator_to_code(indicator: str,
                      params: Optional[Dict[str, Any]] = None,
                      df_var: str = "df",
                      computed_cache: Optional[Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, str]]] = None
                      ) -> Tuple[List[str], Dict[str, str]]:
    """
    Returns:
        lines:   실행 코드 라인들
        varmap:  컴포넌트명 → 변수명
                 - MACD:   {"macd_line": "...", "macd_signal": "...", "macd_hist": "..."}
                 - BBANDS: {"bb_upper": "...", "bb_mid": "...", "bb_lower": "..."}
                 - 단일:   {"main": "..."}
    """
    base_indicator = _norm_indicator(indicator)
    local_params = dict(params or {})

    # SMA50, EMA200 등 파라미터 내장 네이밍 파싱
    m = re.match(r'^(SMA|EMA|WMA|DEMA|TEMA|TRIMA|KAMA|MAMA|T3)(\d+)$', base_indicator, re.IGNORECASE)
    if m:
        base_indicator = m.group(1).upper()
        timeperiod = int(m.group(2))
        local_params = {**local_params, "timeperiod": timeperiod}

    # MACD_SIGNAL 등 서브 별칭이 단독으로 들어오면, 실제 계산 대상은 MACD임
    if base_indicator in COMPONENT_ALIAS:
        base_indicator = COMPONENT_ALIAS[base_indicator][0]  # ("MACD", "macd_signal") -> "MACD"

    # if base_indicator not in INDICATOR_FUNC_MAP:
    #     # 알 수 없는 지표 => 코드 생성 불가
    #     raise ValueError(f"Unknown indicator: {indicator}")

    if computed_cache is None:
        computed_cache = {}

    lines: List[str] = []
    varmap: Dict[str, str] = {}

    if base_indicator in INDICATOR_FUNC_MAP:
        meta = INDICATOR_FUNC_MAP[base_indicator]
        func = meta["func"]
        inputs = meta["input"]
        all_params = dict(meta.get("params", {}))
        all_params.update(local_params)

        key = (base_indicator, tuple(sorted(all_params.items())))
        if key in computed_cache:
            return [], computed_cache[key]

        input_args = ", ".join([f"{df_var}['{col}']" for col in inputs])
        param_str = ", ".join([f"{k}={repr(v)}" for k, v in all_params.items()])

        if base_indicator == "MACD":
            suf = _macd_suffix(all_params)
            v_line, v_sig, v_hist = f"macd_line{suf}", f"macd_signal{suf}", f"macd_hist{suf}"
            lines.append(f"{v_line}, {v_sig}, {v_hist} = {func}({input_args}, {param_str})")
            varmap = {"macd_line": v_line, "macd_signal": v_sig, "macd_hist": v_hist}

        elif base_indicator == "BBANDS":
            suf = _bb_suffix(all_params)
            v_up, v_mid, v_low = f"bb_upper{suf}", f"bb_mid{suf}", f"bb_lower{suf}"
            lines.append(f"{v_up}, {v_mid}, {v_low} = {func}({input_args}, {param_str})")
            varmap = {"bb_upper": v_up, "bb_mid": v_mid, "bb_lower": v_low}

        elif base_indicator in ("SMA", "EMA"):
            tp = _ma_suffix(all_params)
            v = f"{base_indicator.lower()}{tp}"
            assign = f"{v} = {func}({input_args}, {param_str})" if param_str else f"{v} = {func}({input_args})"
            lines.append(assign)
            varmap = {"main": v}

        else:
            v = base_indicator.lower()
            assign = f"{v} = {func}({input_args}, {param_str})" if param_str else f"{v} = {func}({input_args})"
            lines.append(assign)
            varmap = {"main": v}

        computed_cache[key] = varmap
        return lines, varmap


# ===========================================================
# 5) 조건식 빌더: 시리즈/상수 비교, 교차, 추세, 랙 등 전부 처리
# ===========================================================
def dsl_to_code(dsl: StrategyDSL, df_var: str = 'df') -> str:
    """
    DSL(StrategyDSL) → 실행 가능한 파이썬 코드 문자열.
    exec(...) 하면 다음 변수가 생성됩니다:
      - entry_signal_i (i=0..)
      - exit_signal_i  (i=0..)
      - final_buy_signal (bool Series)
      - final_sell_signal (bool Series)
    """
    code_lines: List[str] = []
    computed_cache: Dict[Tuple[str, Tuple[Tuple[str, Any], ...]], Dict[str, str]] = {}

    def false_series() -> str:
        return f"pd.Series([False]*len({df_var}), index={df_var}.index)"

    def ensure_series(ind_name: str,
                      params: Optional[Dict[str, Any]] = None,
                      prefer_component: Optional[str] = None) -> str:
        name_u = _norm_indicator(ind_name)

        # 서브컴포넌트 별칭 → 기본지표 계산 후 해당 컴포넌트 반환
        if name_u in COMPONENT_ALIAS:
            base, comp = COMPONENT_ALIAS[name_u]
            lines, varmap = indicator_to_code(base, params, df_var=df_var, computed_cache=computed_cache)
            code_lines.extend(lines)
            return varmap[comp]

        # 일반 지표
        lines, varmap = indicator_to_code(name_u, params, df_var=df_var, computed_cache=computed_cache)
        code_lines.extend(lines)

        if name_u == "MACD":
            comp = prefer_component if prefer_component in ("macd_line", "macd_signal", "macd_hist") else "macd_line"
            return varmap[comp]
        if name_u == "BBANDS":
            comp = prefer_component if prefer_component in ("bb_upper", "bb_mid", "bb_lower") else "bb_mid"
            return varmap[comp]
        return varmap["main"]

    def resolve_compare_to(compare_to: Any,
                           left_indicator_hint: Optional[str] = None) -> Tuple[bool, str | float]:
        """
        compare_to가 dict/str/숫자 모두 가능:
          - dict: {"indicator": "...", "params": {...}}
          - str:  "MACD_SIGNAL" / "SMA60" / "EMA20" / "BB_UPPER" ...
          - num:  70, 0, ...
        반환: (is_series, name_or_value)
        """
        if compare_to is None:
            return False, 0.0
        # dict 오브젝트
        if isinstance(compare_to, dict):
            ind2 = _norm_indicator(compare_to.get("indicator"))
            params2 = compare_to.get("params") or {}
            prefer = None
            if ind2 in COMPONENT_ALIAS:
                prefer = COMPONENT_ALIAS[ind2][1]
            v = ensure_series(ind2, params2, prefer_component=prefer)
            return True, v
        # 문자열: 서브컴포넌트/MA숫자 등
        if isinstance(compare_to, str):
            ind2 = _norm_indicator(compare_to)
            prefer = None
            if ind2 in COMPONENT_ALIAS:
                prefer = COMPONENT_ALIAS[ind2][1]
            v = ensure_series(ind2, None, prefer_component=prefer)
            return True, v
        # 숫자
        num = _to_float_safe(compare_to)
        return (False, 0.0 if num is None else num)

    def apply_lag_to_var(varname: str, lag: int) -> str:
        if not lag or lag == 0:
            return varname
        return f"({varname}.shift({int(lag)}))"

    def cond_to_expr(cond: Condition) -> str:
        # (1) 재귀 논리
        if cond.logic and cond.conditions:
            inner = [cond_to_expr(c) for c in cond.conditions]
            op = "&" if cond.logic == "AND" else "|"
            # 빈 리스트 방지
            if not inner:
                return false_series()
            return f"({op.join(inner)})"

        # (2) 리프 조건
        ind = _norm_indicator(cond.indicator)
        if not ind:
            return false_series()

        op = _norm_operator(cond.operator)
        val = cond.value
        params = cond.params or {}
        lag = getattr(cond, "lag", 0) or 0
        trend = getattr(cond, "trend", None)

        # 좌변 시리즈
        prefer = None
        if ind in COMPONENT_ALIAS:
            prefer = COMPONENT_ALIAS[ind][1]
        left_var = ensure_series(ind, params, prefer_component=prefer)
        left_var_lagged = apply_lag_to_var(left_var, lag)

        # 캔들패턴 (정수 반환) 특수 처리
        if ind.startswith("CDL"):
            # 기본 비교가 없으면 100 시그널로 가정
            if op in ["<", ">", "<=", ">=", "==", "!="]:
                base = f"({left_var_lagged} {op} {int(_to_float_safe(val) or 0)})"
            else:
                base = f"({left_var_lagged} == 100)"
            return base

        # 우변: compare_to 우선, 없으면 value
        is_series, right = resolve_compare_to(getattr(cond, "compare_to", None), left_indicator_hint=ind)
        expr: str

        # (3) 교차/비교/추세
        if op in ("crosses_above", "crosses_below"):
            if is_series:
                right_var = str(right)
                prev_cmp = "<=" if op == "crosses_above" else ">="
                now_cmp  = ">"  if op == "crosses_above" else "<"
                expr = f"(({left_var_lagged}.shift(1) {prev_cmp} {right_var}.shift(1)) & ({left_var_lagged} {now_cmp} {right_var}))"
            else:
                # 상수와의 교차
                valf = _to_float_safe(right if right is not None else val)
                if valf is None:
                    return false_series()
                prev_cmp = "<=" if op == "crosses_above" else ">="
                now_cmp  = ">"  if op == "crosses_above" else "<"
                expr = f"(({left_var_lagged}.shift(1) {prev_cmp} {valf}) & ({left_var_lagged} {now_cmp} {valf}))"

        elif op in ("is_trending_up", "is_trending_down"):
            expr = f"({left_var_lagged} > {left_var_lagged}.shift(1))" if op == "is_trending_up" else f"({left_var_lagged} < {left_var_lagged}.shift(1))"

        elif op in ("<", ">", "<=", ">=", "==", "!="):
            if is_series:
                right_var = str(right)
                expr = f"({left_var_lagged} {op} {right_var})"
            else:
                # 상수 비교
                valf = _to_float_safe(val if val is not None else right)
                if valf is None:
                    return false_series()
                expr = f"({left_var_lagged} {op} {valf})"
        else:
            # 미지원 연산자 → 안전 False
            return false_series()

        # (4) 추가 trend 필터
        if trend == "up":
            expr = f"({expr} & ({left_var_lagged} > {left_var_lagged}.shift(1)))"
        elif trend == "down":
            expr = f"({expr} & ({left_var_lagged} < {left_var_lagged}.shift(1)))"

        return expr

    # =============================
    # entries / exits → 최종 시그널
    # =============================
    entry_vars: List[str] = []
    for i, entry in enumerate(dsl.entries):
        cond_expr = cond_to_expr(entry.root_condition)
        v = f"entry_signal_{i}"
        code_lines.append(f"{v} = {cond_expr}")
        entry_vars.append(v)

    exit_vars: List[str] = []
    for i, ex in enumerate(dsl.exits):
        cond_expr = cond_to_expr(ex.root_condition)
        v = f"exit_signal_{i}"
        code_lines.append(f"{v} = {cond_expr}")
        exit_vars.append(v)

    if entry_vars:
        code_lines.append(f"final_buy_signal = {' | '.join(entry_vars)}")
    else:
        code_lines.append(f"final_buy_signal = {false_series()}")

    if exit_vars:
        code_lines.append(f"final_sell_signal = {' | '.join(exit_vars)}")
    else:
        code_lines.append(f"final_sell_signal = {false_series()}")

    return "\n".join(code_lines)
