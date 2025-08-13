# store.py
from typing import Dict, List

# 주식 데이터 (roomId -> DataFrame)
STOCK_DATA_STORE: Dict[int, object] = {}

# 액션 데이터 (roomId -> { userId -> [actions] })
ACTIONS_STORE: Dict[int, Dict[int, List[int]]] = {}


def save_actions(room_id: int, user_id: int, actions: List[int]) -> None:
    """roomId의 userId별 actions 저장/갱신"""
    room = ACTIONS_STORE.setdefault(room_id, {})
    room[user_id] = actions


def get_actions(room_id: int, user_id: int | None = None):
    """room의 전체/부분 조회"""
    if user_id is None:
        return ACTIONS_STORE.get(room_id, {})
    return ACTIONS_STORE.get(room_id, {}).get(user_id, [])
