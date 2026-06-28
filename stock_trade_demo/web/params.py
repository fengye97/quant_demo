"""保护6：TimingParams dataclass，统一封装择时 API 的查询参数解析。

把分散在 web_app.py 各个 timing 路由里的「从 request.args 把各种 int / float / bool / str
拼成 strategy 构造参数」的代码集中到这里，让 blueprint 只关心 HTTP 流程，参数解析逻辑只在
一个地方维护。

字段分类（与原 web_app._collect_realism_params 完全一致）：
  - bool : profit_lock_enabled
  - int  : limit_max_delay_days, confirm_days, probe_confirm_days,
           fast_window, slow_window, momentum_window,
           breakout_window, exit_window, trend_window,
           momentum_short_window, momentum_long_window
  - float: 阈值类 (enter/add/trim/exit_threshold, max_entry_exposure, probe_entry_exposure),
           sigmoid_k, max_leverage, base_position, inertia, crisis_vix,
           fed_block_weight, restrictive_threshold, pivot_relief, base_floor,
           momentum_threshold,
           5 个 profit_lock_* (drawdown / level_1 / level_2 / level_3),
           7 个费用/滑点 (slippage_bps, cash_interest_rate, commission_rate,
                          commission_min, stamp_tax_rate, transfer_fee_rate)
  - str  : exposure_mode

`from_query(args)` 直接吃 Flask 的 `request.args`（MultiDict），返回一个 TimingParams 实例。
`to_kwargs()` 输出可直接展开传给 build_timing_strategy / build_us_timing_strategy 的 dict
（不包含值为 None 的项，保留默认值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Mapping, Optional


# ── 类型分类 ────────────────────────────────────────────────────────────
_BOOL_KEYS = {'profit_lock_enabled'}

_INT_KEYS = {
    'limit_max_delay_days', 'confirm_days', 'probe_confirm_days',
    'fast_window', 'slow_window', 'momentum_window',
    'breakout_window', 'exit_window', 'trend_window',
    'momentum_short_window', 'momentum_long_window',
}

_FLOAT_KEYS = {
    'enter_threshold', 'add_threshold', 'trim_threshold', 'exit_threshold',
    'max_entry_exposure', 'probe_entry_exposure',
    'sigmoid_k', 'max_leverage', 'base_position', 'inertia', 'crisis_vix',
    'fed_block_weight', 'restrictive_threshold', 'pivot_relief',
    'base_floor', 'momentum_threshold',
    # profit_lock_*
    'profit_lock_drawdown', 'profit_lock_level_1', 'profit_lock_level_2', 'profit_lock_level_3',
    # fee / slippage
    'slippage_bps', 'cash_interest_rate',
    'commission_rate', 'commission_min',
    'stamp_tax_rate', 'transfer_fee_rate',
}

_STR_KEYS = {'exposure_mode'}

REALISM_BOOL_KEYS = _BOOL_KEYS
REALISM_INT_KEYS = {'limit_max_delay_days'}
REALISM_FLOAT_KEYS = {
    'profit_lock_drawdown', 'profit_lock_level_1', 'profit_lock_level_2', 'profit_lock_level_3',
    'slippage_bps', 'cash_interest_rate',
    'commission_rate', 'commission_min',
    'stamp_tax_rate', 'transfer_fee_rate',
    'base_floor',
}
REALISM_ALL_KEYS = REALISM_BOOL_KEYS | REALISM_INT_KEYS | REALISM_FLOAT_KEYS


def _parse_bool(raw: Any) -> Optional[bool]:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in {'on', 'true', '1', 'yes', 'y'}:
        return True
    if s in {'off', 'false', '0', 'no', 'n', ''}:
        return False
    return None


def _parse_int(raw: Any) -> Optional[int]:
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def _parse_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@dataclass
class TimingParams:
    """择时策略参数统一容器。所有字段默认为 None，from_query 只填充确实出现的参数。"""

    # bool
    profit_lock_enabled: Optional[bool] = None

    # int (state / fee)
    limit_max_delay_days: Optional[int] = None
    confirm_days: Optional[int] = None
    probe_confirm_days: Optional[int] = None

    # int (策略窗口)
    fast_window: Optional[int] = None
    slow_window: Optional[int] = None
    momentum_window: Optional[int] = None
    breakout_window: Optional[int] = None
    exit_window: Optional[int] = None
    trend_window: Optional[int] = None
    momentum_short_window: Optional[int] = None
    momentum_long_window: Optional[int] = None

    # float (阈值)
    enter_threshold: Optional[float] = None
    add_threshold: Optional[float] = None
    trim_threshold: Optional[float] = None
    exit_threshold: Optional[float] = None
    max_entry_exposure: Optional[float] = None
    probe_entry_exposure: Optional[float] = None

    # float (宏观/趋势)
    sigmoid_k: Optional[float] = None
    max_leverage: Optional[float] = None
    base_position: Optional[float] = None
    inertia: Optional[float] = None
    crisis_vix: Optional[float] = None
    fed_block_weight: Optional[float] = None
    restrictive_threshold: Optional[float] = None
    pivot_relief: Optional[float] = None
    base_floor: Optional[float] = None
    momentum_threshold: Optional[float] = None

    # float (profit lock)
    profit_lock_drawdown: Optional[float] = None
    profit_lock_level_1: Optional[float] = None
    profit_lock_level_2: Optional[float] = None
    profit_lock_level_3: Optional[float] = None

    # float (fee / slippage)
    slippage_bps: Optional[float] = None
    cash_interest_rate: Optional[float] = None
    commission_rate: Optional[float] = None
    commission_min: Optional[float] = None
    stamp_tax_rate: Optional[float] = None
    transfer_fee_rate: Optional[float] = None

    # str
    exposure_mode: Optional[str] = None

    @classmethod
    def from_query(cls, args: Mapping[str, Any]) -> 'TimingParams':
        """解析 Flask request.args（或任何 Mapping）到 TimingParams。不存在的 key 留 None。"""
        kwargs: dict = {}
        for key in _BOOL_KEYS:
            raw = args.get(key)
            if raw is None:
                continue
            val = _parse_bool(raw)
            if val is not None:
                kwargs[key] = val
        for key in _INT_KEYS:
            raw = args.get(key)
            if raw is None:
                continue
            val = _parse_int(raw)
            if val is not None:
                kwargs[key] = val
        for key in _FLOAT_KEYS:
            raw = args.get(key)
            if raw is None:
                continue
            val = _parse_float(raw)
            if val is not None:
                kwargs[key] = val
        for key in _STR_KEYS:
            raw = args.get(key)
            if raw is None or raw == '':
                continue
            kwargs[key] = str(raw)
        return cls(**kwargs)

    def to_kwargs(self) -> dict:
        """生成可直接展开传给 build_timing_strategy(**) 的 dict（不含 None）。"""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def realism_kwargs(self) -> dict:
        """仅 realism 子集（12 项），与原 _collect_realism_params 返回值等价。"""
        return {k: v for k, v in asdict(self).items()
                if v is not None and k in REALISM_ALL_KEYS}

    def diff_from_defaults(self, effective_defaults: Mapping[str, Any],
                           float_atol: float = 1e-9) -> dict:
        """返回与 effective_defaults 不一致的 (key -> request_val) 字典。

        bool / str / int 走严格 ==，float 走 abs(a-b) <= atol 的近似比较，避免前端
        URL 把 0.3 序列化成 "0.3" 再反序列化后跟 0.3 在 repr 上的细微差异引起误判。
        缺失字段（self.k is None 或 default 不存在该 key）视为「未指定」，不计差异。
        """
        diffs: dict = {}
        for key, req_val in asdict(self).items():
            if req_val is None:
                continue
            if key not in effective_defaults:
                # 请求带了一个 cache 默认值里不存在的 key（比如 chinext 没有 breakout_window 但用户传了）
                # → 视为非默认参数，必然 cache_miss
                diffs[key] = req_val
                continue
            default_val = effective_defaults[key]
            if isinstance(default_val, bool) or isinstance(req_val, bool):
                if bool(req_val) != bool(default_val):
                    diffs[key] = req_val
            elif isinstance(default_val, (int,)) and not isinstance(default_val, bool) \
                    and isinstance(req_val, (int,)) and not isinstance(req_val, bool):
                if int(req_val) != int(default_val):
                    diffs[key] = req_val
            elif isinstance(default_val, (int, float)) and isinstance(req_val, (int, float)):
                if abs(float(req_val) - float(default_val)) > float_atol:
                    diffs[key] = req_val
            else:
                if str(req_val) != str(default_val):
                    diffs[key] = req_val
        return diffs

    def is_all_defaults(self, effective_defaults: Mapping[str, Any],
                       float_atol: float = 1e-9) -> bool:
        """请求里出现的所有参数是否都等于 effective_defaults。"""
        return not self.diff_from_defaults(effective_defaults, float_atol=float_atol)
