"""实盘记录 API：/api/live/records GET, /api/live/record POST, /api/live/record/<id> DELETE, /api/live/reconcile。

CLAUDE.md 红线 15：实盘记录的写入必须通过 services.live_trades 走文件锁，
绝对禁止重建 / 覆盖 / 删除主文件 data/live_trades.csv。
"""
from __future__ import annotations

from datetime import datetime as _datetime
import pandas as pd
from flask import Blueprint, request, jsonify

from web import state
from services import live_trades as _live_trades_service

bp = Blueprint('live_api', __name__)


@bp.route('/api/live/records', methods=['GET'])
def api_live_records():
    strategy = request.args.get('strategy')
    rows = _live_trades_service.read_all()
    if strategy:
        rows = [r for r in rows if r.get('strategy') == strategy]
    rows.sort(key=lambda r: (r.get('date') or '', r.get('record_id') or ''))
    return jsonify({'records': rows})


@bp.route('/api/live/record', methods=['POST'])
def api_live_record_create():
    payload = request.get_json(silent=True) or {}
    required = ['date', 'strategy']
    missing = [k for k in required if payload.get(k) in (None, '')]
    if missing:
        return jsonify({'error': f'缺少字段: {missing}'}), 400
    known_strategies = set(state.TIMING_STRATEGY_MAP.keys()) | set(state.US_TIMING_STRATEGY_MAP.keys())
    if payload['strategy'] not in known_strategies:
        return jsonify({'error': f'未知策略: {payload["strategy"]}'}), 400

    strategy = payload['strategy']
    try:
        capital = float(payload.get('capital') or state._LIVE_INITIAL_CAPITAL)
    except (TypeError, ValueError):
        return jsonify({'error': 'capital 必须是数值'}), 400
    if capital <= 0:
        return jsonify({'error': 'capital 必须 > 0'}), 400

    exec_price = payload.get('exec_price')
    shares = payload.get('shares')
    actual_position = payload.get('actual_position')

    if exec_price not in (None, '') and shares not in (None, ''):
        try:
            exec_price_f = float(exec_price)
            shares_f = float(shares)
        except (TypeError, ValueError):
            return jsonify({'error': 'exec_price / shares 必须是数值'}), 400
        if exec_price_f <= 0 or shares_f < 0:
            return jsonify({'error': 'exec_price 必须 > 0，shares 必须 >= 0'}), 400
        holding_value = exec_price_f * shares_f
        actual_position_f = holding_value / capital
        if actual_position_f > 1.0:
            actual_position_f = 1.0
        exec_price_str = f'{exec_price_f:.4f}'
        shares_str = f'{shares_f:.4f}' if shares_f != int(shares_f) else str(int(shares_f))
    elif actual_position not in (None, ''):
        try:
            actual_position_f = float(actual_position)
        except (TypeError, ValueError):
            return jsonify({'error': 'actual_position 必须是 0~1 的浮点数'}), 400
        if not (0.0 <= actual_position_f <= 1.0):
            return jsonify({'error': 'actual_position 必须在 [0, 1] 之间'}), 400
        exec_price_str = str(exec_price or '')
        shares_str = str(shares or '')
    else:
        return jsonify({'error': '请提供 exec_price + shares，或直接提供 actual_position'}), 400

    new_row = _live_trades_service.append_record({
        'date': str(payload['date']),
        'strategy': str(strategy),
        'signal_target': str(payload.get('signal_target') or ''),
        'actual_position': f'{actual_position_f:.4f}',
        'exec_price': exec_price_str,
        'capital': f'{capital:.2f}',
        'notes': str(payload.get('notes') or ''),
        'created_at': _datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'shares': shares_str,
    })
    return jsonify({'ok': True, 'record': new_row})


@bp.route('/api/live/record/<int:record_id>', methods=['DELETE'])
def api_live_record_delete(record_id):
    if not _live_trades_service.delete_record(record_id):
        return jsonify({'error': f'未找到 record_id={record_id}'}), 404
    return jsonify({'ok': True, 'deleted': record_id})


@bp.route('/api/live/reconcile')
def api_live_reconcile():
    strategy_name = request.args.get('strategy')
    if not strategy_name:
        return jsonify({'error': '缺少 strategy 参数'}), 400
    cache = state.TIMING_CACHE if strategy_name in state.TIMING_STRATEGY_MAP else state.US_TIMING_CACHE
    if strategy_name not in state.TIMING_STRATEGY_MAP and strategy_name not in state.US_TIMING_STRATEGY_MAP:
        return jsonify({'error': f'未知策略: {strategy_name}'}), 404
    if strategy_name in state.TIMING_STRATEGY_MAP:
        try:
            state.init_timing_cache()
        except Exception as e:
            return jsonify({'error': f'策略缓存初始化失败: {e}'}), 500
    else:
        try:
            state.init_us_timing_cache()
        except Exception as e:
            return jsonify({'error': f'美股策略缓存初始化失败: {e}'}), 500
    result = cache.get(strategy_name)
    if result is None or len(result) == 0:
        return jsonify({'error': '策略缓存为空'}), 500
    all_rows = _live_trades_service.read_all()
    live_rows = sorted(
        [r for r in all_rows if r.get('strategy') == strategy_name and r.get('date')],
        key=lambda r: (r['date'], int(r.get('record_id') or 0)),
    )
    if not live_rows:
        return jsonify({
            'strategy_id': strategy_name,
            'live_records': 0,
            'empty_state': True,
            'initial_nav': 1.0,
            'initial_position': 0.0,
            'initial_capital': state._LIVE_INITIAL_CAPITAL,
            'currency': state._LIVE_CURRENCY.get(strategy_name, 'CNY'),
            'lot_size': state._LIVE_LOT_SIZE.get(strategy_name, 1),
            'message': '默认空仓状态：实盘 NAV = 1.0，当前持仓 = 0%。录入第一笔实盘交易后开始对账。',
            'series': [],
        })

    df = result.copy()
    df['交易日期'] = pd.to_datetime(df['交易日期'])
    cache_max = df['交易日期'].max()
    execution_date_col = 'execution_date'
    if execution_date_col not in df.columns:
        df[execution_date_col] = df['交易日期']
        try:
            all_dates = pd.to_datetime(df['交易日期']).tolist()
            shifted = all_dates[1:] + [pd.NaT]
            df[execution_date_col] = shifted
        except Exception:
            pass
    df[execution_date_col] = pd.to_datetime(df[execution_date_col], errors='coerce')

    first_record_date = pd.to_datetime(live_rows[0]['date'])
    df = df[df[execution_date_col] >= first_record_date].reset_index(drop=True)
    if len(df) == 0:
        return jsonify({
            'strategy_id': strategy_name,
            'live_records': len(live_rows),
            'empty_state': False,
            'pending_cache': True,
            'cache_max_date': cache_max.strftime('%Y-%m-%d') if pd.notna(cache_max) else None,
            'first_record_date': first_record_date.strftime('%Y-%m-%d'),
            'initial_capital': state._LIVE_INITIAL_CAPITAL,
            'currency': state._LIVE_CURRENCY.get(strategy_name, 'CNY'),
            'lot_size': state._LIVE_LOT_SIZE.get(strategy_name, 1),
            'message': f'已录入 {len(live_rows)} 条实盘记录，但首条日期 {first_record_date.strftime("%Y-%m-%d")} 已超过策略缓存的最后交易日 {cache_max.strftime("%Y-%m-%d") if pd.notna(cache_max) else "—"}。等下次数据刷新后将自动纳入对账。',
            'series': [],
        })

    base_nav = float(df['累积净值'].iloc[0]) or 1.0
    df['strategy_nav'] = df['累积净值'].astype(float) / base_nav

    initial_capital = float(live_rows[0].get('capital') or state._LIVE_INITIAL_CAPITAL)
    current_cash = initial_capital
    current_shares = 0.0
    current_position = 0.0
    last_close = None
    approximate_mode = False
    records_by_date = {}
    for r in live_rows:
        records_by_date.setdefault(pd.to_datetime(r['date']).strftime('%Y-%m-%d'), []).append(r)

    series = []
    for _, row in df.iterrows():
        exec_dt = row.get(execution_date_col)
        if pd.isna(exec_dt):
            continue
        exec_date = pd.to_datetime(exec_dt).strftime('%Y-%m-%d')
        trade_price = row.get('etf_open')
        mark_price = row.get('etf_close')
        if exec_date in records_by_date:
            for rec in records_by_date[exec_date]:
                rec_capital = float(rec.get('capital') or initial_capital)
                if rec_capital > 0:
                    initial_capital = rec_capital
                rec_price_raw = rec.get('exec_price')
                rec_shares_raw = rec.get('shares')
                if rec_price_raw not in (None, '') and rec_shares_raw not in (None, ''):
                    try:
                        rec_price = float(rec_price_raw)
                        rec_shares = float(rec_shares_raw)
                    except (TypeError, ValueError):
                        rec_price = float('nan')
                        rec_shares = float('nan')
                    if pd.notna(rec_price) and pd.notna(rec_shares) and rec_price > 0 and rec_shares >= 0:
                        current_shares = rec_shares
                        holding_cost = rec_price * rec_shares
                        current_cash = max(initial_capital - holding_cost, 0.0)
                        current_position = min(max(holding_cost / initial_capital, 0.0), 1.0) if initial_capital > 0 else 0.0
                        continue
                approximate_mode = True
                try:
                    current_position = float(rec.get('actual_position') or 0.0)
                except (TypeError, ValueError):
                    current_position = 0.0
                current_position = min(max(current_position, 0.0), 1.0)
                if pd.notna(trade_price) and trade_price and trade_price > 0 and initial_capital > 0:
                    current_shares = (initial_capital * current_position) / float(trade_price)
                    current_cash = max(initial_capital - current_shares * float(trade_price), 0.0)
                else:
                    current_shares = 0.0
                    current_cash = initial_capital

        if pd.isna(mark_price):
            if pd.notna(last_close):
                mark_price = last_close
            else:
                continue
        mark_price = float(mark_price)
        live_value = current_cash + current_shares * mark_price
        live_nav = live_value / initial_capital if initial_capital > 0 else 1.0
        last_close = mark_price
        current_position = (current_shares * mark_price / live_value) if live_value > 0 else 0.0
        series.append({
            'date': exec_date,
            'signal_date': pd.to_datetime(row['交易日期']).strftime('%Y-%m-%d'),
            'strategy_nav': round(float(row['strategy_nav']), 4),
            'live_nav': round(float(live_nav), 4),
            'actual_position': round(float(current_position), 4),
            'strategy_target': round(float(row.get('target_exposure', 0.0) or 0.0), 4),
            'mark_price': round(mark_price, 4),
            'share_units': round(float(current_shares), 4),
            'cash_balance': round(float(current_cash), 2),
        })

    return jsonify({
        'strategy_id': strategy_name,
        'live_records': len(live_rows),
        'start_date': first_record_date.strftime('%Y-%m-%d'),
        'initial_capital': initial_capital,
        'currency': state._LIVE_CURRENCY.get(strategy_name, 'CNY'),
        'lot_size': state._LIVE_LOT_SIZE.get(strategy_name, 1),
        'approximate_mode': approximate_mode,
        'series': series,
        'final_strategy_nav': series[-1]['strategy_nav'] if series else None,
        'final_live_nav': series[-1]['live_nav'] if series else None,
    })
