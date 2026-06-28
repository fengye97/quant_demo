"""数据管理 API：/api/update_data /api/update_index_data 触发 + status 轮询，
以及前置 check 和自重启端点（前端"先确认是否需要更新 → 更新 → 重启"流程的服务端配对）。
"""
from __future__ import annotations

import threading
from flask import Blueprint, jsonify

from web import state

bp = Blueprint('data_admin_api', __name__)


@bp.route('/api/update_index_data', methods=['POST'])
def api_update_index_data():
    if state._INDEX_UPDATE_STATUS.get('stage') == 'running':
        return jsonify({'error': '指数数据更新正在进行中，请勿重复触发'}), 409
    state._INDEX_UPDATE_STATUS['stage'] = 'idle'
    state._INDEX_UPDATE_STATUS['message'] = ''
    state._INDEX_UPDATE_STATUS['progress'] = 0
    state._INDEX_UPDATE_STATUS['warning'] = None
    state._INDEX_UPDATE_STATUS['details'] = None
    threading.Thread(target=state._run_index_data_update, daemon=True).start()
    return jsonify({'status': 'started'})


@bp.route('/api/update_index_data/status')
def api_update_index_data_status():
    return jsonify(state._INDEX_UPDATE_STATUS)


@bp.route('/api/update_index_data/check')
def api_update_index_data_check():
    """只读判定指数 / ETF 日线缓存是否需要刷新。
    不发起任何外部请求、不写盘；响应应在 ~100ms 内返回。
    """
    return jsonify(state._check_index_etf_freshness())


@bp.route('/api/update_data', methods=['POST'])
def api_update_data():
    if state._UPDATE_DATA_STATUS['running']:
        return jsonify({'error': '数据更新正在进行中，请勿重复触发'}), 409
    threading.Thread(target=state._run_data_update, daemon=True).start()
    return jsonify({'status': 'started', 'message': '数据更新已启动'})


@bp.route('/api/update_data/status')
def api_update_data_status():
    return jsonify({
        'running': state._UPDATE_DATA_STATUS['running'],
        'stage': state._UPDATE_DATA_STATUS['stage'],
        'message': state._UPDATE_DATA_STATUS['message'],
        'progress_pct': state._UPDATE_DATA_STATUS['progress_pct'],
        'error': state._UPDATE_DATA_STATUS['error'],
    })


@bp.route('/api/update_data/check')
def api_update_data_check():
    """只读判定 stock_data.csv 月度面板是否落后于最新交易日所在月。"""
    return jsonify(state._check_stock_data_freshness())


# ── 辅助数据流水线（FRED + A股估值/情绪 + risk_signals 汇总） ──
@bp.route('/api/update_aux_data', methods=['POST'])
def api_update_aux_data():
    if state._AUX_UPDATE_STATUS.get('running'):
        return jsonify({'error': '辅助数据更新正在进行中，请勿重复触发'}), 409
    threading.Thread(target=state._run_aux_data_update, daemon=True).start()
    return jsonify({'status': 'started', 'message': '辅助数据流水线已启动'})


@bp.route('/api/update_aux_data/status')
def api_update_aux_data_status():
    return jsonify({
        'running': state._AUX_UPDATE_STATUS['running'],
        'stage': state._AUX_UPDATE_STATUS['stage'],
        'message': state._AUX_UPDATE_STATUS['message'],
        'progress_pct': state._AUX_UPDATE_STATUS['progress_pct'],
        'error': state._AUX_UPDATE_STATUS['error'],
    })


@bp.route('/api/update_aux_data/check')
def api_update_aux_data_check():
    """只读判定 FRED / A股估值 / risk_signals 是否落后于最新交易日。"""
    return jsonify(state._check_aux_data_freshness())


# ── 衍生因子流水线（行业周度热度等，依赖最新 stock_data） ──
@bp.route('/api/update_factor_data', methods=['POST'])
def api_update_factor_data():
    if state._FACTOR_UPDATE_STATUS.get('running'):
        return jsonify({'error': '衍生因子重算正在进行中，请勿重复触发'}), 409
    threading.Thread(target=state._run_factor_update, daemon=True).start()
    return jsonify({'status': 'started', 'message': '衍生因子流水线已启动'})


@bp.route('/api/update_factor_data/status')
def api_update_factor_data_status():
    return jsonify({
        'running': state._FACTOR_UPDATE_STATUS['running'],
        'stage': state._FACTOR_UPDATE_STATUS['stage'],
        'message': state._FACTOR_UPDATE_STATUS['message'],
        'progress_pct': state._FACTOR_UPDATE_STATUS['progress_pct'],
        'error': state._FACTOR_UPDATE_STATUS['error'],
    })


@bp.route('/api/update_factor_data/check')
def api_update_factor_data_check():
    """只读判定衍生因子（sector_weekly_heat）是否落后于最新 stock_data 月度。"""
    return jsonify(state._check_factor_data_freshness())


@bp.route('/api/restart', methods=['POST'])
def api_restart():
    """自重启 Flask：先检查是否有 in-flight 数据更新任务（防止中断 IO），
    然后落盘 cache、用 detached subprocess + os._exit 替换进程。前端可用 /api/info 轮询恢复。
    """
    if state._UPDATE_DATA_STATUS.get('running'):
        return jsonify({
            'error': 'restart_conflict',
            'message': '股票数据更新正在进行中，请等其完成后再重启'
        }), 409
    if state._INDEX_UPDATE_STATUS.get('stage') == 'running':
        return jsonify({
            'error': 'restart_conflict',
            'message': '指数 / ETF 数据更新正在进行中，请等其完成后再重启'
        }), 409
    if state._AUX_UPDATE_STATUS.get('running'):
        return jsonify({
            'error': 'restart_conflict',
            'message': '辅助数据（FRED / A股估值 / risk_signals）更新正在进行中，请等其完成后再重启'
        }), 409
    if state._FACTOR_UPDATE_STATUS.get('running'):
        return jsonify({
            'error': 'restart_conflict',
            'message': '衍生因子（行业周度热度等）重算正在进行中，请等其完成后再重启'
        }), 409
    state._schedule_self_restart()
    return jsonify({'status': 'restarting', 'message': '服务即将重启（约 1.5 秒后断开）'})
