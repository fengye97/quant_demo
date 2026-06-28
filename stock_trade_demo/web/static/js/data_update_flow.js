// Shared data-update flow for /index, /timing, /us_timing.
// Exports window.DataUpdateFlow.run({ type, onReload }).
//
// Flow:
//   1. GET   /api/<type>/check       → pre-confirm if no update needed
//   2. POST  /api/<type>             → trigger
//   3. POLL  /api/<type>/status      → show progress modal
//   4. on done → call onReload() (page-specific data refresh)
//   5. show "建议重启" banner with [立即重启] [稍后]
//   6. on restart click → POST /api/restart, show overlay, poll /api/info,
//      reload page once it's back

(function () {
  'use strict';

  const ENDPOINTS = {
    stock: {
      check: '/api/update_data/check',
      trigger: '/api/update_data',
      status: '/api/update_data/status',
      label: 'A 股日线 + 月度面板',
      // /api/update_data/status returns { progress_pct, stage, message, error, running }
      normalize: (s) => ({
        progress: Number(s.progress_pct || 0),
        stage: s.stage || 'idle',
        message: s.message || '',
        done: s.stage === 'done',
        error: s.stage === 'error' ? (s.error || s.message || '更新失败') : null,
      }),
    },
    index: {
      check: '/api/update_index_data/check',
      trigger: '/api/update_index_data',
      status: '/api/update_index_data/status',
      label: '指数 / ETF 日线',
      // /api/update_index_data/status returns { progress, stage, message, warning, details }
      normalize: (s) => ({
        progress: Number(s.progress || 0),
        stage: s.stage || 'idle',
        message: s.message || '',
        done: s.stage === 'done',
        error: s.stage === 'error' ? (s.message || '更新失败') : null,
      }),
    },
    aux: {
      check: '/api/update_aux_data/check',
      trigger: '/api/update_aux_data',
      status: '/api/update_aux_data/status',
      label: 'FRED 宏观 + A 股估值 + 风险信号',
      normalize: (s) => ({
        progress: Number(s.progress_pct || 0),
        stage: s.stage || 'idle',
        message: s.message || '',
        done: s.stage === 'done',
        error: s.stage === 'error' ? (s.error || s.message || '更新失败') : null,
      }),
    },
    factor: {
      check: '/api/update_factor_data/check',
      trigger: '/api/update_factor_data',
      status: '/api/update_factor_data/status',
      label: '行业周度热度',
      normalize: (s) => ({
        progress: Number(s.progress_pct || 0),
        stage: s.stage || 'idle',
        message: s.message || '',
        done: s.stage === 'done',
        error: s.stage === 'error' ? (s.error || s.message || '更新失败') : null,
      }),
    },
  };

  // 把秒数格式化成 "MM:SS" 或 "HH:MM:SS"（>=1 小时时才显示小时）
  function _fmtElapsed(ms) {
    const total = Math.max(0, Math.floor(ms / 1000));
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
    const pad = (n) => String(n).padStart(2, '0');
    if (h > 0) return `${h}:${pad(m)}:${pad(s)}`;
    return `${pad(m)}:${pad(s)}`;
  }

  // ── DOM utilities ─────────────────────────────────────────────────────────
  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const k of Object.keys(attrs)) {
        if (k === 'style') Object.assign(node.style, attrs[k]);
        else if (k === 'onclick') node.addEventListener('click', attrs[k]);
        else node.setAttribute(k, attrs[k]);
      }
    }
    if (children) {
      const list = Array.isArray(children) ? children : [children];
      for (const c of list) {
        if (c == null) continue;
        node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
      }
    }
    return node;
  }

  function ensureRoot(id) {
    let n = document.getElementById(id);
    if (n) return n;
    n = el('div', { id });
    document.body.appendChild(n);
    return n;
  }

  // ── Modal helpers ─────────────────────────────────────────────────────────
  function showConfirmDialog({ title, body, confirmText, cancelText }) {
    return new Promise((resolve) => {
      const overlay = el('div', {
        style: {
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)',
          zIndex: 10000, display: 'flex', alignItems: 'center', justifyContent: 'center',
        },
      });
      const box = el('div', {
        style: {
          background: '#222b3d', color: '#dfe6f0', padding: '20px 24px',
          borderRadius: '10px', minWidth: '360px', maxWidth: '520px',
          boxShadow: '0 10px 40px rgba(0,0,0,0.6)',
          fontFamily: 'system-ui, -apple-system, sans-serif',
        },
      });
      box.appendChild(el('div', { style: { fontSize: '15px', fontWeight: '600', marginBottom: '10px' } }, title));
      box.appendChild(el('div', { style: { fontSize: '13px', lineHeight: '1.6', marginBottom: '18px', color: '#aab4c4' } }, body));
      const btnRow = el('div', { style: { display: 'flex', gap: '10px', justifyContent: 'flex-end' } });
      const cancel = el('button', {
        style: {
          padding: '7px 16px', borderRadius: '6px', border: '1px solid #3a4660',
          background: 'transparent', color: '#dfe6f0', cursor: 'pointer', fontSize: '13px',
        },
        onclick: () => { overlay.remove(); resolve(false); },
      }, cancelText);
      const confirm = el('button', {
        style: {
          padding: '7px 16px', borderRadius: '6px', border: 'none',
          background: '#0984e3', color: '#fff', cursor: 'pointer', fontSize: '13px', fontWeight: '600',
        },
        onclick: () => { overlay.remove(); resolve(true); },
      }, confirmText);
      btnRow.appendChild(cancel);
      btnRow.appendChild(confirm);
      box.appendChild(btnRow);
      overlay.appendChild(box);
      document.body.appendChild(overlay);
    });
  }

  function showProgressModal() {
    const root = ensureRoot('dataflow-progress-modal');
    root.innerHTML = '';
    Object.assign(root.style, {
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.55)',
      zIndex: 10001, display: 'flex', alignItems: 'center', justifyContent: 'center',
    });
    const box = el('div', {
      style: {
        background: '#222b3d', color: '#dfe6f0', padding: '20px 24px',
        borderRadius: '10px', minWidth: '420px', maxWidth: '560px',
        boxShadow: '0 10px 40px rgba(0,0,0,0.6)',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      },
    });
    // 标题 + 右上角计时器（让用户能看到是真的在跑，没卡死）
    const titleRow = el('div', { style: { display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '12px', gap: '12px' } });
    const title = el('div', { id: 'dataflow-progress-title', style: { fontSize: '15px', fontWeight: '600' } }, '数据更新中...');
    const elapsed = el('div', { id: 'dataflow-progress-elapsed', style: { fontSize: '13px', color: '#7fb7e8', fontFamily: 'SF Mono, Menlo, monospace', fontVariantNumeric: 'tabular-nums' } }, '00:00');
    titleRow.appendChild(title);
    titleRow.appendChild(elapsed);

    const stageLabel = el('div', { id: 'dataflow-progress-stage', style: { fontSize: '12px', color: '#8fa3c0', marginBottom: '6px' } }, '');
    const message = el('div', { id: 'dataflow-progress-message', style: { fontSize: '13px', marginBottom: '12px', minHeight: '20px' } }, '正在启动...');
    const barOuter = el('div', { style: { background: '#1a2030', borderRadius: '6px', height: '10px', overflow: 'hidden' } });
    const barFill = el('div', { id: 'dataflow-progress-fill', style: { width: '0%', height: '100%', background: '#0984e3', transition: 'width 0.4s ease' } });
    barOuter.appendChild(barFill);
    const pctLabel = el('div', { id: 'dataflow-progress-pct', style: { fontSize: '12px', color: '#aab4c4', marginTop: '6px', textAlign: 'right' } }, '0%');
    const closeBtn = el('button', {
      id: 'dataflow-progress-close',
      style: {
        marginTop: '14px', padding: '7px 16px', borderRadius: '6px', border: '1px solid #3a4660',
        background: 'transparent', color: '#dfe6f0', cursor: 'pointer', fontSize: '13px', display: 'none',
      },
      onclick: () => { root.style.display = 'none'; },
    }, '关闭');

    box.appendChild(titleRow);
    box.appendChild(stageLabel);
    box.appendChild(message);
    box.appendChild(barOuter);
    box.appendChild(pctLabel);
    box.appendChild(closeBtn);
    root.appendChild(box);

    // 启动计时器（每 500ms 更新一次）
    const startTime = Date.now();
    let stopped = false;
    const tickTimer = setInterval(() => {
      if (stopped) return;
      elapsed.textContent = _fmtElapsed(Date.now() - startTime);
    }, 500);

    return {
      setTitle(t) { title.textContent = t; },
      setStage(s) { stageLabel.textContent = s; },
      setMessage(m) { message.textContent = m; },
      setProgress(p) { barFill.style.width = Math.max(0, Math.min(100, p)) + '%'; pctLabel.textContent = Math.round(p) + '%'; },
      setColor(c) { barFill.style.background = c; },
      showClose() { closeBtn.style.display = 'inline-block'; },
      hide() { root.style.display = 'none'; clearInterval(tickTimer); stopped = true; },
      stopTimer() {
        // 停止计时但保留显示，方便用户看到总耗时
        clearInterval(tickTimer);
        stopped = true;
        // 最后再 set 一次（避免最后 500ms 内未刷新）
        elapsed.textContent = _fmtElapsed(Date.now() - startTime);
      },
      elapsedMs() { return Date.now() - startTime; },
    };
  }

  function showRestartBanner({ message, onRestart, onLater }) {
    let banner = document.getElementById('dataflow-restart-banner');
    if (banner) banner.remove();
    banner = el('div', {
      id: 'dataflow-restart-banner',
      style: {
        position: 'fixed', top: '0', left: '0', right: '0', zIndex: 9999,
        padding: '10px 18px', background: 'linear-gradient(90deg,#3a2a1c 0%,#5a3a1c 100%)',
        color: '#fff8e8', fontSize: '13px', lineHeight: '1.5',
        boxShadow: '0 2px 10px rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', gap: '12px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      },
    });
    const text = el('div', { style: { flex: '1' } }, message);
    const restartBtn = el('button', {
      style: {
        padding: '6px 14px', borderRadius: '5px', border: 'none',
        background: '#e17055', color: '#fff', cursor: 'pointer', fontSize: '12.5px', fontWeight: '600',
      },
      onclick: () => onRestart(),
    }, '立即重启');
    const laterBtn = el('button', {
      style: {
        padding: '6px 14px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.3)',
        background: 'transparent', color: '#fff8e8', cursor: 'pointer', fontSize: '12.5px',
      },
      onclick: () => { if (onLater) onLater(); banner.remove(); },
    }, '稍后');
    banner.appendChild(text);
    banner.appendChild(restartBtn);
    banner.appendChild(laterBtn);
    document.body.appendChild(banner);
    return banner;
  }

  function showRestartOverlay(msg) {
    let overlay = document.getElementById('dataflow-restart-overlay');
    if (overlay) overlay.remove();
    overlay = el('div', {
      id: 'dataflow-restart-overlay',
      style: {
        position: 'fixed', inset: 0, background: 'rgba(15,20,30,0.92)',
        zIndex: 10002, display: 'flex', alignItems: 'center', justifyContent: 'center',
        color: '#dfe6f0', fontSize: '15px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      },
    });
    const card = el('div', {
      style: {
        background: '#222b3d', padding: '24px 32px', borderRadius: '10px',
        boxShadow: '0 10px 40px rgba(0,0,0,0.6)', textAlign: 'center', minWidth: '320px',
      },
    });
    card.appendChild(el('div', { style: { fontSize: '15px', fontWeight: '600', marginBottom: '12px' } }, '服务正在重启'));
    card.appendChild(el('div', { id: 'dataflow-restart-msg', style: { fontSize: '13px', color: '#aab4c4', minHeight: '20px' } }, msg || '请稍候，页面会在恢复后自动刷新...'));
    overlay.appendChild(card);
    document.body.appendChild(overlay);
    return overlay;
  }

  // ── Restart sequence ──────────────────────────────────────────────────────
  async function triggerRestart() {
    const overlay = showRestartOverlay('正在发送重启请求...');
    const msgNode = document.getElementById('dataflow-restart-msg');
    try {
      const resp = await fetch('/api/restart', { method: 'POST' });
      const data = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        msgNode.textContent = data.message || '重启失败：' + resp.status;
        setTimeout(() => overlay.remove(), 4000);
        return;
      }
      msgNode.textContent = '服务即将断开，正在等待重新上线...';
    } catch (e) {
      // network drop is expected when the server actually restarts; proceed to poll
      msgNode.textContent = '连接已断开，正在等待重新上线...';
    }
    // Poll /api/info until 200
    let attempts = 0;
    const maxAttempts = 60; // ~90s
    const poll = async () => {
      attempts++;
      try {
        const r = await fetch('/api/info', { cache: 'no-store' });
        if (r.ok) {
          msgNode.textContent = '已恢复，正在刷新页面...';
          setTimeout(() => location.reload(), 600);
          return;
        }
      } catch (_) { /* still down */ }
      if (attempts >= maxAttempts) {
        msgNode.textContent = '超时未恢复，请手动检查服务进程后刷新页面。';
        return;
      }
      setTimeout(poll, 1500);
    };
    setTimeout(poll, 2000); // give execv 2s before first poll
  }

  // ── 单个类型的执行（trigger + poll + ui 更新）；不创建/关闭 modal，由调用方管理 ──
  // 返回 { ok: bool, aborted_reason?, detail? }
  async function _executeOne(cfg, ui, { titlePrefix } = {}) {
    const prefix = titlePrefix || '';
    ui.setTitle(`${prefix}正在更新 ${cfg.label}`);
    ui.setStage('启动中');
    ui.setMessage('正在向后端发起更新请求...');
    ui.setProgress(0);
    ui.setColor('#0984e3');

    // trigger
    try {
      const r = await fetch(cfg.trigger, { method: 'POST' });
      const d = await r.json().catch(() => ({}));
      if (!r.ok) {
        ui.setColor('#d63031');
        ui.setMessage(d.error || `启动失败 (${r.status})`);
        return { ok: false, aborted_reason: 'trigger_failed', detail: d.error };
      }
    } catch (e) {
      ui.setColor('#d63031');
      ui.setMessage('网络错误：' + e.message);
      return { ok: false, aborted_reason: 'network_error', detail: e.message };
    }

    // poll
    const finalStatus = await new Promise((resolve) => {
      const timer = setInterval(async () => {
        try {
          const r = await fetch(cfg.status);
          const raw = await r.json();
          const s = cfg.normalize(raw);
          ui.setProgress(s.progress);
          ui.setMessage(s.message || `阶段: ${s.stage}`);
          ui.setStage(s.stage);
          if (s.done) { clearInterval(timer); ui.setColor('#00b894'); resolve(s); }
          else if (s.error) { clearInterval(timer); ui.setColor('#d63031'); ui.setMessage(s.error); resolve(s); }
        } catch (_) { /* keep polling */ }
      }, 2000);
    });

    if (finalStatus.error) {
      return { ok: false, aborted_reason: 'job_error', detail: finalStatus.error };
    }
    return { ok: true };
  }

  // 对一个 cfg 做前置 check；返回 { proceed: bool, payload }
  // proceed=false → user 取消（仅当 needs_update=false 且用户拒绝强刷）；true → 应跑
  async function _preCheck(cfg) {
    let needsUpdate = true;
    let payload = null;
    try {
      const r = await fetch(cfg.check);
      payload = await r.json();
      needsUpdate = !!payload.needs_update;
    } catch (e) {
      console.warn('[DataUpdateFlow] check failed for ' + cfg.label + ':', e);
    }
    return { needsUpdate, payload };
  }

  // ── 单类型流程（保留供 /timing /us_timing 用） ────────────────────────────
  async function run({ type, onReload }) {
    const cfg = ENDPOINTS[type];
    if (!cfg) throw new Error('Unknown DataUpdateFlow type: ' + type);

    const { needsUpdate, payload } = await _preCheck(cfg);
    if (!needsUpdate) {
      const localDate = (payload && payload.current_local_date) || '未知';
      const marketDate = (payload && payload.latest_market_date) || '未知';
      const ok = await showConfirmDialog({
        title: `${cfg.label} 已是最新`,
        body: `本地数据已更新至 ${localDate}（最新交易日 ${marketDate}）。是否仍要强制重新拉取？这通常需要几十秒到几分钟。`,
        confirmText: '仍然刷新',
        cancelText: '取消',
      });
      if (!ok) return { aborted: true, reason: 'no_update_needed' };
    }

    const ui = showProgressModal();
    const res = await _executeOne(cfg, ui);
    if (!res.ok) {
      ui.stopTimer();
      ui.showClose();
      return { aborted: true, reason: res.aborted_reason, detail: res.detail };
    }
    ui.setMessage('数据更新完成，正在刷新当前页面数据...');
    if (typeof onReload === 'function') {
      try { await onReload(); } catch (e) { console.warn('[DataUpdateFlow] onReload failed:', e); }
    }
    ui.stopTimer();
    ui.setMessage(`数据更新完成（总耗时 ${_fmtElapsed(ui.elapsedMs())}）`);
    ui.showClose();
    setTimeout(() => ui.hide(), 1500);
    showRestartBanner({
      message: `${cfg.label} 已更新到最新。部分缓存（US 择时、模块层常量、代码改动）需要重启服务才能全部生效。`,
      onRestart: triggerRestart,
    });
    return { aborted: false };
  }

  // 失败步骤的"仅重试失败步骤"banner（顶部红色，与重启 banner 区分）
  function showRetryFailedBanner({ failedTypes, onRetry }) {
    let banner = document.getElementById('dataflow-retry-banner');
    if (banner) banner.remove();
    banner = el('div', {
      id: 'dataflow-retry-banner',
      style: {
        position: 'fixed', top: '0', left: '0', right: '0', zIndex: 9998,
        padding: '10px 18px', background: 'linear-gradient(90deg,#3a1c1c 0%,#5a2a2a 100%)',
        color: '#ffe8e8', fontSize: '13px', lineHeight: '1.5',
        boxShadow: '0 2px 10px rgba(0,0,0,0.4)', display: 'flex', alignItems: 'center', gap: '12px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
      },
    });
    const labels = failedTypes.map((t) => ENDPOINTS[t]?.label || t).join(' / ');
    banner.appendChild(el('div', { style: { flex: '1' } },
      `${failedTypes.length} 个步骤失败：${labels}。多数是 akshare 上游瞬断，建议稍后重试。`));
    banner.appendChild(el('button', {
      style: {
        padding: '6px 14px', borderRadius: '5px', border: 'none',
        background: '#e17055', color: '#fff', cursor: 'pointer', fontSize: '12.5px', fontWeight: '600',
      },
      onclick: () => { banner.remove(); onRetry(failedTypes); },
    }, '重试失败步骤'));
    banner.appendChild(el('button', {
      style: {
        padding: '6px 14px', borderRadius: '5px', border: '1px solid rgba(255,255,255,0.3)',
        background: 'transparent', color: '#ffe8e8', cursor: 'pointer', fontSize: '12.5px',
      },
      onclick: () => banner.remove(),
    }, '关闭'));
    document.body.appendChild(banner);
    return banner;
  }

  // ── 多类型链式流程（/index 的 [更新数据] 用）────────────────────────────
  // 用同一个 modal 串跑多个类型；不论失败成功都跑完所有 step；末尾根据成功/失败状态展示对应 banner。
  // types 顺序很重要：先 'index'（拿到最新 ETF/index 日线）→ 然后 'aux'（FRED + A股估值）
  // → 最后 'stock'（A股 supplement，末尾会用最新 ETF rebuild timing cache）
  async function runChained({ types, onReload }) {
    if (!Array.isArray(types) || types.length === 0) {
      throw new Error('runChained requires non-empty types array');
    }
    const cfgs = types.map((t) => {
      const c = ENDPOINTS[t];
      if (!c) throw new Error('Unknown type: ' + t);
      return { type: t, cfg: c };
    });

    // ── Step A: 一次性 check 所有类型 ──
    const checks = await Promise.all(cfgs.map(({ cfg }) => _preCheck(cfg)));
    const anyNeeds = checks.some((x) => x.needsUpdate);

    // ── Step B: 全部已是最新 → 单个 confirm 弹窗问"是否仍强刷全部" ──
    if (!anyNeeds) {
      const lines = checks.map(({ payload }, i) => {
        const local = (payload && payload.current_local_date) || '未知';
        const market = (payload && payload.latest_market_date) || '未知';
        return `· ${cfgs[i].cfg.label}：本地 ${local} / 市场 ${market}`;
      }).join('\n');
      const ok = await showConfirmDialog({
        title: '全部数据已是最新',
        body: lines + '\n\n是否仍要按顺序强制重新拉取（指数/ETF → 辅助 → A 股）？整段约 3–8 分钟。',
        confirmText: '仍然刷新',
        cancelText: '取消',
      });
      if (!ok) return { aborted: true, reason: 'no_update_needed' };
    }

    // ── Step C: 单 modal 串跑；失败不 abort，继续下一步 ──
    const ui = showProgressModal();
    const total = cfgs.length;
    const stepResults = [];
    for (let i = 0; i < cfgs.length; i++) {
      const { cfg } = cfgs[i];
      const prefix = `[${i + 1}/${total}] `;
      const res = await _executeOne(cfg, ui, { titlePrefix: prefix });
      stepResults.push({ type: cfgs[i].type, label: cfg.label, ...res });
      // 失败就继续下一步；最后再汇总展示
    }

    // ── Step D: 汇总 ──
    const failed = stepResults.filter((r) => !r.ok);
    const succeeded = stepResults.filter((r) => r.ok);
    ui.stopTimer();
    const totalElapsed = _fmtElapsed(ui.elapsedMs());

    if (failed.length === 0) {
      // 全部成功
      ui.setMessage('全部完成，正在刷新当前页面数据...');
      if (typeof onReload === 'function') {
        try { await onReload(); } catch (e) { console.warn('[DataUpdateFlow] onReload failed:', e); }
      }
      ui.setTitle(`✓ ${total} 项数据全部刷新完成（总耗时 ${totalElapsed}）`);
      ui.setMessage(cfgs.map((c) => `· ${c.cfg.label}`).join('  '));
      ui.showClose();
      setTimeout(() => ui.hide(), 2000);
      showRestartBanner({
        message: `${cfgs.map((c) => c.cfg.label).join(' / ')} 已全部更新（耗时 ${totalElapsed}）。模块层常量与部分缓存（US 择时、Macro v3.3 因子等）需要重启服务才能完全生效。`,
        onRestart: triggerRestart,
      });
      return { aborted: false, steps: stepResults };
    }

    // 部分或全部失败
    ui.setColor('#d63031');
    ui.setTitle(`${succeeded.length}/${total} 成功，${failed.length} 失败（${totalElapsed}）`);
    const summary = stepResults.map((r) =>
      `${r.ok ? '✓' : '✗'} ${r.label}${r.ok ? '' : ' — ' + (r.detail || r.aborted_reason || 'unknown')}`
    ).join('\n');
    ui.setMessage(summary);
    ui.showClose();
    // 不自动 hide，让用户读完
    // 即使部分失败也跑 onReload —— 成功的 step 数据已落盘
    if (succeeded.length > 0 && typeof onReload === 'function') {
      try { await onReload(); } catch (e) { console.warn('[DataUpdateFlow] onReload failed:', e); }
    }
    showRetryFailedBanner({
      failedTypes: failed.map((r) => r.type),
      onRetry: (failedTypes) => runChained({ types: failedTypes, onReload }),
    });
    return { aborted: false, partial: true, steps: stepResults, failed: failed.map(r => r.type) };
  }

  window.DataUpdateFlow = { run, runChained, triggerRestart };
})();
