---
name: timing-etf-sync
description: Refresh locally cached timing ETF data for the quant timing page, rebuild server-side timing caches, and verify the live /timing UI with screenshots.
---

Use this skill when the user wants the latest ETF data for the timing page or when timing ETF trade details look stale.

## Rule

Keep the workflow local. ETF data is fetched into project-local CSV cache files, timing backtests are recomputed in Flask/Pandas, and the frontend only renders the returned JSON.

## Required workflow

1. Ensure the Flask app is running at `http://localhost:8080`.
2. Trigger `POST /api/update_index_data` so index data and timing ETF daily caches refresh together.
3. Poll `/api/update_index_data/status` until the refresh finishes.
4. Confirm the dedicated ETF cache directory contains the three files under `stock_trade_demo/.cache/timing_etf/`.
5. Open `http://localhost:8080/timing` and confirm the three cards rerender from the refreshed local data.
6. Use the local `frontend-screenshot-verify` workflow to capture screenshot evidence of the live page.

## Expected cache layout

- `stock_trade_demo/.cache/timing_etf/csi1000_etf_daily.csv`
- `stock_trade_demo/.cache/timing_etf/chinext_etf_daily.csv`
- `stock_trade_demo/.cache/timing_etf/star50_etf_daily.csv`

## Notes

- The timing page button may be labeled “刷新指数/ETF数据”; it is expected to refresh both benchmark index data and timing ETF cache.
- If old ETF CSVs still exist in `stock_trade_demo/.cache/`, the backend may migrate them into `stock_trade_demo/.cache/timing_etf/` on first use.
- If browser automation fails, report BLOCKED rather than relying only on API output.
