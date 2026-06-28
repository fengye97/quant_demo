---
description: Enforce the breakthrough-strategy workflow for stock_trade_demo after a strategy beats the prior return high or is promoted to the new default.
disable-model-invocation: true
argument-hint: [strategy-id and claim]
---

You are executing the `stock_trade_demo` post-breakthrough workflow.

User input: $ARGUMENTS

Use this skill whenever a strategy is claimed to have broken the prior return high, materially upgraded the retained production/default version, or needs to be formally promoted after a successful experiment.

Required workflow:

1. Read the relevant implementation, default parameters, and benchmark/baseline comparison.
2. Update `/Users/fatcat/Desktop/quant/STRATEGY_CHANGELOG.md` with:
   - what changed
   - why it counts as a breakthrough or retained winner
   - key parameter deltas
   - headline before/after metrics
   - meaningful tradeoffs or caveats
3. Cross-check `CLAUDE.md` rules before claiming success.
   - Confirm the strategy search guidance was respected.
   - Confirm timing execution semantics remain: signal at close(t), ETF fill at next trading day open(t+1), mark-to-market at next trading day close(t+1).
   - Confirm missing ETF history is not fabricated.
   - Confirm interval backtests still use full history before visible slicing.
   - Confirm the frontend remains a viewer over prepared/cached data rather than hidden request-path recomputation.
4. Verify the update is visible in the frontend.
   - Update the relevant backend metadata and template if no visible update surface exists yet.
   - Check the actual page, not just API payloads.
5. Capture screenshot evidence with the real UI before declaring the workflow complete.

Completion rule:
- Do not report success if any of these are missing: changelog update, CLAUDE.md rule cross-check, frontend-visible update, screenshot verification.

Repo-specific hints:
- A-share timing page: `/timing`
- US timing page: `/us_timing`
- Preferred visible update pattern: backend metadata in `web_app.py` + template rendering in `web/templates/*.html`
- Preferred evidence path: browser screenshot saved to a temporary path and cited in the final note.
