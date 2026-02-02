# Analysis Tools

This guide covers the Edge Analysis, Rule Tester, and Vulnerability Modeler applications—tools for understanding why your strategy performs the way it does.

---

## Edge Analysis

### Purpose

Validate entry quality by measuring how price moves after entry using E-ratio calculations.

### Launching

```bash
python ctk_edge_analysis_gui.py
```

### Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       EDGE ANALYSIS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  [E-Ratio] [R-Multiple] [Validation] [Trade Details]           ◄── Tabs    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Configuration                                                              │
│  ─────────────                                                             │
│  Price Data Path: [raw_data/daily/          ] [Browse]                     │
│  Max Days:        [ 30 ]                                                   │
│                                                                             │
│  Trade Logs                                                                │
│  ──────────                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ ☑ logs/backtests/portfolio/test1/portfolio_trades.csv   (47 trades)│   │
│  │ ☑ logs/backtests/portfolio/test2/portfolio_trades.csv   (52 trades)│   │
│  │ ☐ logs/backtests/portfolio/test3/portfolio_trades.csv   (38 trades)│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  [Add Files]  [Remove]  [Calculate E-Ratio]                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### E-Ratio Tab

Shows aggregate E-ratio across all trades:

```
E-Ratio by Holding Period
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  1.6 ┤                                                                     │
│      │            ╭─────╮                                                  │
│  1.4 ┤         ╭──╯     ╰──╮                                               │
│      │      ╭──╯           ╰──╮                                            │
│  1.2 ┤   ╭──╯                 ╰──────────────                              │
│      │╭──╯                                                                 │
│  1.0 ┼──────────────────────────────────────  ← Random baseline           │
│      │                                                                     │
│  0.8 ┤                                                                     │
│      ┼────┬────┬────┬────┬────┬────┬────┬────                             │
│      1    5    10   15   20   25   30                                      │
│                   Days After Entry                                          │
│                                                                             │
│  Peak E-ratio: 1.42 at day 8                                               │
│  Interpretation: Entries have positive edge, optimal holding ~8 days       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### R-Multiple Tab

Distribution of trade outcomes normalized by risk:

```
R-Multiple Distribution
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  Winners                          Losers                                   │
│  ┌─────────────────────┐         ┌─────────────────────┐                  │
│  │        ██           │         │  ██                 │                  │
│  │     ██ ██           │         │  ██ ██              │                  │
│  │  ██ ██ ██ ██        │         │  ██ ██ ██           │                  │
│  │  ██ ██ ██ ██ ██     │         │  ██ ██ ██ ██        │                  │
│  ├─────────────────────┤         ├─────────────────────┤                  │
│  │ 1R  2R  3R  4R  5R+ │         │ -1R -2R -3R -4R     │                  │
│  └─────────────────────┘         └─────────────────────┘                  │
│                                                                             │
│  Win Rate: 55.3%                                                           │
│  Average Win: 2.1R                                                         │
│  Average Loss: -1.2R                                                       │
│  Expectancy: 0.62R per trade                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Validation Tab

Data quality and statistical checks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      VALIDATION REPORT                                      │
│                                                                             │
│  Summary: 23 passed │ 1 warning │ 0 errors                                 │
│                                                                             │
│  DATA CHECKS                                                               │
│  ───────────                                                               │
│  ✓ Price data integrity                    All files valid                │
│  ✓ ATR computation                         All positive                   │
│  ✓ Date alignment                          99 of 99 matched               │
│  ⚠ Entry price consistency                 3 trades >5% mismatch          │
│                                                                             │
│  AGGREGATION CHECKS                                                        │
│  ─────────────────                                                         │
│  ✓ Trade count per horizon                 Min 85 trades at all horizons  │
│  ✓ E-ratio bounds                          Range 0.95 to 1.42             │
│                                                                             │
│  BIAS CHECKS                                                               │
│  ───────────                                                               │
│  ✓ Sample consistency                      100% at all horizons           │
│  ✓ Outlier impact                          Raw vs winsorized: 8% diff     │
│                                                                             │
│  OUTLIER LOG                                                               │
│  ──────────                                                                │
│  Trade #23 (NVDA): Entry price 245.00 vs data 252.30 (3.0% diff)          │
│  Trade #41 (MSFT): Entry price 310.00 vs data 318.50 (2.7% diff)          │
│  Trade #67 (AAPL): Entry price 175.00 vs data 182.00 (3.8% diff)          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Rule Tester

### Purpose

Test variations of exit rules to find improvements without full strategy re-optimization.

### Launching

```bash
python ctk_rule_tester_gui.py
```

### How It Works

Takes existing trade entries and tests different exit rules:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RULE TESTER CONCEPT                                  │
│                                                                             │
│   Original Trade:                                                          │
│   ├── Entry: Buy AAPL at $150 on 2023-06-01                               │
│   ├── Original Exit: Sell at $158 on 2023-06-15 (strategy signal)         │
│   └── Original P/L: +5.3%                                                  │
│                                                                             │
│   Test Alternative Exits:                                                  │
│   ├── Rule A (Tighter stop): Exit at $147 on 2023-06-05 → P/L: -2.0%      │
│   ├── Rule B (Trailing stop): Exit at $155 on 2023-06-12 → P/L: +3.3%     │
│   ├── Rule C (Time exit): Exit at $152 on 2023-06-08 → P/L: +1.3%         │
│   └── Rule D (Target exit): Exit at $157 on 2023-06-10 → P/L: +4.7%       │
│                                                                             │
│   Compare results across all trades to find best rule                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXIT RULE CONFIGURATION                                │
│                                                                             │
│  Load Trade Log: [logs/backtests/.../trades.csv    ] [Browse]              │
│                                                                             │
│  EXIT RULES TO TEST                                                        │
│  ─────────────────                                                         │
│                                                                             │
│  ☑ Stop Loss Variations                                                    │
│     ATR Multiplier: [ 1.0 ] to [ 3.0 ] step [ 0.5 ]                       │
│                                                                             │
│  ☑ Trailing Stop                                                           │
│     Trail Percent: [ 2.0 ] to [ 8.0 ] step [ 1.0 ]                        │
│                                                                             │
│  ☑ Time-Based Exit                                                         │
│     Max Days: [ 5 ] to [ 20 ] step [ 5 ]                                  │
│                                                                             │
│  ☑ Profit Target                                                           │
│     Target Percent: [ 3.0 ] to [ 10.0 ] step [ 1.0 ]                      │
│                                                                             │
│  ☐ Combined Rules                                                          │
│                                                                             │
│  [Run Tests]                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      RULE TEST RESULTS                                      │
│                                                                             │
│  Exit Rule               Return    Sharpe    Win Rate    Avg Trade         │
│  ─────────────────────────────────────────────────────────────────         │
│  Original                +24.5%     1.42      55.3%       +0.52%           │
│  ─────────────────────────────────────────────────────────────────         │
│  Stop Loss 1.5 ATR       +18.2%     1.28      48.9%       +0.39%           │
│  Stop Loss 2.0 ATR       +22.1%     1.38      52.1%       +0.47%           │
│  Stop Loss 2.5 ATR       +23.8%     1.40      54.3%       +0.51%  ★       │
│  ─────────────────────────────────────────────────────────────────         │
│  Trailing 4%             +26.8%     1.51      58.5%       +0.57%  ★Best   │
│  Trailing 6%             +25.2%     1.45      56.4%       +0.54%           │
│  ─────────────────────────────────────────────────────────────────         │
│  Time Exit 10 days       +19.5%     1.22      51.1%       +0.42%           │
│  Time Exit 15 days       +22.8%     1.35      53.2%       +0.49%           │
│  ─────────────────────────────────────────────────────────────────         │
│                                                                             │
│  Recommendation: 4% Trailing Stop improves Sharpe by 6.3%                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vulnerability Modeler

### Purpose

Optimize position scoring for capital contention in portfolio mode.

### Launching

```bash
python ctk_vulnerability_gui.py
```

### What It Does

When multiple signals occur but capital/slots are limited, vulnerability scoring decides which positions to keep or close:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VULNERABILITY SCORING                                    │
│                                                                             │
│   Current Positions (max 3):                                               │
│   ├── AAPL: Score 0.72 (profitable, 2 days old)                           │
│   ├── MSFT: Score 0.58 (breakeven, 5 days old)                            │
│   └── NVDA: Score 0.35 (losing, 8 days old)      ◄── Weakest              │
│                                                                             │
│   New Signal:                                                              │
│   └── GOOGL: Score 0.68 (strong setup)                                    │
│                                                                             │
│   Decision: Close NVDA (0.35), Open GOOGL (0.68)                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Score Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    VULNERABILITY SCORE FACTORS                              │
│                                                                             │
│   Factor              Weight      Description                              │
│   ─────────────────────────────────────────────────────────────           │
│   Profitability       [ 0.40 ]    Current unrealized P/L                  │
│   Age                 [ 0.20 ]    Days since entry (older = weaker)       │
│   Distance to Stop    [ 0.15 ]    Risk remaining in position              │
│   Momentum            [ 0.15 ]    Recent price direction                  │
│   Signal Strength     [ 0.10 ]    Entry signal quality                    │
│                                                                             │
│   Total               [ 1.00 ]                                             │
│                                                                             │
│   [Load Preset ▼]  [Save Preset]  [Optimize Weights]                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Optimization

The modeler can optimize weights using historical data:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WEIGHT OPTIMIZATION                                      │
│                                                                             │
│   Load Portfolio Backtest: [logs/backtests/portfolio/test1/] [Browse]      │
│                                                                             │
│   Optimization Results:                                                    │
│   ─────────────────────                                                    │
│   Original Weights:     Sharpe = 1.28    Return = +18.5%                  │
│   Optimized Weights:    Sharpe = 1.41    Return = +21.2%                  │
│                                                                             │
│   Improvement: +10.1% Sharpe, +14.6% Return                                │
│                                                                             │
│   Optimized Weight Values:                                                 │
│   ├── Profitability:    0.35 (was 0.40)                                   │
│   ├── Age:              0.25 (was 0.20)                                   │
│   ├── Distance to Stop: 0.18 (was 0.15)                                   │
│   ├── Momentum:         0.12 (was 0.15)                                   │
│   └── Signal Strength:  0.10 (unchanged)                                  │
│                                                                             │
│   [Apply Optimized Weights]  [Export Preset]                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visualization

Compare scoring approaches:

```
Portfolio Value: Scoring Comparison
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│  $130k ┤                                          ╭─── Optimized Scoring   │
│        │                                    ╭─────╯                        │
│  $120k ┤                              ╭─────╯                              │
│        │                        ╭─────╯      ╭─────── Default Scoring      │
│  $110k ┤                  ╭─────╯      ╭─────╯                              │
│        │            ╭─────╯      ╭─────╯                                    │
│  $100k ┼────────────╯      ╭─────╯      ╭───────────── No Scoring          │
│        │                  ╭╯                                                │
│   $90k ┤                                                                   │
│        ┼────────┬────────┬────────┬────────┬────────┬────────              │
│        Jan     Mar      May      Jul      Sep      Nov                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## When to Use Each Tool

| Situation | Tool |
|-----------|------|
| "Are my entries actually good?" | Edge Analysis |
| "What holding period is optimal?" | Edge Analysis |
| "Can I improve exits without changing entry?" | Rule Tester |
| "Which stop loss level works best?" | Rule Tester |
| "How should I prioritize positions?" | Vulnerability Modeler |
| "Capital contention hurting returns?" | Vulnerability Modeler |

---

## Related Documentation

- [Edge Analysis Concepts](../concepts/EDGE_ANALYSIS.md) — E-ratio theory
- [Vulnerability Scoring Concepts](../concepts/VULNERABILITY_SCORING.md) — Scoring methodology
- [Backtest GUI](BACKTEST_GUI.md) — Running backtests to generate trade logs
- [Factor Analysis](FACTOR_ANALYSIS_GUI.md) — Deeper trade analysis
