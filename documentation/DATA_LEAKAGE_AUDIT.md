# Data Leakage and Look-Ahead Bias Audit Report

**Date:** 2025-11-30
**Framework:** BackTestingFramework
**Auditor:** Claude Code

## Executive Summary

This audit examined the BackTestingFramework for potential data leakage and look-ahead bias issues. While the framework has good design intentions with documentation about preventing look-ahead bias, **several critical vulnerabilities were identified** that could allow strategies to inadvertently or deliberately use future information, leading to unrealistic backtest results.

### Severity Classification
- 🔴 **CRITICAL**: Can directly cause look-ahead bias, must fix
- 🟡 **WARNING**: Risk of look-ahead bias through misuse
- 🟢 **GOOD**: Properly implemented without bias concerns

---

## Critical Issues Found

### 🔴 CRITICAL #1: Full Dataset Exposure in StrategyContext

**Location:** `Classes/Strategy/strategy_context.py:34`

**Issue:** The `StrategyContext` passes the **entire dataset** to strategies via the `data` attribute, including all future bars. While helper methods enforce temporal boundaries, strategies can bypass these and directly access future data.

**Vulnerable Code:**
```python
@dataclass(frozen=True)
class StrategyContext:
    data: pd.DataFrame  # FULL DATASET - contains future data!
    current_index: int  # Current position
    ...
```

**How it can be exploited:**
```python
# In a strategy's generate_signal() method:
def generate_signal(self, context: StrategyContext) -> Signal:
    # BAD: Can access future data!
    future_price = context.data.iloc[context.current_index + 1]['close']

    # BAD: Can peek ahead multiple bars
    next_week_prices = context.data.iloc[context.current_index:context.current_index+7]['close']

    # This would give unrealistic trading signals!
    if context.current_price < future_price:
        return Signal.buy(...)
```

**Current Protection:** Documentation only - relies on developers to:
1. Only use helper methods (`get_bar()`, `get_indicator_value()`)
2. Never directly index `context.data` beyond `current_index`

**Risk Level:** CRITICAL - Easy to accidentally introduce look-ahead bias

---

### 🔴 CRITICAL #2: prepare_data() Receives Full Dataset

**Location:** `Classes/Strategy/base_strategy.py:63`, `Classes/Engine/single_security_engine.py:88`

**Issue:** The `prepare_data()` method receives the **entire dataset** before backtesting begins. While rolling/expanding operations are safe, strategies could use non-causal operations that leak future information into past indicators.

**Vulnerable Code:**
```python
# In single_security_engine.py:88
data = strategy.prepare_data(data)  # Full dataset passed to strategy!

# Process each bar
for i in range(total_bars):
    ...
```

**How it can be exploited:**
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # BAD: Using future data!
    df['future_return'] = df['close'].shift(-5)  # Looks 5 days ahead!

    # BAD: Using full dataset statistics
    df['z_score'] = (df['close'] - df['close'].mean()) / df['close'].std()
    # This uses mean/std from entire period, including future data!

    # BAD: Using future peaks/troughs
    df['next_peak'] = df['high'].rolling(window=20).max().shift(-20)

    return df
```

**Safe Usage:**
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()

    # GOOD: Rolling operations (look backward only)
    df['sma_20'] = df['close'].rolling(window=20).mean()

    # GOOD: Expanding operations (cumulative from start)
    df['expanding_max'] = df['high'].expanding().max()

    return df
```

**Current Protection:** Documentation suggests using rolling/expanding operations, but doesn't enforce it.

**Risk Level:** CRITICAL - Can subtly introduce look-ahead bias that's hard to detect

---

### 🔴 CRITICAL #3: Engine Inconsistency - Data Handling

**Location:**
- `Classes/Engine/single_security_engine.py:112-113`
- `Classes/Engine/portfolio_engine.py:131-132`

**Issue:** The two engines handle data differently:

**Single Security Engine (MORE VULNERABLE):**
```python
# Line 112-113
context = StrategyContext(
    data=data,  # FULL DATASET!
    current_index=i,  # Positional index
    ...
)
```

**Portfolio Engine (SAFER):**
```python
# Line 125
historical_data = data.iloc[:bar_index+1].copy()

# Line 131-132
context = StrategyContext(
    data=historical_data,  # Only historical data!
    current_index=bar_index,
    ...
)
```

**Impact:**
- Single security backtests are MORE vulnerable to look-ahead bias
- Portfolio backtests have better protection (but still not perfect)
- **Inconsistent behavior** between the two engines

**Risk Level:** CRITICAL - Creates confusion and inconsistent safety guarantees

---

### 🔴 CRITICAL #4: Index Type Mismatch in Portfolio Engine

**Location:** `Classes/Engine/portfolio_engine.py:124-133`

**Issue:** The portfolio engine mixes label-based and position-based indexing, which could cause incorrect data access.

**Problematic Code:**
```python
# Line 124 - Gets LABEL-based index
bar_index = data[date_mask].index[0]  # Returns DataFrame label (e.g., 42, or a date)

# Line 125 - Uses with POSITION-based iloc
historical_data = data.iloc[:bar_index+1].copy()  # Expects positional index!

# Line 133 - Uses as current_index in context
current_index=bar_index,  # This is a label, not a position!
```

**When it breaks:**
- If DataFrame has non-sequential index (e.g., after filtering)
- If DataFrame uses date-based index
- If DataFrame hasn't been reset after operations

**Example failure:**
```python
# If data has been filtered and index is [0, 5, 10, 15, 20, ...]
bar_index = data[date_mask].index[0]  # Returns 15 (label)
historical_data = data.iloc[:15+1]    # Gets POSITIONS 0-15, not labels!
# This would include too much data!
```

**Risk Level:** CRITICAL - Can cause incorrect data slicing and look-ahead bias

---

### 🔴 CRITICAL #5: No Train/Test Split in Optimization

**Location:** `Classes/Optimization/optimizer.py`

**Issue:** The optimizer tests all parameter combinations on the **entire dataset** without a train/test split. This leads to overfitting and unrealistic performance expectations.

**Problematic Code:**
```python
# Line 69-70
engine = SingleSecurityEngine(self.backtest_config)
backtest_result = engine.run(symbol, data, strategy)  # Full dataset!
```

**Why it's problematic:**
1. Parameters are optimized on the same data used for evaluation
2. Best parameters are overfitted to historical patterns
3. No out-of-sample validation
4. No walk-forward analysis

**Common Industry Practice:**
- 70/30 train/test split
- Walk-forward optimization
- Cross-validation for time series
- Out-of-sample validation period

**Risk Level:** CRITICAL - Optimization results are unreliable and overfitted

---

## Warnings and Concerns

### 🟡 WARNING #1: Reliance on Developer Discipline

**Issue:** The framework relies heavily on developers following best practices rather than enforcing them through code structure.

**Documentation vs. Enforcement:**
```python
# strategy_context.py:19-20 (documentation)
"""
Strategies must use current_index to access only historical data (0 to current_index)
to prevent lookahead bias. Helper methods enforce this automatically.
"""

# But the actual implementation doesn't enforce it!
data: pd.DataFrame  # Full dataset accessible
```

**Risk:** Human error, rushed development, or lack of understanding can easily introduce bias.

---

### 🟡 WARNING #2: No Data Validation for Causality

**Issue:** No validation checks to ensure that strategies/indicators only use causal operations.

**Missing Safeguards:**
- No checks for `.shift(-n)` with negative shifts
- No warnings when accessing future indices
- No validation of indicator calculations
- No automated testing for look-ahead bias

---

## Good Implementations Found

### 🟢 GOOD #1: Slippage Implementation

**Location:** `Classes/Engine/single_security_engine.py:339, 434, 497`

**Implementation:**
```python
# BUY orders - pay more due to slippage
execution_price = price * (1 + self.config.slippage_percent / 100)

# SELL orders - receive less due to slippage
execution_price = price * (1 - self.config.slippage_percent / 100)
```

**Why it's good:**
- Realistic transaction costs
- Conservative (reduces backtest performance)
- Consistently applied to all trades

---

### 🟢 GOOD #2: Order Execution Timing

**Location:** `Classes/Engine/single_security_engine.py:99`

**Implementation:**
```python
current_price = current_bar['close']  # Uses close price
```

**Why it's good:**
- Executes at close prices (realistic for EOD strategies)
- Doesn't use high/low prices which would be unrealistic
- Clear and consistent timing

---

### 🟢 GOOD #3: Stop Loss/Take Profit Timing

**Location:** `Classes/Engine/single_security_engine.py:124-159`

**Implementation:**
```python
# Check stop loss first (before strategy signal)
if self.position_manager.has_position:
    if strategy.should_check_stop_loss(context):
        if self.position_manager.check_stop_loss(current_price):
            # Stop loss hit - close position
            capital = self._close_position(...)
```

**Why it's good:**
- Risk management executes before strategy signals
- Uses current bar's price for checks
- Realistic priority (stops trigger first)

---

### 🟢 GOOD #4: Rolling Operations in AlphaTrend

**Location:** `strategies/alphatrend_strategy.py:161-202`

**Implementation:**
```python
# Vectorized rolling operations (causal)
positive_mf = df['positive_flow'].rolling(window=14).sum()
df['mfi_upper'] = df['mfi'].rolling(window=self.percentile_period).quantile(0.70)
df['vol_short_ma'] = df['volume'].rolling(window=self.volume_short_ma).mean()
```

**Why it's good:**
- Uses `.rolling()` which only looks backward
- Correctly implements moving averages and percentiles
- No use of `.shift(-n)` or future-looking operations

---

### 🟢 GOOD #5: Data Sorting

**Location:** `Classes/Data/data_loader.py:59`

**Implementation:**
```python
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)
```

**Why it's good:**
- Ensures chronological order
- Resets index to sequential positions
- Prevents temporal ordering issues

---

## Recommendations

### Priority 1: Critical Fixes (Implement Immediately)

#### 1.1 Restrict StrategyContext Data Access

**Option A: Pass only historical data slice**
```python
# In single_security_engine.py
for i in range(total_bars):
    # Only pass historical data up to current bar
    historical_data = data.iloc[:i+1].copy()

    context = StrategyContext(
        data=historical_data,
        current_index=i,  # Now this is the last index in historical_data
        ...
    )
```

**Option B: Create a read-only wrapper**
```python
class HistoricalDataView:
    """Read-only view that prevents future data access."""
    def __init__(self, data: pd.DataFrame, current_index: int):
        self._data = data
        self._current_index = current_index

    def iloc(self, index):
        if isinstance(index, slice):
            # Check slice doesn't go past current_index
            if index.stop is None or index.stop > self._current_index + 1:
                raise ValueError("Cannot access future data")
        elif index > self._current_index:
            raise ValueError("Cannot access future data")
        return self._data.iloc[index]
```

#### 1.2 Fix Portfolio Engine Index Issue

```python
# In portfolio_engine.py, replace lines 124-125
# OLD (BUGGY):
bar_index = data[date_mask].index[0]
historical_data = data.iloc[:bar_index+1].copy()

# NEW (FIXED):
bar_position = data[date_mask].index.tolist()[0]  # Get position, not label
if isinstance(bar_position, int):
    # If index is already positional
    historical_data = data.iloc[:bar_position+1].copy()
    current_bar_index = bar_position
else:
    # If index is label-based, get positional index
    current_bar_index = data.index.get_loc(bar_position)
    historical_data = data.iloc[:current_bar_index+1].copy()
```

**Better solution:** Ensure data is always position-indexed:
```python
# At the start of portfolio_engine.run():
for symbol, data in data_dict.items():
    # Ensure sequential positional index
    if not isinstance(data.index, pd.RangeIndex):
        data = data.reset_index(drop=True)
    data_dict[symbol] = data
```

#### 1.3 Unify Engine Behavior

Make both engines use the same data handling approach (prefer the safer portfolio engine approach):

```python
# In single_security_engine.py:88-123
# After prepare_data, modify the loop:
for i in range(total_bars):
    current_bar = data.iloc[i]
    current_date = current_bar['date']
    current_price = current_bar['close']

    # CRITICAL FIX: Only pass historical data
    historical_data = data.iloc[:i+1].copy()

    position_value = self.position_manager.get_position_value(current_price)
    position_value = self._convert_to_base_currency(position_value, symbol, current_date)
    total_equity = capital + position_value

    fx_rate = self._get_fx_rate(symbol, current_date)

    context = StrategyContext(
        data=historical_data,  # Only historical data!
        current_index=i,       # Now this is the last valid index
        current_price=current_price,
        current_date=current_date,
        position=self.position_manager.get_position(),
        available_capital=capital,
        total_equity=total_equity,
        symbol=symbol,
        fx_rate=fx_rate
    )
```

#### 1.4 Add prepare_data() Validation

Add validation to detect non-causal operations:

```python
# In base_strategy.py
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-calculate custom strategy-specific indicators before backtesting begins.

    CRITICAL: Only use CAUSAL operations that don't look ahead!
    - ✅ ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0
    - ❌ FORBIDDEN: .shift(-n) where n > 0, global .mean()/.std()
    """
    result = self._prepare_data_impl(data)

    # Validate no new NaN values at the END (indicates backward shift)
    if len(result) > 0:
        new_cols = set(result.columns) - set(data.columns)
        for col in new_cols:
            # Check last 10 rows for NaN (backward shift would create trailing NaNs)
            if result[col].iloc[-10:].isna().any():
                print(f"⚠️  WARNING: Column '{col}' has NaN values at end - possible look-ahead bias!")

    return result

@abstractmethod
def _prepare_data_impl(self, data: pd.DataFrame) -> pd.DataFrame:
    """Override this instead of prepare_data()."""
    return data
```

### Priority 2: Optimization Improvements

#### 2.1 Add Train/Test Split to Optimizer

```python
# In optimizer.py
class StrategyOptimizer:
    def __init__(self, config: OptimizationConfig, backtest_config: BacktestConfig,
                 train_test_split: float = 0.7):
        self.config = config
        self.backtest_config = backtest_config
        self.train_test_split = train_test_split

    def optimize(self, strategy_class: Type[BaseStrategy],
                param_grid: Dict[str, List[Any]],
                symbol: str, data: pd.DataFrame) -> OptimizationResults:
        """
        Optimize with train/test split to prevent overfitting.
        """
        # Split data
        split_idx = int(len(data) * self.train_test_split)
        train_data = data.iloc[:split_idx].copy()
        test_data = data.iloc[split_idx:].copy()

        # Optimize on training data
        train_results = self._optimize_on_data(
            strategy_class, param_grid, symbol, train_data
        )

        # Get best parameters
        best_params = train_results.get_best_parameters()

        # Validate on test data
        test_strategy = strategy_class(**best_params)
        engine = SingleSecurityEngine(self.backtest_config)
        test_result = engine.run(symbol, test_data, test_strategy)

        # Return both train and test results
        return OptimizationResults(
            train_results=train_results,
            test_result=test_result,
            best_parameters=best_params
        )
```

#### 2.2 Add Walk-Forward Analysis

```python
def walk_forward_optimize(self, strategy_class: Type[BaseStrategy],
                         param_grid: Dict[str, List[Any]],
                         symbol: str, data: pd.DataFrame,
                         train_period_bars: int = 252,
                         test_period_bars: int = 63,
                         step_bars: int = 63) -> WalkForwardResults:
    """
    Walk-forward optimization to prevent overfitting.

    Args:
        train_period_bars: Number of bars for training window
        test_period_bars: Number of bars for testing window
        step_bars: Number of bars to step forward
    """
    results = []

    start_idx = 0
    while start_idx + train_period_bars + test_period_bars <= len(data):
        # Define windows
        train_end = start_idx + train_period_bars
        test_end = train_end + test_period_bars

        train_data = data.iloc[start_idx:train_end]
        test_data = data.iloc[train_end:test_end]

        # Optimize on training window
        train_results = self._optimize_on_data(
            strategy_class, param_grid, symbol, train_data
        )
        best_params = train_results.get_best_parameters()

        # Test on out-of-sample data
        test_strategy = strategy_class(**best_params)
        engine = SingleSecurityEngine(self.backtest_config)
        test_result = engine.run(symbol, test_data, test_strategy)

        results.append({
            'train_start': train_data.iloc[0]['date'],
            'train_end': train_data.iloc[-1]['date'],
            'test_start': test_data.iloc[0]['date'],
            'test_end': test_data.iloc[-1]['date'],
            'parameters': best_params,
            'test_return': test_result.total_return_pct
        })

        # Step forward
        start_idx += step_bars

    return WalkForwardResults(results)
```

### Priority 3: Additional Safeguards

#### 3.1 Add Automated Look-Ahead Bias Detection

```python
def detect_lookahead_bias(backtest_result: BacktestResult,
                         strategy_name: str) -> List[str]:
    """
    Detect potential look-ahead bias by checking for unrealistic patterns.

    Returns list of warnings.
    """
    warnings = []

    # Check 1: Unrealistically high win rate
    if backtest_result.win_rate > 0.80:
        warnings.append(
            f"⚠️  Win rate {backtest_result.win_rate:.1%} is suspiciously high. "
            f"Check for look-ahead bias."
        )

    # Check 2: Perfect entries (buying at exact lows)
    trades = backtest_result.trades
    if len(trades) > 10:
        perfect_entries = 0
        for trade in trades:
            # Check if entry was within 1% of the low in next 5 bars
            # This would be suspicious
            pass  # Implement logic

    # Check 3: Sharpe ratio too high
    sharpe = calculate_sharpe_ratio(backtest_result.equity_curve)
    if sharpe > 3.0:
        warnings.append(
            f"⚠️  Sharpe ratio {sharpe:.2f} is extremely high. "
            f"Verify no look-ahead bias."
        )

    return warnings
```

#### 3.2 Add Data Integrity Checks

```python
# In data_loader.py
def load_csv(self, symbol: str, required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Load CSV with integrity checks."""
    df = pd.read_csv(file_path)

    # ... existing code ...

    # NEW: Check for data integrity
    self._validate_data_integrity(df, symbol)

    return df

def _validate_data_integrity(self, df: pd.DataFrame, symbol: str) -> None:
    """Validate data doesn't have future leakage issues."""

    # Check 1: Date ordering
    if not df['date'].is_monotonic_increasing:
        raise ValueError(f"{symbol}: Dates are not in ascending order!")

    # Check 2: No duplicate dates
    if df['date'].duplicated().any():
        raise ValueError(f"{symbol}: Duplicate dates found!")

    # Check 3: OHLC relationship (High >= Low, High >= Open/Close, Low <= Open/Close)
    if 'high' in df.columns and 'low' in df.columns:
        if (df['high'] < df['low']).any():
            raise ValueError(f"{symbol}: High < Low detected!")
        if 'close' in df.columns:
            if (df['high'] < df['close']).any() or (df['low'] > df['close']).any():
                raise ValueError(f"{symbol}: Invalid OHLC relationship!")
```

### Priority 4: Documentation Improvements

#### 4.1 Add Prominent Warnings

Add to README.md:

```markdown
## ⚠️  CRITICAL: Preventing Look-Ahead Bias

This framework passes the full dataset to strategies for performance reasons.
**You MUST follow these rules to prevent look-ahead bias:**

### In `generate_signal()`:
- ❌ **NEVER** access `context.data.iloc[context.current_index + n]` where n > 0
- ❌ **NEVER** use `context.data.loc[future_date]`
- ✅ **ALWAYS** use `context.get_bar(offset)` with offset <= 0
- ✅ **ALWAYS** use `context.get_indicator_value(column, offset)` with offset <= 0

### In `prepare_data()`:
- ❌ **NEVER** use `.shift(-n)` where n > 0 (looks ahead)
- ❌ **NEVER** use global statistics (`.mean()`, `.std()` on full series)
- ✅ **ALWAYS** use `.rolling()` or `.expanding()` operations
- ✅ **ALWAYS** use `.shift(n)` where n >= 0 (looks back or stays at current)

### Example of CORRECT usage:
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # ✅ GOOD: Rolling operations look backward
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['returns'] = df['close'].pct_change()  # Uses shift(1) internally
    return df
```

### Example of INCORRECT usage:
```python
def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    # ❌ BAD: Looks ahead 5 bars!
    df['future_high'] = df['high'].shift(-5)
    # ❌ BAD: Uses statistics from entire dataset!
    df['z_score'] = (df['close'] - df['close'].mean()) / df['close'].std()
    return df
```
```

#### 4.2 Add Strategy Template with Comments

Create `strategies/template_strategy.py`:

```python
"""
Template strategy with best practices for avoiding look-ahead bias.
"""
from typing import List
import pandas as pd
from Classes.Strategy.base_strategy import BaseStrategy
from Classes.Strategy.strategy_context import StrategyContext
from Classes.Models.signal import Signal


class TemplateStrategy(BaseStrategy):
    """
    Template strategy demonstrating correct practices.

    CRITICAL: This template shows how to avoid look-ahead bias!
    """

    def required_columns(self) -> List[str]:
        return ['date', 'open', 'high', 'low', 'close', 'volume']

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Pre-calculate indicators ONCE before backtesting.

        ⚠️  ONLY USE CAUSAL OPERATIONS!
        - ✅ ALLOWED: .rolling(), .expanding(), .shift(n) where n >= 0
        - ❌ FORBIDDEN: .shift(-n) where n > 0, global .mean()/.std()
        """
        df = data.copy()

        # ✅ GOOD: Rolling mean (looks backward)
        df['sma_20'] = df['close'].rolling(window=20).mean()

        # ✅ GOOD: Expanding max (cumulative from start)
        df['expanding_high'] = df['high'].expanding().max()

        # ✅ GOOD: Price change (uses shift(1) internally)
        df['returns'] = df['close'].pct_change()

        # ❌ BAD EXAMPLE (DO NOT DO THIS):
        # df['future_return'] = df['close'].shift(-5)  # LOOK-AHEAD BIAS!

        return df

    def generate_signal(self, context: StrategyContext) -> Signal:
        """
        Generate trading signal based on current context.

        ⚠️  ONLY ACCESS HISTORICAL DATA!
        - ✅ ALLOWED: context.get_bar(offset) where offset <= 0
        - ❌ FORBIDDEN: context.data.iloc[context.current_index + n] where n > 0
        """
        # ✅ GOOD: Access current bar
        current_price = context.current_price
        sma_20 = context.get_indicator_value('sma_20')

        # ✅ GOOD: Access previous bar
        prev_price = context.get_indicator_value('close', offset=-1)

        # ❌ BAD EXAMPLE (DO NOT DO THIS):
        # next_price = context.data.iloc[context.current_index + 1]['close']  # LOOK-AHEAD BIAS!

        # Entry logic
        if not context.has_position:
            if sma_20 and current_price > sma_20:
                return Signal.buy(size=0.5, reason="Price above SMA")

        # Exit logic
        else:
            if sma_20 and current_price < sma_20:
                return Signal.sell(reason="Price below SMA")

        return Signal.hold()
```

---

## Testing Recommendations

### 1. Create Look-Ahead Bias Test Suite

```python
# tests/test_lookahead_bias.py
import pytest
from Classes.Strategy.strategy_context import StrategyContext

def test_context_prevents_future_access():
    """Test that accessing future data raises an error."""
    # This test will FAIL with current implementation!
    # It should PASS after implementing the fix

    # Create test data
    data = create_test_data(100)

    context = StrategyContext(
        data=data,
        current_index=50,
        ...
    )

    # Should raise error when trying to access future data
    with pytest.raises(ValueError, match="Cannot access future data"):
        future_bar = context.data.iloc[51]

def test_prepare_data_validation():
    """Test that prepare_data detects non-causal operations."""

    class BadStrategy(BaseStrategy):
        def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
            df = data.copy()
            df['future_price'] = df['close'].shift(-5)  # Look-ahead!
            return df

    # Should detect and warn about look-ahead bias
    strategy = BadStrategy()
    data = create_test_data(100)

    with pytest.warns(UserWarning, match="possible look-ahead bias"):
        result = strategy.prepare_data(data)
```

### 2. Add Regression Tests

```python
def test_portfolio_vs_single_security_consistency():
    """Test that both engines produce similar results."""

    # Run same strategy on same data with both engines
    single_result = run_single_security_backtest(...)
    portfolio_result = run_portfolio_backtest(...)  # With 1 security

    # Results should be very similar (within tolerance for numerical precision)
    assert abs(single_result.total_return - portfolio_result.total_return) < 0.01
```

### 3. Create Benchmark Test

```python
def test_realistic_performance():
    """Test that backtest results are realistic (not too good to be true)."""

    result = run_backtest(...)

    # Realistic win rate (40-65%)
    assert 0.40 <= result.win_rate <= 0.65, \
        f"Win rate {result.win_rate:.1%} is suspicious"

    # Realistic Sharpe ratio (<2.5 for most strategies)
    sharpe = calculate_sharpe_ratio(result.equity_curve)
    assert sharpe < 2.5, \
        f"Sharpe ratio {sharpe:.2f} is suspiciously high"

    # Realistic max drawdown (>5%)
    assert result.max_drawdown_pct > 5.0, \
        f"Max drawdown {result.max_drawdown_pct:.1%} is too small"
```

---

## Summary Table

| Issue | Severity | Location | Impact | Fix Priority |
|-------|----------|----------|--------|--------------|
| Full dataset exposure in StrategyContext | 🔴 CRITICAL | `strategy_context.py:34` | Direct look-ahead bias risk | P1 |
| prepare_data() receives full dataset | 🔴 CRITICAL | `base_strategy.py:63` | Subtle look-ahead bias risk | P1 |
| Engine inconsistency | 🔴 CRITICAL | Both engines | Confusion, inconsistent safety | P1 |
| Index type mismatch | 🔴 CRITICAL | `portfolio_engine.py:124` | Incorrect data slicing | P1 |
| No train/test split | 🔴 CRITICAL | `optimizer.py` | Overfitting, unrealistic results | P2 |
| Developer discipline reliance | 🟡 WARNING | Framework-wide | Human error risk | P3 |
| No causality validation | 🟡 WARNING | Framework-wide | Undetected look-ahead bias | P3 |
| Slippage implementation | 🟢 GOOD | Both engines | Realistic costs | - |
| Order execution timing | 🟢 GOOD | Both engines | Realistic timing | - |
| Stop loss/take profit timing | 🟢 GOOD | Both engines | Proper priority | - |

---

## Conclusion

The BackTestingFramework has **good architectural intentions** but **critical implementation gaps** that make it vulnerable to data leakage and look-ahead bias. The main issues are:

1. **Full dataset exposure** without enforcement of temporal boundaries
2. **Inconsistent behavior** between engines
3. **No train/test split** in optimization
4. **Over-reliance on documentation** instead of code enforcement

These issues are **fixable** with the recommendations provided above. The Priority 1 fixes should be implemented immediately to ensure backtest reliability.

**Current Risk Level:** 🔴 **HIGH** - Backtests may have unrealistic results
**Risk Level After P1 Fixes:** 🟡 **MEDIUM** - Significantly improved but requires developer awareness
**Risk Level After All Fixes:** 🟢 **LOW** - Industry-standard protection against bias

---

## Next Steps

1. **Immediate:** Implement Priority 1 fixes (Critical Issues #1-4)
2. **Short-term:** Add train/test split to optimizer (Priority 2)
3. **Medium-term:** Add validation and safeguards (Priority 3)
4. **Ongoing:** Improve documentation and create test suite (Priority 4)

---

**Report End**
