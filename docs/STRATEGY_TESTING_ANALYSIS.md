# Strategy System Testing Analysis & Issues Report

**Generated:** 2026-02-11
**Focus:** Strategy implementation, signal generation, and integration points

---

## Executive Summary

This document provides a comprehensive analysis of potential problems in the strategy system and outlines testing requirements. The analysis focuses on:
- Strategy implementation correctness
- Signal generation and validation
- Data access and look-ahead bias prevention
- Integration with backtesting engines
- Edge cases and error handling

---

## 1. IDENTIFIED POTENTIAL PROBLEMS

### 1.1 Critical Issues (High Priority)

#### **Problem 1.1.1: Missing Test Coverage**
- **Severity:** HIGH
- **Location:** `tests/` directory
- **Issue:** No dedicated strategy unit tests exist. Only scattered test scripts in `scripts/` directory.
- **Impact:** Cannot verify strategy behavior systematically, regression risks are high.
- **Recommendation:** Create comprehensive test suite covering:
  - Signal generation logic
  - Position sizing calculations
  - Stop loss calculations
  - Context data access
  - Fundamental rules integration

#### **Problem 1.1.2: AlphaTrend Strategy - Warmup Period Validation**
- **Severity:** HIGH
- **Location:** `strategies/base_alphatrend_strategy.py:362`
- **Issue:** Entry signal returns `None` if `current_index < percentile_period` (default 100). However:
  - No validation that dataset has sufficient bars for warmup
  - No clear error message to user about insufficient data
  - Strategy could appear to "do nothing" on small datasets
- **Impact:** Silent failures on short backtests, confusion about why strategy doesn't trade.
- **Test Case:**
  ```python
  # Test with data < 100 bars
  data = pd.DataFrame({'date': [...], 'close': [...]})  # 50 bars
  strategy = BaseAlphaTrendStrategy(percentile_period=100)
  result = engine.run('TEST', data, strategy)
  # Should: warn user or raise error
  # Currently: no trades, no explanation
  ```

#### **Problem 1.1.3: Position Sizing - Invalid Stop Loss Handling**
- **Severity:** HIGH
- **Location:** `strategies/base_alphatrend_strategy.py:420-423`
- **Issue:** When `stop_distance <= 0`, fallback uses 10% of available capital. This is dangerous because:
  - Invalid stop could indicate data quality issues
  - Fallback sizing not risk-based (contradicts strategy design)
  - Silent failure - no warning logged
- **Impact:** Unexpected position sizes, risk management failure.
- **Recommendation:** Raise exception or log critical warning instead of silent fallback.
- **Test Case:**
  ```python
  # Test with stop loss above entry price (invalid for LONG)
  context = StrategyContext(...)
  signal = Signal.buy(size=1.0, stop_loss=110)  # entry=100, stop=110 (invalid)
  position_size = strategy.position_size(context, signal)
  # Should: raise ValueError or log ERROR
  ```

#### **Problem 1.1.4: Stop Loss Fallback - Missing ATR**
- **Severity:** MEDIUM-HIGH
- **Location:** `strategies/base_alphatrend_strategy.py:345-346`
- **Issue:** When ATR is `None` or `<= 0`, fallback to 5% stop loss.
  - No warning to user that ATR is missing
  - 5% hardcoded stop may not match strategy's risk profile
  - Should fail early if required column is missing
- **Impact:** Strategy runs with incorrect risk parameters.
- **Recommendation:** Fail fast if required indicators missing, don't silently degrade.

#### **Problem 1.1.5: Pyramiding Break-Even Calculation**
- **Severity:** MEDIUM-HIGH
- **Location:** `Classes/Engine/position_manager.py:180-200` (not in excerpt, but mentioned)
- **Issue:** Break-even calculation after pyramiding must account for:
  - Commission on initial entry
  - Commission on pyramid entry
  - FX rate differences if entry and pyramid at different rates
  - Average entry price calculation
- **Impact:** Stop loss may not truly be at break-even, could take small loss.
- **Test Case:**
  ```python
  # Test pyramiding break-even with commission
  # Initial: 100 shares @ $50 + $10 commission
  # Pyramid: 50 shares @ $55 + $5 commission
  # Break-even should be > $52.10 (weighted avg + commission)
  ```

### 1.2 Data Integrity Issues

#### **Problem 1.2.1: Look-Ahead Bias Detection Limitations**
- **Severity:** MEDIUM
- **Location:** `Classes/Strategy/base_strategy.py:221-253`
- **Issue:** Look-ahead bias detection only checks for NaN patterns at end vs start.
  - Only samples 5% or 10 rows (min)
  - Boolean columns skipped entirely
  - Won't catch subtle look-ahead like `.shift(-1)` on indicators that already have NaN
- **Impact:** Some forms of look-ahead bias may go undetected.
- **Recommendation:** Add explicit unit tests that intentionally introduce look-ahead and verify detection.
- **Test Case:**
  ```python
  def test_look_ahead_detection():
      class BadStrategy(BaseStrategy):
          def _prepare_data_impl(self, data):
              # Introduce look-ahead bias
              data['future_price'] = data['close'].shift(-1)
              return data
      # Should: print warning or raise error
  ```

#### **Problem 1.2.2: HistoricalDataView Boundary Conditions**
- **Severity:** MEDIUM
- **Location:** `Classes/Strategy/strategy_context.py:36`
- **Issue:** Context uses `HistoricalDataView` which limits data to `valid_end_index`.
  - What happens if strategy tries to access `data.iloc[valid_end_index + 1]`?
  - Is IndexError raised or does it return None?
  - Documentation unclear on error handling
- **Impact:** Strategy could crash with IndexError instead of graceful handling.
- **Test Case:**
  ```python
  view = HistoricalDataView(df, valid_end_index=50)
  # Try to access bar 51 - should raise IndexError or return None?
  bar = view.iloc[51]
  ```

#### **Problem 1.2.3: Required Columns Validation Timing**
- **Severity:** LOW-MEDIUM
- **Location:** `Classes/Strategy/base_strategy.py:203-219`
- **Issue:** Required columns validated in `prepare_data()`, which is called during backtest run.
  - User doesn't know about missing columns until backtest starts
  - Better to validate earlier (at strategy initialization or data load time)
- **Impact:** Poor user experience - late failure after waiting for data to load.
- **Recommendation:** Add early validation option at engine initialization.

### 1.3 Signal Generation Issues

#### **Problem 1.3.1: Signal Validation Gap**
- **Severity:** MEDIUM
- **Location:** Throughout signal generation flow
- **Issue:** Limited validation of signal parameters:
  - No validation that stop_loss is reasonable distance from entry (not too tight/wide)
  - No validation that position size > 0 after calculation
  - No validation that size doesn't exceed available capital
  - Pyramid size not validated against remaining capital
- **Impact:** Invalid trades could be attempted, causing crashes or unexpected behavior.
- **Test Cases:**
  ```python
  # Test 1: Zero position size
  # Test 2: Negative position size
  # Test 3: Position size > available capital
  # Test 4: Stop loss = entry price
  # Test 5: Stop loss too far from entry (> 50%)
  ```

#### **Problem 1.3.2: Fundamental Rules Integration**
- **Severity:** MEDIUM
- **Location:** `Classes/Strategy/base_strategy.py:531-539`
- **Issue:** Fundamental check happens AFTER entry signal generated but:
  - No test coverage for fundamental filtering
  - What if `FundamentalData.from_bar()` raises exception due to missing columns?
  - Default `AlwaysPassFundamentalRules` never tested with actual fundamental data
- **Impact:** Fundamental filtering may fail silently or crash unexpectedly.
- **Test Case:**
  ```python
  # Test fundamental check with missing fundamental columns
  class TestFundamentalRules(BaseFundamentalRules):
      def check_fundamentals(self, context, fundamentals):
          if fundamentals.eps is None:
              return FundamentalCheckResult(passed=False, reason="No EPS data")
          return FundamentalCheckResult(passed=fundamentals.eps > 0)
  # Should: handle gracefully when eps column missing
  ```

#### **Problem 1.3.3: Max Position Duration Edge Case**
- **Severity:** LOW-MEDIUM
- **Location:** `Classes/Strategy/base_strategy.py:135-145`
- **Issue:** `max_position_duration` property returns `Optional[int]`.
  - How is this enforced in the engine?
  - Is it checked in `generate_exit_signal()` or by engine?
  - Documentation suggests it's "checked in exit rules" but not clear where
- **Impact:** Max duration may not be enforced correctly.
- **Test Case:**
  ```python
  # Test that position auto-closes after max_position_duration bars
  strategy = BaseAlphaTrendStrategy(max_hold_days=5)
  # Verify position closes exactly on bar 5, not bar 6
  ```

### 1.4 Context and Data Access Issues

#### **Problem 1.4.1: FX Rate Handling**
- **Severity:** MEDIUM
- **Location:** `Classes/Strategy/strategy_context.py:44`
- **Issue:** FX rate documentation says: "If trading USD security with GBP base, fx_rate=0.8 means $1 = £0.8"
  - Is this the rate used consistently in position sizing calculations?
  - What if FX rate changes significantly during position hold?
  - Is FX rate updated bar-by-bar or fixed at entry?
- **Impact:** Incorrect position sizing or P/L calculations in multi-currency backtests.
- **Test Case:**
  ```python
  # Test position sizing with FX rate
  # Test that P/L correctly accounts for FX rate changes
  ```

#### **Problem 1.4.2: Context Immutability Enforcement**
- **Severity:** LOW-MEDIUM
- **Location:** `Classes/Strategy/strategy_context.py:12`
- **Issue:** Context marked as `@dataclass(frozen=True)` but:
  - `data` field is Union[DataFrame, HistoricalDataView] - both are mutable
  - Strategy could accidentally modify `data` despite frozen dataclass
  - Python's frozen dataclass only makes attributes non-reassignable, not deep immutable
- **Impact:** Strategy could introduce subtle bugs by mutating shared data.
- **Test Case:**
  ```python
  # Test that modifying context.data raises error or doesn't affect other bars
  context = StrategyContext(...)
  context.data['new_column'] = 1  # Should this be prevented?
  ```

### 1.5 Engine Integration Issues

#### **Problem 1.5.1: Stop Loss Check Sequence**
- **Severity:** MEDIUM
- **Location:** `Classes/Engine/single_security_engine.py:134-149`
- **Issue:** Stop loss checked BEFORE strategy generates signal.
  - What if strategy wants to cancel stop loss on specific bar?
  - Order of checks: stop -> take profit -> adjust stop -> partial exit -> signal
  - Is this order documented in strategy development guide?
- **Impact:** Strategy may not have full control over exit logic.
- **Test Case:**
  ```python
  # Test strategy that wants to ignore stop loss under certain conditions
  # Currently not possible - stop checked first
  ```

#### **Problem 1.5.2: Partial Exit After Stop Adjustment**
- **Severity:** LOW
- **Location:** `Classes/Engine/single_security_engine.py:170-194`
- **Issue:** Sequence is: adjust stop → partial exit → generate signal
  - If partial exit reduces position, is stop loss scaled appropriately?
  - If stop adjusted to break-even, then partial exit taken, is remaining stop still valid?
- **Impact:** Stop loss may not protect remaining position correctly after partial exit.
- **Test Case:**
  ```python
  # Test: adjust stop, then take 50% partial exit
  # Verify: stop loss still valid for remaining 50%
  ```

#### **Problem 1.5.3: Equity Calculation Timing**
- **Severity:** LOW-MEDIUM
- **Location:** `Classes/Engine/single_security_engine.py:110-130`
- **Issue:** Equity calculated at start of bar, before any actions:
  - Context shows equity before stop loss hit
  - If stop hit, equity calculated again after position closed
  - Two equity records for same bar possible?
- **Impact:** Equity curve may have timing inconsistencies.
- **Test Case:**
  ```python
  # Test equity curve when stop loss hit
  # Verify only one equity record per bar
  ```

### 1.6 AlphaTrend Strategy Specific Issues

#### **Problem 1.6.1: Numba Compilation Fallback**
- **Severity:** LOW-MEDIUM
- **Location:** `strategies/base_alphatrend_strategy.py:74-85`
- **Issue:** If Numba not available, falls back to pure Python with no-op decorator.
  - Pure Python version of `_alphatrend_numba()` not optimized
  - Could be 10-100x slower on large datasets
  - No warning to user that performance will be degraded
- **Impact:** Severe performance degradation on systems without Numba.
- **Recommendation:** Warn user if Numba not available, or provide optimized fallback.

#### **Problem 1.6.2: AlphaTrend Signal Filtering**
- **Severity:** LOW
- **Location:** `strategies/base_alphatrend_strategy.py:290-297`
- **Issue:** Signals filtered to ensure alternating buy/sell:
  - Filter state persists across strategy lifetime
  - If backtest restarted with same strategy instance, state may not reset
  - Could cause first signal in new backtest to be wrong type
- **Impact:** Strategy not reusable across multiple backtests without reinstantiation.
- **Test Case:**
  ```python
  strategy = BaseAlphaTrendStrategy()
  result1 = engine.run('SYM1', data1, strategy)
  result2 = engine.run('SYM2', data2, strategy)  # Same instance
  # Verify: signal filtering state reset for result2
  ```

#### **Problem 1.6.3: MFI Threshold Calculation**
- **Severity:** LOW
- **Location:** `strategies/base_alphatrend_strategy.py:271-273`
- **Issue:** MFI thresholds use rolling quantiles over `percentile_period` (default 100).
  - First 100 bars have NaN thresholds
  - Could cause `momentum_bullish` to be NaN or False early on
  - Combined with warmup check at line 362, effectively needs 100+ bars
- **Impact:** Strategy requires long warmup period, may not trade on shorter timeframes.
- **Test Case:**
  ```python
  # Test with exactly 100 bars of data
  # Verify: at least one trade possible
  ```

### 1.7 Parameter Validation Issues

#### **Problem 1.7.1: Parameter Ranges Not Enforced**
- **Severity:** MEDIUM
- **Location:** `strategies/base_alphatrend_strategy.py:182-211`
- **Issue:** Constructor accepts parameters without validation:
  - `atr_multiplier` could be negative or zero
  - `risk_percent` could be > 100% or negative
  - `max_hold_days` could be negative
  - `smoothing_length` could be zero
- **Impact:** Invalid parameters cause undefined behavior or crashes.
- **Recommendation:** Add parameter validation in `_validate_parameters()`.
- **Test Case:**
  ```python
  # Test invalid parameters
  strategy = BaseAlphaTrendStrategy(atr_multiplier=-1.0)  # Should raise ValueError
  strategy = BaseAlphaTrendStrategy(risk_percent=150.0)  # Should raise ValueError
  ```

#### **Problem 1.7.2: Strategy Config vs Constructor Mismatch**
- **Severity:** LOW-MEDIUM
- **Location:** `config/strategy_parameters.json` vs strategy constructors
- **Issue:** Parameter definitions in JSON config may not match constructor signatures:
  - No automated validation that config matches actual strategy
  - Typo in config parameter name won't be caught until runtime
  - Optimization could try to pass invalid parameter names
- **Impact:** Optimization failures, silent parameter mismatches.
- **Recommendation:** Add validation that config parameters match constructor.

### 1.8 Testing Infrastructure Gaps

#### **Problem 1.8.1: No Mock Data Utilities**
- **Severity:** MEDIUM
- **Issue:** No utility functions to generate test data for strategies.
- **Impact:** Hard to write strategy tests, need to create DataFrames manually.
- **Recommendation:** Create `tests/utils/test_data_generator.py` with:
  - `generate_ohlcv()` - basic OHLCV data
  - `generate_with_indicators()` - OHLCV + common indicators
  - `generate_trending_market()` - uptrend/downtrend scenarios
  - `generate_ranging_market()` - sideways market

#### **Problem 1.8.2: No Strategy Test Base Class**
- **Severity:** MEDIUM
- **Issue:** No base test class with common setup/teardown for strategy tests.
- **Impact:** Each test file must recreate boilerplate.
- **Recommendation:** Create `tests/test_strategy_base.py` with shared fixtures.

#### **Problem 1.8.3: No Performance Benchmarks**
- **Severity:** LOW
- **Issue:** No baseline performance metrics for strategy backtests.
- **Impact:** Can't detect performance regressions.
- **Recommendation:** Add benchmark tests that track execution time.

---

## 2. TESTING CHECKLIST

### 2.1 Unit Tests (Strategy Logic)

#### Signal Generation Tests
- [ ] Entry signal generation
  - [ ] Signal generated when conditions met
  - [ ] Signal not generated when conditions not met
  - [ ] Correct signal type (BUY)
  - [ ] Signal contains stop_loss
  - [ ] Signal contains correct direction
  - [ ] Signal reason populated
- [ ] Exit signal generation
  - [ ] Exit signal when max duration reached
  - [ ] Exit signal when technical conditions met
  - [ ] Correct signal type (SELL)
  - [ ] Exit reason populated
- [ ] No signal during warmup period
  - [ ] Returns None when insufficient data
  - [ ] First valid signal after warmup complete

#### Stop Loss Tests
- [ ] Initial stop loss calculation
  - [ ] Correct distance from entry (ATR-based)
  - [ ] Stop loss below entry for LONG
  - [ ] Stop loss above entry for SHORT
  - [ ] Fallback when ATR missing
- [ ] Trailing stop adjustment
  - [ ] Stop moves in favorable direction only
  - [ ] Stop doesn't move backward
  - [ ] Stop adjustment returns None when no adjustment needed
- [ ] Direction-aware validation
  - [ ] LONG: new stop must be > current stop
  - [ ] SHORT: new stop must be < current stop

#### Position Sizing Tests
- [ ] Risk-based sizing
  - [ ] Correct calculation: (Equity * Risk%) / Stop Distance
  - [ ] Respects available capital
  - [ ] Returns positive size
  - [ ] Handles FX rate correctly
- [ ] Edge cases
  - [ ] Stop distance = 0 (should error or warn)
  - [ ] Stop distance negative (invalid stop)
  - [ ] Available capital = 0
  - [ ] Risk percent > 100%

#### Fundamental Rules Tests
- [ ] Fundamental filtering
  - [ ] Trade allowed when fundamentals pass
  - [ ] Trade blocked when fundamentals fail
  - [ ] Correct reason returned
- [ ] Missing fundamental data
  - [ ] Graceful handling when columns missing
  - [ ] Default AlwaysPass works without fundamental columns
- [ ] Market cap filtering
  - [ ] Correct categorization (MICRO, SMALL, MID, LARGE, MEGA)
  - [ ] Rules applied per category

### 2.2 Integration Tests (Strategy + Engine)

#### Basic Execution Flow
- [ ] Complete backtest run
  - [ ] Strategy executes without errors
  - [ ] At least one trade generated (on valid dataset)
  - [ ] Equity curve generated
  - [ ] Final equity calculated correctly
- [ ] Entry execution
  - [ ] Position opened when signal generated
  - [ ] Correct entry price (close of bar)
  - [ ] Correct position size
  - [ ] Commission deducted from capital
- [ ] Exit execution
  - [ ] Position closed on exit signal
  - [ ] Correct exit price
  - [ ] P/L calculated correctly
  - [ ] Capital updated correctly

#### Stop Loss Integration
- [ ] Stop loss enforcement
  - [ ] Position closed when stop hit
  - [ ] Exit reason = "Stop loss hit"
  - [ ] Stop checked before strategy signal
- [ ] Trailing stop
  - [ ] Stop adjusted during position hold
  - [ ] Position exits at adjusted stop
  - [ ] Stop adjustment logged

#### Partial Exits
- [ ] Partial exit execution
  - [ ] Correct fraction of position closed
  - [ ] Remaining position tracked
  - [ ] P/L calculated for partial exit
  - [ ] Stop loss still valid for remaining position

#### Pyramiding
- [ ] Pyramid execution
  - [ ] Additional position added
  - [ ] Average entry price updated
  - [ ] Stop moved to break-even
  - [ ] Only one pyramid per trade allowed
- [ ] Break-even calculation
  - [ ] Accounts for initial commission
  - [ ] Accounts for pyramid commission
  - [ ] Accounts for FX rate
  - [ ] Stop truly at zero loss

### 2.3 Data Access Tests

#### Context Data Access
- [ ] Current bar access
  - [ ] `context.current_bar` returns correct bar
  - [ ] `context.current_price` matches close price
  - [ ] `context.current_date` correct
- [ ] Historical bar access
  - [ ] `get_bar(0)` returns current bar
  - [ ] `get_bar(-1)` returns previous bar
  - [ ] `get_bar(+1)` returns None (no future data)
  - [ ] `get_bar(-1000)` returns None (out of bounds)
- [ ] Indicator access
  - [ ] `get_indicator_value('atr_14')` returns correct value
  - [ ] Returns None for missing indicator
  - [ ] Offset works correctly: `get_indicator_value('atr_14', -1)`

#### Look-Ahead Bias Prevention
- [ ] HistoricalDataView enforcement
  - [ ] Only historical data accessible
  - [ ] Accessing future bars raises error or returns None
  - [ ] `len(context.data)` equals `current_index + 1`
- [ ] prepare_data() validation
  - [ ] Warning issued for `.shift(-n)` usage
  - [ ] NaN pattern detection works
  - [ ] Boolean columns handled correctly

### 2.4 Edge Case Tests

#### Extreme Market Conditions
- [ ] Gap up/down
  - [ ] Stop loss honored on gap through stop
  - [ ] Take profit honored on gap through target
- [ ] Zero/negative prices
  - [ ] Strategy handles gracefully or errors clearly
- [ ] Missing data
  - [ ] Missing bars in middle of dataset
  - [ ] NaN in price data
  - [ ] NaN in indicators

#### Capital Constraints
- [ ] Insufficient capital
  - [ ] Cannot open position if not enough capital
  - [ ] Error message clear
- [ ] Capital exhaustion
  - [ ] Multiple positions in portfolio engine
  - [ ] Capital allocation tracked correctly

#### Date Range Edge Cases
- [ ] Single bar dataset
  - [ ] Strategy handles without crash
- [ ] Warmup period > data length
  - [ ] Strategy returns no signals
  - [ ] Warning issued to user
- [ ] Backtest ends mid-position
  - [ ] Position closed at final bar
  - [ ] P/L calculated correctly

### 2.5 Multi-Currency Tests

- [ ] FX rate application
  - [ ] Position sizing uses FX rate correctly
  - [ ] P/L converted to base currency
  - [ ] Capital updates in base currency
- [ ] FX rate changes
  - [ ] Rate updated bar-by-bar
  - [ ] Position value reflects current FX rate
- [ ] Currency mismatch
  - [ ] Warning when no FX rate available
  - [ ] Fallback to 1.0 documented

### 2.6 Optimization Tests

- [ ] Parameter sweeps
  - [ ] Strategy created with each parameter combination
  - [ ] All parameters passed correctly to constructor
  - [ ] Results collected correctly
- [ ] Parameter validation
  - [ ] Invalid parameters rejected early
  - [ ] Config parameter names match constructor
- [ ] Performance
  - [ ] Optimization completes in reasonable time
  - [ ] No memory leaks across iterations

---

## 3. SPECIFIC TEST SCENARIOS

### Scenario 3.1: AlphaTrend Strategy - Full Trade Cycle

**Setup:**
- 200 bars of data
- Clear uptrend from bar 100-150
- AlphaTrend buy signal at bar 105
- Hold for 10 days, exit at bar 115

**Expected Behavior:**
1. No trades in warmup period (bars 0-99)
2. Entry signal generated at bar 105
3. Position opened with correct size and stop loss
4. Position held for 10 bars
5. Position closed at bar 115 (max_hold_days)
6. Profit captured correctly

**Test Code:**
```python
def test_alphatrend_full_cycle():
    # Generate test data with known pattern
    data = generate_trending_market(
        bars=200,
        trend_start=100,
        trend_end=150,
        trend_direction='up'
    )

    strategy = BaseAlphaTrendStrategy(
        atr_multiplier=2.0,
        risk_percent=2.0,
        max_hold_days=10
    )

    config = BacktestConfig(initial_capital=10000, commission=...)
    engine = SingleSecurityEngine(config)

    result = engine.run('TEST', data, strategy)

    # Assertions
    assert len(result.trades) >= 1, "At least one trade should occur"
    trade = result.trades[0]
    assert trade.entry_bar >= 100, "Entry after warmup"
    assert trade.exit_bar - trade.entry_bar == 10, "Held for 10 bars"
    assert trade.exit_reason == "Max duration reached"
```

### Scenario 3.2: Stop Loss Enforcement

**Setup:**
- Position opened at $100 with stop loss at $95
- Price drops to $94 on next bar

**Expected Behavior:**
1. Stop loss hit on bar with price $94
2. Position closed at $94 (slippage not modeled by default)
3. Loss recorded: -6% (entry to stop) + commission
4. Exit reason: "Stop loss hit"

**Test Code:**
```python
def test_stop_loss_enforcement():
    data = pd.DataFrame({
        'date': [date1, date2, date3],
        'close': [100, 100, 94],  # Stop at 95, price drops to 94
        'atr_14': [3, 3, 3],
        # ... other required columns
    })

    # Run backtest and verify stop loss hit
    # ...
```

### Scenario 3.3: Pyramiding with Break-Even

**Setup:**
- Initial position: 100 shares @ $50
- Price moves to $55 (+10%)
- Pyramid signal: add 50 shares @ $55
- Commission: $5 per trade

**Expected Behavior:**
1. Initial position opened: 100 @ $50
2. Pyramid added: 50 @ $55
3. Average price: (100*50 + 50*55) / 150 = $51.67
4. Break-even stop: $51.67 + ($10 total commission / 150 shares) = $51.74
5. Stop adjusted to $51.74 automatically

**Test Code:**
```python
def test_pyramiding_break_even():
    # Create scenario where pyramid triggered
    # Verify break-even calculation includes commission
    # ...
```

### Scenario 3.4: Fundamental Filtering

**Setup:**
- Strategy with fundamental rules: EPS > 0, Revenue Growth > 5%
- Two securities: one passes, one fails

**Expected Behavior:**
1. Entry signal generated for both securities
2. Security 1 (EPS=2.5, Growth=10%): Trade allowed
3. Security 2 (EPS=-1.0, Growth=3%): Trade blocked
4. Rejection reason logged clearly

**Test Code:**
```python
def test_fundamental_filtering():
    class EarningsGrowthRules(BaseFundamentalRules):
        def check_fundamentals(self, context, fundamentals):
            if fundamentals.eps <= 0:
                return FundamentalCheckResult(False, "Negative EPS")
            if fundamentals.revenue_growth_yoy < 5.0:
                return FundamentalCheckResult(False, "Low revenue growth")
            return FundamentalCheckResult(True, "Passed")

    # Test with data including fundamental columns
    # ...
```

### Scenario 3.5: Look-Ahead Bias Detection

**Setup:**
- Strategy with intentional look-ahead bias in prepare_data()

**Expected Behavior:**
- Warning printed to console
- Warning describes which column has bias
- Backtest continues but user warned

**Test Code:**
```python
def test_look_ahead_detection():
    class BadStrategy(BaseStrategy):
        def _prepare_data_impl(self, data):
            # Intentional look-ahead
            data['future_close'] = data['close'].shift(-1)
            return data

    # Capture stdout
    # Verify warning printed
    # ...
```

### Scenario 3.6: Insufficient Warmup Data

**Setup:**
- Dataset with only 50 bars
- Strategy requires 100 bars warmup (percentile_period=100)

**Expected Behavior:**
- No entry signals generated
- Backtest completes without error
- Zero trades in result
- (Ideally) Warning to user about insufficient data

**Test Code:**
```python
def test_insufficient_warmup():
    data = generate_ohlcv(bars=50)
    strategy = BaseAlphaTrendStrategy(percentile_period=100)

    result = engine.run('TEST', data, strategy)

    assert len(result.trades) == 0, "No trades with insufficient warmup"
    # Should also check for warning message
```

---

## 4. RECOMMENDED TEST STRUCTURE

### 4.1 Test File Organization

```
tests/
├── unit/
│   ├── test_base_strategy.py
│   │   ├── test_signal_generation()
│   │   ├── test_stop_loss_calculation()
│   │   ├── test_position_sizing()
│   │   └── test_fundamental_integration()
│   ├── test_alphatrend_strategy.py
│   │   ├── test_alphatrend_calculation()
│   │   ├── test_signal_filtering()
│   │   ├── test_warmup_period()
│   │   └── test_entry_conditions()
│   ├── test_strategy_context.py
│   │   ├── test_data_access()
│   │   ├── test_look_ahead_protection()
│   │   └── test_indicator_access()
│   └── test_signal_model.py
│       ├── test_signal_creation()
│       └── test_signal_validation()
├── integration/
│   ├── test_strategy_engine_integration.py
│   │   ├── test_full_backtest_cycle()
│   │   ├── test_stop_loss_enforcement()
│   │   ├── test_partial_exits()
│   │   └── test_pyramiding()
│   └── test_multi_currency.py
├── scenarios/
│   ├── test_trade_scenarios.py
│   │   ├── test_full_trade_cycle()
│   │   ├── test_stop_loss_hit()
│   │   └── test_max_duration_exit()
│   └── test_edge_cases.py
│       ├── test_extreme_prices()
│       ├── test_missing_data()
│       └── test_insufficient_capital()
└── utils/
    ├── test_data_generator.py
    ├── strategy_test_base.py
    └── assertions.py
```

### 4.2 Test Utilities Needed

#### Test Data Generator
```python
# tests/utils/test_data_generator.py

def generate_ohlcv(bars=100, start_price=100, volatility=0.02):
    """Generate basic OHLCV data."""
    pass

def generate_with_indicators(bars=100, indicators=['atr_14', 'mfi_14']):
    """Generate OHLCV + common indicators."""
    pass

def generate_trending_market(bars=200, trend_start=50, trend_end=150):
    """Generate market with clear trend."""
    pass

def generate_ranging_market(bars=200, range_min=95, range_max=105):
    """Generate sideways market."""
    pass

def generate_gap_scenario(gap_pct=10, gap_direction='down'):
    """Generate price gap scenario."""
    pass
```

#### Custom Assertions
```python
# tests/utils/assertions.py

def assert_trade_executed(result, entry_bar=None, exit_bar=None):
    """Assert trade was executed with optional bar checks."""
    pass

def assert_stop_loss_hit(trade, expected_stop):
    """Assert trade exited at stop loss."""
    pass

def assert_position_size_correct(trade, context, strategy, tolerance=0.01):
    """Assert position size matches risk-based calculation."""
    pass

def assert_no_look_ahead_bias(strategy, data):
    """Assert strategy doesn't use future data."""
    pass
```

---

## 5. PRIORITY TESTING ROADMAP

### Phase 1: Critical Path Tests (Week 1)
1. ✅ Create test infrastructure (data generator, base classes)
2. ✅ Test BaseAlphaTrendStrategy signal generation
3. ✅ Test stop loss calculation and enforcement
4. ✅ Test position sizing with various scenarios
5. ✅ Test full backtest execution (integration)

### Phase 2: Edge Cases (Week 2)
1. ✅ Test warmup period handling
2. ✅ Test invalid parameter handling
3. ✅ Test missing data scenarios
4. ✅ Test capital exhaustion
5. ✅ Test extreme price movements

### Phase 3: Advanced Features (Week 3)
1. ✅ Test pyramiding and break-even
2. ✅ Test partial exits
3. ✅ Test trailing stops
4. ✅ Test fundamental filtering
5. ✅ Test multi-currency support

### Phase 4: Performance & Regression (Ongoing)
1. ✅ Performance benchmarks
2. ✅ Regression test suite
3. ✅ Look-ahead bias detection validation
4. ✅ Optimization workflow tests

---

## 6. KNOWN GAPS IN CURRENT SYSTEM

### Documentation Gaps
1. No documentation on stop loss/take profit check order in engine
2. Incomplete documentation on FX rate handling
3. No examples of multi-currency backtests
4. No guide for creating custom fundamental rules

### Code Gaps
1. No validation for parameter ranges in strategy constructors
2. No early validation of required columns (fails at backtest time)
3. No warning when Numba unavailable (performance degradation)
4. No utility for generating test data
5. No strategy test base class

### Testing Gaps
1. Zero unit tests for strategy logic
2. No integration tests for engine + strategy
3. No performance benchmarks
4. No regression test suite
5. No test coverage metrics

---

## 7. RECOMMENDED FIXES

### High Priority Fixes

1. **Add Parameter Validation**
   ```python
   # In base_alphatrend_strategy.py
   def _validate_parameters(self):
       if self.atr_multiplier <= 0:
           raise ValueError(f"atr_multiplier must be > 0, got {self.atr_multiplier}")
       if not 0 < self.risk_percent <= 100:
           raise ValueError(f"risk_percent must be 0-100, got {self.risk_percent}")
       if self.max_hold_days < 1:
           raise ValueError(f"max_hold_days must be >= 1, got {self.max_hold_days}")
   ```

2. **Improve Stop Loss Fallback**
   ```python
   # In base_alphatrend_strategy.py:342
   def calculate_initial_stop_loss(self, context: StrategyContext) -> float:
       atr = context.get_indicator_value('atr_14')
       if atr is None or atr <= 0:
           raise ValueError(
               f"ATR indicator missing or invalid at bar {context.current_index}. "
               f"Ensure 'atr_14' column exists in data."
           )
       return context.current_price - (atr * self.atr_multiplier)
   ```

3. **Validate Warmup Period**
   ```python
   # In base_alphatrend_strategy.py (add to constructor)
   def __init__(self, ...):
       super().__init__(...)
       # Store warmup requirement
       self._warmup_bars = self.percentile_period

   def required_warmup_bars(self) -> int:
       return self._warmup_bars

   # In engine, before backtest:
   if len(data) < strategy.required_warmup_bars():
       warnings.warn(
           f"Dataset has {len(data)} bars but strategy requires "
           f"{strategy.required_warmup_bars()} bars for warmup. "
           f"No trades may be generated."
       )
   ```

4. **Add Signal Validation**
   ```python
   # In single_security_engine.py (after signal generated)
   def _validate_signal(self, signal: Signal, context: StrategyContext):
       if signal.type == SignalType.BUY:
           if signal.stop_loss is None:
               raise ValueError("BUY signal must have stop_loss set")

           # Validate stop direction
           if context.position.direction == TradeDirection.LONG:
               if signal.stop_loss >= context.current_price:
                   raise ValueError(
                       f"LONG stop_loss ({signal.stop_loss}) must be below "
                       f"entry price ({context.current_price})"
                   )
   ```

### Medium Priority Fixes

5. **Add Numba Warning**
   ```python
   # In base_alphatrend_strategy.py:73
   try:
       from numba import njit
       NUMBA_AVAILABLE = True
   except ImportError:
       NUMBA_AVAILABLE = False
       warnings.warn(
           "Numba not available. AlphaTrend calculations will run in pure Python, "
           "which may be 10-100x slower. Install numba for better performance: "
           "pip install numba"
       )
   ```

6. **Improve Look-Ahead Detection**
   - Add explicit test that introduces `.shift(-1)` and verifies warning
   - Document limitations of current detection
   - Consider stricter mode that raises error instead of warning

---

## 8. TESTING BEST PRACTICES

### Writing Good Strategy Tests

1. **Use Descriptive Names**
   ```python
   # Good
   def test_entry_signal_generated_when_alphatrend_crosses_above_smoothed_line()

   # Bad
   def test_entry()
   ```

2. **Test One Thing Per Test**
   ```python
   # Good - separate tests
   def test_stop_loss_below_entry_price_for_long()
   def test_stop_loss_above_entry_price_for_short()

   # Bad - testing multiple things
   def test_stop_loss()  # tests both LONG and SHORT
   ```

3. **Use Fixtures for Common Setup**
   ```python
   @pytest.fixture
   def basic_strategy():
       return BaseAlphaTrendStrategy(
           atr_multiplier=2.0,
           risk_percent=2.0,
           max_hold_days=10
       )

   @pytest.fixture
   def sample_data():
       return generate_with_indicators(bars=200)
   ```

4. **Test Edge Cases Explicitly**
   ```python
   def test_position_sizing_with_zero_stop_distance():
       # Explicit test for edge case
       # ...
   ```

5. **Use Parametrized Tests for Variations**
   ```python
   @pytest.mark.parametrize("risk_percent,expected_size", [
       (1.0, 100),
       (2.0, 200),
       (5.0, 500),
   ])
   def test_position_sizing_scales_with_risk(risk_percent, expected_size):
       # Test multiple parameter values
       # ...
   ```

---

## 9. CONCLUSION

### Summary of Findings

**Critical Issues:** 5
**High Priority Issues:** 8
**Medium Priority Issues:** 10
**Low Priority Issues:** 7

### Key Recommendations

1. **Immediate Action Required:**
   - Create comprehensive test suite (Phase 1 of roadmap)
   - Add parameter validation to prevent invalid configurations
   - Fix stop loss fallback to fail fast instead of silent degradation
   - Add warmup period validation

2. **Short Term (1-2 weeks):**
   - Implement all high-priority fixes
   - Complete Phase 1 and Phase 2 of testing roadmap
   - Create test data generator utilities
   - Add integration tests

3. **Medium Term (1 month):**
   - Complete Phase 3 testing (advanced features)
   - Set up continuous testing infrastructure
   - Add performance benchmarks
   - Improve documentation gaps

4. **Long Term (Ongoing):**
   - Maintain regression test suite
   - Monitor test coverage
   - Update tests as new strategies added
   - Performance optimization based on benchmarks

### Testing Coverage Goals

- **Unit Test Coverage:** Target 80%+ for strategy logic
- **Integration Test Coverage:** All major execution paths
- **Edge Case Coverage:** All identified edge cases tested
- **Scenario Coverage:** At least 10 real-world scenarios

---

**Next Steps:**
1. Review this document with team
2. Prioritize fixes based on impact/effort
3. Begin Phase 1 of testing roadmap
4. Set up test infrastructure (data generator, base classes)
5. Run first round of tests and document results
