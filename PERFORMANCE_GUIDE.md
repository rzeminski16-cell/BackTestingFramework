# Optimization Performance Guide

## Speed Improvements Implemented

### ✅ Bug Fix
Fixed the `'BacktestConfig' object is not callable` error by removing the incorrect `config` parameter from `engine.run()` calls.

### ✅ Speed Optimizations Added

#### 1. **Parallel Processing (CPU Multi-core)** - Linux/Mac Only
Configure in GUI or config file:
```yaml
bayesian_optimization:
  n_jobs: 4  # Use 4 CPU cores (or -1 for all cores)
```

**Speed Improvement**: Up to 4x faster with 4 cores, varies by CPU
- Each core runs a separate backtest in parallel
- Memory usage increases proportionally
- Recommended: Start with 2 cores, increase if stable

**⚠️ Windows Limitation**:
- Parallel processing is **not available on Windows** due to platform-specific multiprocessing limitations
- Windows users will automatically use serial processing (n_jobs=1)
- This is expected behavior - you'll see: "Parallel processing not available on Windows (expected). Using serial processing."
- Use speed modes (Quick/Fast) instead for faster optimization on Windows

#### 2. **Speed Mode Selection**
Three modes available in GUI or config:

```yaml
bayesian_optimization:
  speed_mode: fast  # or "full" or "quick"
```

| Mode | Iterations | Speed | Accuracy | Best For |
|------|-----------|-------|----------|----------|
| **Full** | 100 | Baseline | Best | Final optimization |
| **Fast** | 50 | 2x faster | Good | Development & testing |
| **Quick** | 25 | 4x faster | Moderate | Rapid prototyping |

#### 3. **Skip Sensitivity Analysis**
Uncheck "Run Sensitivity Analysis" in GUI for 30-50% faster runs.
- Still get walk-forward results
- Skip robustness testing
- Good for initial exploration

## Performance Expectations

### Typical Optimization Times (Single Security, Full Mode, 1 Core)

| Data Size | Windows | Total Time |
|-----------|---------|------------|
| 3 years | ~5 windows | 15-30 min |
| 5 years | ~8 windows | 25-45 min |
| 10 years | ~15 windows | 45-90 min |

### With Speed Optimizations

| Configuration | Speed Improvement | Example Time (5 years) |
|---------------|-------------------|------------------------|
| Fast Mode + 1 Core | 2x | 12-22 min |
| Full Mode + 4 Cores | 3-4x | 6-12 min |
| Fast Mode + 4 Cores | 6-8x | 3-6 min |
| Quick Mode + 4 Cores + No Sensitivity | 15-20x | 1-3 min |

## Why No GPU Acceleration?

### Technical Reasons:

1. **Sequential Dependencies**
   - Backtesting is inherently sequential (bar-by-bar)
   - Each bar depends on previous position state
   - Cannot parallelize across time dimension

2. **Branching Logic**
   - Heavy use of if/else conditions
   - State management (positions, stops, etc.)
   - GPUs excel at matrix operations, not branching

3. **Data Transfer Overhead**
   - Moving data to/from GPU takes time
   - Overhead exceeds benefits for this workload

4. **Small Data Sizes**
   - Typical backtest: 1000-5000 bars
   - Too small to benefit from GPU parallelism
   - GPU overhead dominates for small datasets

### What GPUs Are Good For:
- ❌ Sequential backtesting (what we're doing)
- ❌ State-dependent algorithms
- ✅ Neural network training (millions of parallel matrix operations)
- ✅ Image processing (millions of pixels processed independently)
- ✅ Monte Carlo simulations (millions of independent runs)

## Optimization Tips

### For Maximum Speed:

1. **Use Quick Mode for Initial Testing**
   ```yaml
   speed_mode: quick
   n_jobs: 4
   ```
   - Get results in 1-5 minutes
   - Iterate on strategy quickly
   - Switch to Full Mode for final optimization

2. **Skip Sensitivity Analysis During Development**
   - Uncheck in GUI
   - Only run for final results
   - Saves 30-50% time

3. **Optimize One Security at a Time**
   - Run multiple GUI instances for different securities
   - Each instance can use different CPU cores
   - Aggregate results manually

4. **Use More CPU Cores**
   ```yaml
   n_jobs: -1  # Use all available cores
   ```
   - Best on machines with 4+ cores
   - Monitor memory usage (each core uses ~500MB-1GB)

### For Maximum Accuracy:

1. **Use Full Mode**
   ```yaml
   speed_mode: full
   n_iterations: 100
   ```

2. **Always Run Sensitivity Analysis**
   - Essential for detecting overfitting
   - Worth the extra time

3. **Use More Windows (Longer Data)**
   - More validation windows = more robust
   - Requires longer time period in data

## Recommended Configurations

### Development (Quick Iteration)
```yaml
bayesian_optimization:
  speed_mode: quick
  n_iterations: 25  # (automatically set by quick mode)
  n_jobs: 4

sensitivity_analysis:
  test_mode: individual  # Faster than combinations
```
**GUI Settings:**
- Speed Mode: Quick (25 iter)
- CPU Cores: 4
- Sensitivity Analysis: Unchecked

**Time**: 1-5 minutes per security

---

### Testing (Good Balance)
```yaml
bayesian_optimization:
  speed_mode: fast
  n_iterations: 50  # (automatically set by fast mode)
  n_jobs: 2

sensitivity_analysis:
  test_mode: combinations
  max_sensitivity_tests: 500
```
**GUI Settings:**
- Speed Mode: Fast (50 iter)
- CPU Cores: 2
- Sensitivity Analysis: Checked

**Time**: 5-15 minutes per security

---

### Production (Maximum Accuracy)
```yaml
bayesian_optimization:
  speed_mode: full
  n_iterations: 100
  n_jobs: 1  # More stable, less memory

sensitivity_analysis:
  test_mode: combinations
  max_sensitivity_tests: 1000
```
**GUI Settings:**
- Speed Mode: Full (100 iter)
- CPU Cores: 1
- Sensitivity Analysis: Checked

**Time**: 20-60 minutes per security

## Bottlenecks & Solutions

### Issue: "It's still too slow"

**Solutions:**
1. ✅ Use Quick mode (4x faster)
2. ✅ Use 4+ CPU cores (4x faster)
3. ✅ Skip sensitivity analysis (1.5x faster)
4. ✅ Reduce parameter search space (edit config)
5. ✅ Use shorter data periods (fewer windows)

**Combined**: 20-30x speed improvement possible

### Issue: "Using 4 cores but still slow"

**Check:**
- Is your CPU actually 4+ cores? (Run `python -c "import os; print(os.cpu_count())"`)
- Memory usage high? Reduce n_jobs
- Other processes using CPU? Close them

### Issue: "Running out of memory"

**Solutions:**
- Reduce n_jobs (e.g., from 4 to 2)
- Use individual sensitivity mode instead of combinations
- Close other applications
- Optimize one security at a time

## Example Workflow

### Phase 1: Rapid Development
**Linux/Mac (5 minutes):**
```bash
python optimize_gui.py
```
- Select strategy: AlphaTrendStrategy
- Select security: AAPL
- Speed Mode: Quick (25 iter)
- CPU Cores: 4
- Sensitivity Analysis: Unchecked
- Click "Start Optimization"

**Windows (10-15 minutes):**
Same as above but:
- CPU Cores: 1 (only option available)
- Will automatically use serial processing

**Result**: Quick feedback on whether strategy works at all

---

### Phase 2: Validation
**Linux/Mac (15 minutes):**
- Speed Mode: Fast (50 iter)
- CPU Cores: 2
- Sensitivity Analysis: Checked

**Windows (25-30 minutes):**
- Speed Mode: Fast (50 iter)
- CPU Cores: 1
- Sensitivity Analysis: Checked

**Result**: More confident parameters with robustness check

---

### Phase 3: Production
**All Platforms (30-50 minutes):**
Before going live:
- Speed Mode: Full (100 iter)
- CPU Cores: 1 (most stable, required on Windows)
- Sensitivity Analysis: Checked
- Multiple securities

**Result**: Production-ready parameters with full validation

## Monitoring Performance

### During Optimization:

Check the GUI progress output:
```
Speed Mode: FAST
CPU Cores: 4
Optimizing window 1/8: Iteration 50/50 (100%)
```

### After Optimization:

Check the Excel report "Summary" sheet:
- Total Windows: Should be 5-15 for good validation
- Success Rate: Should be >75%
- Degradation: Should be <15%

## Future Optimizations (Not Yet Implemented)

Potential future improvements:
1. ⏱️ Vectorized backtesting (10-100x faster, but less flexible)
2. ⏱️ Caching of repeated backtests
3. ⏱️ Distributed computing across multiple machines
4. ⏱️ Incremental optimization (update instead of full rerun)

## Conclusion

While GPU acceleration isn't suitable for this workload, the implemented optimizations provide:
- ✅ **4x faster** with multi-core processing
- ✅ **2-4x faster** with speed modes
- ✅ **1.5x faster** by skipping sensitivity
- ✅ **Combined: 12-24x faster** for quick iterations

This is sufficient for practical use while maintaining accuracy.
