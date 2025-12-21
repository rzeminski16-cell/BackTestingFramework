# Initial Base creation
## Goals
- Create a small balanced portfolio of 15 large cap stocks to run initial optimisation on and find base values for the strategies parameters. 
- Using these parameters create a vulnerability scoring algorithm that improves on the FILO approach to the capital contingency problem. 
- Lastly, run portfolio back test ready for deeper analysis

## Outcomes
**20/12/2025 - Optimisation**
Optimisation Report: [Optimisation Report 15 Large Cap]("C:\Users\rzemi\OneDrive\Desktop\Back Testing Framework\logs\optimization_reports\optimization_AlphaTrendStrategy_PORTFOLIO(AMZN,CAT,GRMN,HOOD,JPM,LLY,MCK,NEE,NFLX,NVDA,PYPL,SPGI,TMUS,WDAY,XOM)_20251220_151241.xlsx")

The optimisation was not very successful as the out of sample ratios were not recorded correctly and the number of windows used was too low to make the process conclusive.

The only reliable parameter was that of the ATR stop loss at 1.5 which was shown to be very stable.
The below parameters were selected for running individual stock back tests that will be later used for vulnerability algorithm adjustment.
- volume_short_ma: 5
- volume_long_ma: 21
- volume_alignment_window: 11
- stop_loss_percent: 0.0
- atr_stop_loss_multiple: 1.5
- grace_period_bars: 24
- momentum_gain_pct: 4.5
- momentum_lookback: 7
- risk_percent: 5.0

**20/12/2025 - Individual Back tests**
Folder: [Individual Back Testing Folder](C:\Users\rzemi\OneDrive\Desktop\Back Testing Framework\logs\AlphaTrendStrategy_Optimised)

The above folder contains individual back tests for 33 Large cap stocks, sampled from each asset category. The result of each back tests can be mostly ignored for now as it is more important that the logs and suitable for vulnerability score analysis.

**21/12/2025 - Vulnerability Score Alignment**

First settings for the core parameters are
- Immunity Days: 12 (half of the grace period)
- Base Score: 100
- Swap Threshold: 50

The goal of the vulnerability score algorithm is to help exit stagnant or loosing trades early to free up capital for a potentially more profitable trade. Meanwhile, allowing profitable trades to run their course.

Considering these targets only 3 features of the algorithm will be enabled 'Days Held', 'P/L Momentum (7D)' and 'P/L Momentum (14D)'.

### Iteration 1
Report: [Trade Log]("C:\Users\rzemi\OneDrive\Desktop\Back Testing Framework\reports\vulnerability\Base_Set_Up\trade_analysis_Defaults.xlsx")
With the default settings without decay/ stagnation logic, the first thing I notice is that the cumulative P/L is consistently lower with vulnerability score and the extra large wins are closed too early resulting in a huge P/L gap.![[Pasted image 20251221133505.png]]

This suggests 27% of trades were hurt and as a result a cumulative loss of £880k, where £450k of those losses where the result of only the top 5 (3%) biggest losses. 
In conclusion the most important problem to tackle first is to avoid winners being not able to run their full way.

Iteration 2
Report: 
