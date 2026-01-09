"""
Pattern Analysis Module for Per-Trade Analysis

Handles signal strength scoring, pattern flagging, and aggregate pattern
identification across winning and losing trades.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime


# =============================================================================
# SIGNAL STRENGTH SCORING
# =============================================================================

@dataclass
class SignalStrengthScore:
    """Entry signal strength score with breakdown."""
    total_score: float
    max_possible: float = 100.0
    breakdown: Dict[str, Dict] = field(default_factory=dict)
    tier: str = ""
    description: str = ""

    def __post_init__(self):
        self.tier = self._calculate_tier()
        self.description = self._generate_description()

    def _calculate_tier(self) -> str:
        """Determine tier based on score."""
        if self.total_score >= 80:
            return "excellent"
        elif self.total_score >= 60:
            return "good"
        elif self.total_score >= 40:
            return "moderate"
        else:
            return "weak"

    def _generate_description(self) -> str:
        """Generate human-readable description."""
        tier_descriptions = {
            "excellent": "Excellent confluence - very strong setup",
            "good": "Good confluence - strong setup",
            "moderate": "Moderate confluence - reasonable setup",
            "weak": "Weak confluence - questionable setup"
        }
        return tier_descriptions.get(self.tier, "Unknown")


class SignalStrengthCalculator:
    """Calculate entry signal strength based on technical confluence."""

    # Scoring weights for each factor
    SCORING_FACTORS = {
        'trend_aligned': {
            'max_points': 25,
            'description': 'Trend aligned (SMA_20 > SMA_50 > SMA_200 for LONG)'
        },
        'bollinger_position': {
            'max_points': 12,
            'description': 'Price within middle third of Bollinger Bands'
        },
        'volume_confirmation': {
            'max_points': 15,
            'description': 'Entry bar volume > 110% of 20-day average'
        },
        'macd_aligned': {
            'max_points': 15,
            'description': 'MACD > Signal line for LONG (or reverse for SHORT)'
        },
        'ma_distance': {
            'max_points': 10,
            'description': 'Price within 2% of SMA_50'
        },
        'rsi_neutral': {
            'max_points': 10,
            'description': 'RSI between 40-60 (neutral, not extreme)'
        },
        'atr_normal': {
            'max_points': 8,
            'description': 'ATR within 0.8-1.2x of 60-day average'
        },
        'price_momentum': {
            'max_points': 5,
            'description': 'Close > Open on entry day for LONG'
        }
    }

    def calculate_signal_strength(self,
                                  entry_row: pd.Series,
                                  price_history: pd.DataFrame,
                                  side: str = 'LONG') -> SignalStrengthScore:
        """
        Calculate signal strength score for an entry.

        Args:
            entry_row: Series with entry day data including indicators
            price_history: Historical price data for context
            side: Trade direction ('LONG' or 'SHORT')

        Returns:
            SignalStrengthScore with total and breakdown
        """
        breakdown = {}
        total = 0.0

        is_long = side.upper() == 'LONG'

        # 1. Trend Alignment (25 points)
        points, detail = self._check_trend_alignment(entry_row, is_long)
        breakdown['trend_aligned'] = {
            'points': points,
            'max': 25,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 2. Bollinger Band Position (12 points)
        points, detail = self._check_bollinger_position(entry_row)
        breakdown['bollinger_position'] = {
            'points': points,
            'max': 12,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 3. Volume Confirmation (15 points)
        points, detail = self._check_volume_confirmation(entry_row, price_history)
        breakdown['volume_confirmation'] = {
            'points': points,
            'max': 15,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 4. MACD Alignment (15 points)
        points, detail = self._check_macd_alignment(entry_row, is_long)
        breakdown['macd_aligned'] = {
            'points': points,
            'max': 15,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 5. Moving Average Distance (10 points)
        points, detail = self._check_ma_distance(entry_row)
        breakdown['ma_distance'] = {
            'points': points,
            'max': 10,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 6. RSI Neutral Zone (10 points)
        points, detail = self._check_rsi_neutral(entry_row)
        breakdown['rsi_neutral'] = {
            'points': points,
            'max': 10,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 7. ATR Volatility Normal (8 points)
        points, detail = self._check_atr_normal(entry_row, price_history)
        breakdown['atr_normal'] = {
            'points': points,
            'max': 8,
            'met': points > 0,
            'detail': detail
        }
        total += points

        # 8. Price Momentum (5 points)
        points, detail = self._check_price_momentum(entry_row, is_long)
        breakdown['price_momentum'] = {
            'points': points,
            'max': 5,
            'met': points > 0,
            'detail': detail
        }
        total += points

        return SignalStrengthScore(
            total_score=total,
            breakdown=breakdown
        )

    def _get_indicator_value(self, row: pd.Series, names: List[str]) -> Optional[float]:
        """Get indicator value trying multiple column name variations."""
        for name in names:
            if name in row.index and pd.notna(row[name]):
                return float(row[name])
        return None

    def _check_trend_alignment(self, row: pd.Series, is_long: bool) -> Tuple[float, str]:
        """Check if SMAs are properly aligned."""
        sma_20 = self._get_indicator_value(row, ['sma_20_sma', 'sma_20'])
        sma_50 = self._get_indicator_value(row, ['sma_50_sma', 'sma_50'])
        sma_200 = self._get_indicator_value(row, ['sma_200_sma', 'sma_200'])

        if sma_20 is None or sma_50 is None or sma_200 is None:
            return 0, "SMAs not available"

        if is_long:
            if sma_20 > sma_50 > sma_200:
                return 25, f"Bullish alignment: SMA20 ({sma_20:.2f}) > SMA50 ({sma_50:.2f}) > SMA200 ({sma_200:.2f})"
            elif sma_20 > sma_50:
                return 12, f"Partial bullish: SMA20 > SMA50, but SMA50 < SMA200"
        else:
            if sma_20 < sma_50 < sma_200:
                return 25, f"Bearish alignment: SMA20 < SMA50 < SMA200"
            elif sma_20 < sma_50:
                return 12, f"Partial bearish: SMA20 < SMA50"

        return 0, "SMAs not aligned with trade direction"

    def _check_bollinger_position(self, row: pd.Series) -> Tuple[float, str]:
        """Check if price is in middle third of Bollinger Bands."""
        bb_upper = self._get_indicator_value(row, ['bbands_20_real upper band', 'bb_upper'])
        bb_lower = self._get_indicator_value(row, ['bbands_20_real lower band', 'bb_lower'])
        close = row.get('close')

        if bb_upper is None or bb_lower is None or close is None:
            return 0, "Bollinger Bands not available"

        bb_range = bb_upper - bb_lower
        if bb_range <= 0:
            return 0, "Invalid Bollinger Band range"

        # Calculate position (0 = lower band, 1 = upper band)
        position = (close - bb_lower) / bb_range

        if 0.33 <= position <= 0.67:
            return 12, f"Price in middle third ({position:.1%} of BB range)"
        elif 0.2 <= position <= 0.8:
            return 6, f"Price in outer-middle area ({position:.1%} of BB range)"
        else:
            return 0, f"Price at extreme ({position:.1%} of BB range)"

    def _check_volume_confirmation(self, row: pd.Series, history: pd.DataFrame) -> Tuple[float, str]:
        """Check if volume is above 20-day average."""
        volume = row.get('volume')

        if volume is None or history is None or len(history) < 20:
            return 0, "Volume data not available"

        if 'volume' not in history.columns:
            return 0, "Volume history not available"

        avg_volume = history['volume'].tail(20).mean()

        if avg_volume <= 0:
            return 0, "Invalid average volume"

        volume_ratio = volume / avg_volume

        if volume_ratio >= 2.0:
            return 15, f"Volume spike: {volume_ratio:.1f}x average"
        elif volume_ratio >= 1.1:
            return 15, f"Volume confirmed: {volume_ratio:.1f}x average"
        elif volume_ratio >= 0.8:
            return 7, f"Volume normal: {volume_ratio:.1f}x average"
        else:
            return 0, f"Low volume: {volume_ratio:.1f}x average"

    def _check_macd_alignment(self, row: pd.Series, is_long: bool) -> Tuple[float, str]:
        """Check MACD alignment with trade direction."""
        macd = self._get_indicator_value(row, ['macd_14_macd', 'macd'])
        signal = self._get_indicator_value(row, ['macd_14_macd_signal', 'macd_signal'])

        if macd is None or signal is None:
            return 0, "MACD not available"

        if is_long:
            if macd > signal and macd > 0:
                return 15, f"MACD bullish crossover above zero (MACD: {macd:.4f})"
            elif macd > signal:
                return 10, f"MACD above signal but below zero"
            else:
                return 0, f"MACD bearish (below signal)"
        else:
            if macd < signal and macd < 0:
                return 15, f"MACD bearish crossover below zero"
            elif macd < signal:
                return 10, f"MACD below signal but above zero"
            else:
                return 0, f"MACD bullish (above signal)"

    def _check_ma_distance(self, row: pd.Series) -> Tuple[float, str]:
        """Check distance from SMA_50."""
        sma_50 = self._get_indicator_value(row, ['sma_50_sma', 'sma_50'])
        close = row.get('close')

        if sma_50 is None or close is None:
            return 0, "SMA_50 not available"

        distance_pct = abs((close - sma_50) / sma_50) * 100

        if distance_pct <= 2:
            return 10, f"Price within 2% of SMA_50 ({distance_pct:.1f}%)"
        elif distance_pct <= 5:
            return 5, f"Price within 5% of SMA_50 ({distance_pct:.1f}%)"
        else:
            return 0, f"Price far from SMA_50 ({distance_pct:.1f}%)"

    def _check_rsi_neutral(self, row: pd.Series) -> Tuple[float, str]:
        """Check if RSI is in neutral zone."""
        rsi = self._get_indicator_value(row, ['rsi_14_rsi', 'rsi'])

        if rsi is None:
            return 0, "RSI not available"

        if 40 <= rsi <= 60:
            return 10, f"RSI neutral ({rsi:.1f})"
        elif 30 <= rsi <= 70:
            return 5, f"RSI moderate ({rsi:.1f})"
        elif rsi < 30:
            return 0, f"RSI oversold ({rsi:.1f})"
        else:
            return 0, f"RSI overbought ({rsi:.1f})"

    def _check_atr_normal(self, row: pd.Series, history: pd.DataFrame) -> Tuple[float, str]:
        """Check if ATR is within normal range."""
        atr = self._get_indicator_value(row, ['atr_14_atr', 'atr'])

        if atr is None or history is None or len(history) < 60:
            return 0, "ATR data not available"

        atr_col = 'atr_14_atr' if 'atr_14_atr' in history.columns else 'atr'
        if atr_col not in history.columns:
            return 0, "ATR history not available"

        avg_atr = history[atr_col].tail(60).mean()

        if avg_atr <= 0:
            return 0, "Invalid average ATR"

        atr_ratio = atr / avg_atr

        if 0.8 <= atr_ratio <= 1.2:
            return 8, f"ATR normal ({atr_ratio:.2f}x average)"
        elif 0.6 <= atr_ratio <= 1.5:
            return 4, f"ATR slightly elevated ({atr_ratio:.2f}x average)"
        else:
            return 0, f"ATR abnormal ({atr_ratio:.2f}x average)"

    def _check_price_momentum(self, row: pd.Series, is_long: bool) -> Tuple[float, str]:
        """Check intraday price momentum."""
        open_price = row.get('open')
        close = row.get('close')

        if open_price is None or close is None:
            return 0, "OHLC data not available"

        if is_long:
            if close > open_price:
                pct = ((close - open_price) / open_price) * 100
                return 5, f"Bullish bar (close > open, +{pct:.2f}%)"
            else:
                return 0, "Bearish bar on entry"
        else:
            if close < open_price:
                pct = ((open_price - close) / open_price) * 100
                return 5, f"Bearish bar (close < open, -{pct:.2f}%)"
            else:
                return 0, "Bullish bar on entry"


# =============================================================================
# PATTERN FLAGGING
# =============================================================================

@dataclass
class PatternFlag:
    """A pattern flag for trade analysis."""
    flag_id: str
    category: str  # 'insider', 'technical', 'market_regime', 'correlation', 'options'
    severity: str  # 'info', 'warning', 'alert'
    description: str
    interpretation: str
    value: Optional[Any] = None


class PatternFlagger:
    """Identify and flag patterns in trade data."""

    def __init__(self, thresholds: Optional[Dict] = None):
        """Initialize with custom thresholds if provided."""
        self.thresholds = thresholds or {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_spike_threshold': 1.5,
            'volume_spike_threshold': 2.0,
            'insider_min_value': 50000,
            'iv_high_percentile': 75,
            'iv_low_percentile': 25,
            'correlation_strong': 0.85,
            'correlation_weak': 0.30
        }

    def flag_all_patterns(self,
                          entry_row: pd.Series,
                          price_history: pd.DataFrame,
                          insider_flags: List[str],
                          options_data: Optional[Dict],
                          correlation_data: Optional[Dict],
                          market_regime: Optional[Any],
                          side: str = 'LONG') -> List[PatternFlag]:
        """
        Generate all pattern flags for a trade.

        Args:
            entry_row: Entry day data
            price_history: Historical prices
            insider_flags: Pre-computed insider flags
            options_data: Options analysis data
            correlation_data: Correlation analysis
            market_regime: Market regime data
            side: Trade direction

        Returns:
            List of PatternFlag objects
        """
        flags = []

        # Technical entry patterns
        flags.extend(self._flag_technical_patterns(entry_row, price_history, side))

        # Insider patterns (from pre-computed flags)
        flags.extend(self._convert_insider_flags(insider_flags))

        # Market regime patterns
        if market_regime:
            flags.extend(self._flag_market_regime_patterns(market_regime, side))

        # Options patterns
        if options_data:
            flags.extend(self._flag_options_patterns(options_data))

        # Correlation patterns
        if correlation_data:
            flags.extend(self._flag_correlation_patterns(correlation_data, side))

        return flags

    def _flag_technical_patterns(self,
                                  entry_row: pd.Series,
                                  history: pd.DataFrame,
                                  side: str) -> List[PatternFlag]:
        """Flag technical entry patterns."""
        flags = []

        # RSI Extreme
        rsi = entry_row.get('rsi_14_rsi') or entry_row.get('rsi')
        if pd.notna(rsi):
            if rsi > self.thresholds['rsi_overbought']:
                flags.append(PatternFlag(
                    flag_id='FLAG_RSI_EXTREME',
                    category='technical',
                    severity='warning',
                    description=f'RSI overbought at entry ({rsi:.1f})',
                    interpretation='Possible mean reversion entry - risky if expecting trend continuation',
                    value=rsi
                ))
            elif rsi < self.thresholds['rsi_oversold']:
                flags.append(PatternFlag(
                    flag_id='FLAG_RSI_EXTREME',
                    category='technical',
                    severity='warning',
                    description=f'RSI oversold at entry ({rsi:.1f})',
                    interpretation='Possible mean reversion entry - risky if expecting trend continuation',
                    value=rsi
                ))

        # ATR Spike
        atr = entry_row.get('atr_14_atr') or entry_row.get('atr')
        if pd.notna(atr) and history is not None and len(history) >= 60:
            atr_col = 'atr_14_atr' if 'atr_14_atr' in history.columns else 'atr'
            if atr_col in history.columns:
                avg_atr = history[atr_col].tail(60).mean()
                if avg_atr > 0 and atr / avg_atr > self.thresholds['atr_spike_threshold']:
                    flags.append(PatternFlag(
                        flag_id='FLAG_ATR_SPIKE',
                        category='technical',
                        severity='warning',
                        description=f'ATR spike at entry ({atr/avg_atr:.2f}x average)',
                        interpretation='High volatility environment - stop losses may need to be wider',
                        value=atr / avg_atr
                    ))

        # Volume Spike
        volume = entry_row.get('volume')
        if pd.notna(volume) and history is not None and len(history) >= 20:
            if 'volume' in history.columns:
                avg_volume = history['volume'].tail(20).mean()
                if avg_volume > 0 and volume / avg_volume > self.thresholds['volume_spike_threshold']:
                    flags.append(PatternFlag(
                        flag_id='FLAG_VOLUME_SPIKE',
                        category='technical',
                        severity='info',
                        description=f'Volume spike at entry ({volume/avg_volume:.1f}x average)',
                        interpretation='Potential exhaustion or capitulation - watch for reversals',
                        value=volume / avg_volume
                    ))

        # Counter-trend check
        sma_200 = entry_row.get('sma_200_sma') or entry_row.get('sma_200')
        close = entry_row.get('close')
        if pd.notna(sma_200) and pd.notna(close):
            is_long = side.upper() == 'LONG'
            if (is_long and close < sma_200) or (not is_long and close > sma_200):
                flags.append(PatternFlag(
                    flag_id='FLAG_COUNTER_TREND',
                    category='market_regime',
                    severity='warning',
                    description=f'Entry against 200-SMA direction',
                    interpretation='Mean reversion play - statistically more likely to fail',
                    value=None
                ))

        return flags

    def _convert_insider_flags(self, insider_flags: List[str]) -> List[PatternFlag]:
        """Convert string insider flags to PatternFlag objects."""
        flags = []

        flag_interpretations = {
            'FLAG_INSIDER_BUYING_PREENTRY': 'Potential bullish insider knowledge',
            'FLAG_INSIDER_SELLING_DURING_TRADE': 'May indicate insider knows of headwinds',
            'FLAG_INSIDER_BUYING_POSTEXIT': 'You exited before insiders were buying - validate timing',
            'FLAG_COORDINATED_INSIDER_ACTIVITY': 'Suggests coordinated decision - stronger signal'
        }

        for flag_str in insider_flags:
            flag_id = flag_str.split(':')[0] if ':' in flag_str else flag_str
            description = flag_str.split(':')[1].strip() if ':' in flag_str else flag_str

            flags.append(PatternFlag(
                flag_id=flag_id,
                category='insider',
                severity='alert' if 'BUYING' in flag_id else 'warning',
                description=description,
                interpretation=flag_interpretations.get(flag_id, 'Insider activity detected')
            ))

        return flags

    def _flag_market_regime_patterns(self, regime: Any, side: str) -> List[PatternFlag]:
        """Flag market regime patterns."""
        flags = []

        if hasattr(regime, 'volatility'):
            if regime.volatility == 'high':
                flags.append(PatternFlag(
                    flag_id='FLAG_HIGH_VOLATILITY_ENTRY',
                    category='market_regime',
                    severity='warning',
                    description='High volatility regime at entry',
                    interpretation='Whipsaws likely - stops may be hit easily',
                    value=regime.volatility
                ))
            elif regime.volatility == 'low':
                flags.append(PatternFlag(
                    flag_id='FLAG_LOW_VOLATILITY_ENTRY',
                    category='market_regime',
                    severity='info',
                    description='Low volatility regime at entry',
                    interpretation='Market sleepy - moves may be slow or choppy',
                    value=regime.volatility
                ))

        return flags

    def _flag_options_patterns(self, options_data: Dict) -> List[PatternFlag]:
        """Flag options market patterns."""
        flags = []

        iv_percentile = options_data.get('iv_percentile_entry')
        if iv_percentile is not None:
            if iv_percentile >= self.thresholds['iv_high_percentile']:
                flags.append(PatternFlag(
                    flag_id='FLAG_HIGH_IV_ENTRY',
                    category='options',
                    severity='warning',
                    description=f'IV in top quartile at entry ({iv_percentile:.0f}th percentile)',
                    interpretation='High implied volatility priced in - reversals likely',
                    value=iv_percentile
                ))
            elif iv_percentile <= self.thresholds['iv_low_percentile']:
                flags.append(PatternFlag(
                    flag_id='FLAG_LOW_IV_ENTRY',
                    category='options',
                    severity='info',
                    description=f'IV in bottom quartile at entry ({iv_percentile:.0f}th percentile)',
                    interpretation='Low volatility priced in - moves may be muted',
                    value=iv_percentile
                ))

        if options_data.get('earnings_during_trade'):
            flags.append(PatternFlag(
                flag_id='FLAG_EARNINGS_DURING_TRADE',
                category='options',
                severity='alert',
                description='Earnings announcement during trade window',
                interpretation='Catalyst risk - volatility likely to spike',
                value=True
            ))

        return flags

    def _flag_correlation_patterns(self, correlation_data: Dict, side: str) -> List[PatternFlag]:
        """Flag correlation patterns."""
        flags = []

        sector_corr = correlation_data.get('sector_correlation')
        if sector_corr is not None:
            if abs(sector_corr) > self.thresholds['correlation_strong']:
                flags.append(PatternFlag(
                    flag_id='FLAG_STRONG_CORRELATION',
                    category='correlation',
                    severity='info',
                    description=f'Strong sector correlation ({sector_corr:.2f})',
                    interpretation='Could you have just traded the sector instead?',
                    value=sector_corr
                ))
            elif abs(sector_corr) < self.thresholds['correlation_weak']:
                flags.append(PatternFlag(
                    flag_id='FLAG_WEAK_CORRELATION',
                    category='correlation',
                    severity='info',
                    description=f'Weak sector correlation ({sector_corr:.2f})',
                    interpretation='Move driven by company-specific (idiosyncratic) factors',
                    value=sector_corr
                ))

        return flags


# =============================================================================
# AGGREGATE PATTERN ANALYSIS
# =============================================================================

@dataclass
class PatternSummary:
    """Summary of patterns across all analyzed trades."""
    total_trades: int
    winners: int
    losers: int
    win_rate: float

    # Win rates by category
    win_rate_by_signal_strength: Dict[str, Tuple[int, int, float]]  # tier -> (wins, total, rate)
    win_rate_by_market_regime: Dict[str, Tuple[int, int, float]]
    win_rate_by_insider_activity: Dict[str, Tuple[int, int, float]]

    # MAE/MFE averages
    avg_mae_winners: float
    avg_mae_losers: float
    avg_mfe_winners: float
    avg_mfe_losers: float

    # Duration
    avg_duration_winners: float
    avg_duration_losers: float

    # Top patterns in winners vs losers
    common_patterns_winners: List[Tuple[str, int]]
    common_patterns_losers: List[Tuple[str, int]]

    # Summary insights
    insights: List[str] = field(default_factory=list)


class AggregatePatternAnalyzer:
    """Analyze patterns across multiple trades."""

    def analyze_patterns(self,
                         trades_data: List[Any],
                         signal_scores: Dict[str, 'SignalStrengthScore'],
                         pattern_flags: Dict[str, List['PatternFlag']]) -> PatternSummary:
        """
        Perform aggregate pattern analysis across trades.

        Args:
            trades_data: List of TradeAnalysisData objects
            signal_scores: Dict mapping trade_id -> SignalStrengthScore
            pattern_flags: Dict mapping trade_id -> list of PatternFlags

        Returns:
            PatternSummary with aggregate analysis
        """
        if not trades_data:
            return self._empty_summary()

        # Categorize trades
        winners = []
        losers = []

        for trade in trades_data:
            pl_pct = trade.trade_info.get('pl_pct', 0)
            if pl_pct > 0:
                winners.append(trade)
            else:
                losers.append(trade)

        total = len(trades_data)
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = (win_count / total * 100) if total > 0 else 0

        # Win rate by signal strength tier
        win_rate_by_signal = self._analyze_by_signal_strength(
            trades_data, signal_scores
        )

        # Win rate by market regime
        win_rate_by_regime = self._analyze_by_market_regime(trades_data)

        # Win rate by insider activity
        win_rate_by_insider = self._analyze_by_insider_activity(
            trades_data, pattern_flags
        )

        # MAE/MFE analysis
        avg_mae_winners = self._calculate_avg_mae(winners)
        avg_mae_losers = self._calculate_avg_mae(losers)
        avg_mfe_winners = self._calculate_avg_mfe(winners)
        avg_mfe_losers = self._calculate_avg_mfe(losers)

        # Duration analysis
        avg_duration_winners = self._calculate_avg_duration(winners)
        avg_duration_losers = self._calculate_avg_duration(losers)

        # Common patterns
        common_winners = self._find_common_patterns(winners, pattern_flags)
        common_losers = self._find_common_patterns(losers, pattern_flags)

        # Generate insights
        insights = self._generate_insights(
            win_rate, win_rate_by_signal, win_rate_by_regime,
            avg_mae_winners, avg_mae_losers,
            common_winners, common_losers
        )

        return PatternSummary(
            total_trades=total,
            winners=win_count,
            losers=loss_count,
            win_rate=win_rate,
            win_rate_by_signal_strength=win_rate_by_signal,
            win_rate_by_market_regime=win_rate_by_regime,
            win_rate_by_insider_activity=win_rate_by_insider,
            avg_mae_winners=avg_mae_winners,
            avg_mae_losers=avg_mae_losers,
            avg_mfe_winners=avg_mfe_winners,
            avg_mfe_losers=avg_mfe_losers,
            avg_duration_winners=avg_duration_winners,
            avg_duration_losers=avg_duration_losers,
            common_patterns_winners=common_winners,
            common_patterns_losers=common_losers,
            insights=insights
        )

    def _empty_summary(self) -> PatternSummary:
        """Return empty summary when no trades available."""
        return PatternSummary(
            total_trades=0, winners=0, losers=0, win_rate=0,
            win_rate_by_signal_strength={},
            win_rate_by_market_regime={},
            win_rate_by_insider_activity={},
            avg_mae_winners=0, avg_mae_losers=0,
            avg_mfe_winners=0, avg_mfe_losers=0,
            avg_duration_winners=0, avg_duration_losers=0,
            common_patterns_winners=[], common_patterns_losers=[],
            insights=["No trades available for analysis"]
        )

    def _analyze_by_signal_strength(self,
                                     trades: List[Any],
                                     signal_scores: Dict[str, 'SignalStrengthScore']
                                     ) -> Dict[str, Tuple[int, int, float]]:
        """Analyze win rate by signal strength tier."""
        tiers = {'excellent': [0, 0], 'good': [0, 0], 'moderate': [0, 0], 'weak': [0, 0]}

        for trade in trades:
            trade_id = trade.trade_id
            score = signal_scores.get(trade_id)
            if score:
                tier = score.tier
                is_winner = trade.trade_info.get('pl_pct', 0) > 0
                if tier in tiers:
                    tiers[tier][1] += 1  # total
                    if is_winner:
                        tiers[tier][0] += 1  # wins

        return {
            tier: (wins, total, (wins/total*100) if total > 0 else 0)
            for tier, (wins, total) in tiers.items()
        }

    def _analyze_by_market_regime(self, trades: List[Any]) -> Dict[str, Tuple[int, int, float]]:
        """Analyze win rate by market regime."""
        regimes = {'uptrend': [0, 0], 'downtrend': [0, 0], 'ranging': [0, 0], 'unknown': [0, 0]}

        for trade in trades:
            regime = trade.market_regime
            if regime:
                trend = regime.trend if hasattr(regime, 'trend') else 'unknown'
            else:
                trend = 'unknown'

            is_winner = trade.trade_info.get('pl_pct', 0) > 0

            if trend in regimes:
                regimes[trend][1] += 1
                if is_winner:
                    regimes[trend][0] += 1

        return {
            regime: (wins, total, (wins/total*100) if total > 0 else 0)
            for regime, (wins, total) in regimes.items()
            if total > 0
        }

    def _analyze_by_insider_activity(self,
                                      trades: List[Any],
                                      pattern_flags: Dict[str, List['PatternFlag']]
                                      ) -> Dict[str, Tuple[int, int, float]]:
        """Analyze win rate by insider activity presence."""
        categories = {'with_insider_buying': [0, 0], 'without_insider_activity': [0, 0]}

        for trade in trades:
            trade_id = trade.trade_id
            flags = pattern_flags.get(trade_id, [])

            has_insider_buying = any(
                f.flag_id == 'FLAG_INSIDER_BUYING_PREENTRY'
                for f in flags if hasattr(f, 'flag_id')
            )

            is_winner = trade.trade_info.get('pl_pct', 0) > 0
            key = 'with_insider_buying' if has_insider_buying else 'without_insider_activity'

            categories[key][1] += 1
            if is_winner:
                categories[key][0] += 1

        return {
            cat: (wins, total, (wins/total*100) if total > 0 else 0)
            for cat, (wins, total) in categories.items()
            if total > 0
        }

    def _calculate_avg_mae(self, trades: List[Any]) -> float:
        """Calculate average MAE for trades."""
        maes = [t.mae_mfe.mae_pct for t in trades if t.mae_mfe]
        return sum(maes) / len(maes) if maes else 0

    def _calculate_avg_mfe(self, trades: List[Any]) -> float:
        """Calculate average MFE for trades."""
        mfes = [t.mae_mfe.mfe_pct for t in trades if t.mae_mfe]
        return sum(mfes) / len(mfes) if mfes else 0

    def _calculate_avg_duration(self, trades: List[Any]) -> float:
        """Calculate average duration for trades."""
        durations = [t.trade_info.get('duration_days', 0) for t in trades]
        return sum(durations) / len(durations) if durations else 0

    def _find_common_patterns(self,
                               trades: List[Any],
                               pattern_flags: Dict[str, List['PatternFlag']]
                               ) -> List[Tuple[str, int]]:
        """Find most common pattern flags in trades."""
        flag_counts = {}

        for trade in trades:
            flags = pattern_flags.get(trade.trade_id, [])
            for flag in flags:
                flag_id = flag.flag_id if hasattr(flag, 'flag_id') else str(flag)
                flag_counts[flag_id] = flag_counts.get(flag_id, 0) + 1

        # Sort by count descending
        sorted_flags = sorted(flag_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_flags[:5]

    def _generate_insights(self,
                           overall_win_rate: float,
                           by_signal: Dict,
                           by_regime: Dict,
                           mae_winners: float,
                           mae_losers: float,
                           common_winners: List,
                           common_losers: List) -> List[str]:
        """Generate actionable insights from pattern analysis."""
        insights = []

        # Signal strength insight
        if by_signal:
            best_tier = max(by_signal.items(), key=lambda x: x[1][2] if x[1][1] > 0 else 0)
            if best_tier[1][1] > 0:
                insights.append(
                    f"Best performance in '{best_tier[0]}' signal strength tier: "
                    f"{best_tier[1][2]:.0f}% win rate ({best_tier[1][0]}/{best_tier[1][1]} trades)"
                )

        # Market regime insight
        if by_regime:
            best_regime = max(by_regime.items(), key=lambda x: x[1][2] if x[1][1] > 0 else 0)
            if best_regime[1][1] > 0:
                insights.append(
                    f"Best results in '{best_regime[0]}' markets: "
                    f"{best_regime[1][2]:.0f}% win rate"
                )

        # MAE insight
        if mae_winners > 0 and mae_losers > 0:
            if mae_losers > mae_winners * 1.5:
                insights.append(
                    f"Losers experience {mae_losers/mae_winners:.1f}x more adverse excursion "
                    f"({mae_losers:.1f}% vs {mae_winners:.1f}%) - consider tighter stops"
                )

        # Common patterns
        if common_winners:
            winner_patterns = [p[0] for p in common_winners[:3]]
            insights.append(f"Common patterns in winners: {', '.join(winner_patterns)}")

        if common_losers:
            loser_patterns = [p[0] for p in common_losers[:3]]
            insights.append(f"Patterns to avoid (common in losers): {', '.join(loser_patterns)}")

        if not insights:
            insights.append("Insufficient data for pattern insights")

        return insights
