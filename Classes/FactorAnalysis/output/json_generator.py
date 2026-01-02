"""
JSON Payload Generator for Factor Analysis.

Generates JSON payloads for GUI consumption with:
- Structured analysis results
- Chart-ready data
- Summary metrics
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from datetime import datetime
from dataclasses import asdict, is_dataclass

from ..logging.audit_logger import AuditLogger


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        elif is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)


class JsonPayloadGenerator:
    """
    Generates JSON payloads for GUI integration.

    Creates structured JSON with:
    - Summary metrics
    - Chart-ready data arrays
    - Detailed analysis results
    - Scenario definitions
    """

    def __init__(
        self,
        logger: Optional[AuditLogger] = None,
        include_raw_data: bool = False,
        max_raw_rows: int = 1000
    ):
        """
        Initialize JsonPayloadGenerator.

        Args:
            logger: Optional audit logger
            include_raw_data: Whether to include raw trade data
            max_raw_rows: Maximum rows of raw data to include
        """
        self.logger = logger
        self.include_raw_data = include_raw_data
        self.max_raw_rows = max_raw_rows

    def generate(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Generate JSON payload from analysis results.

        Args:
            results: Complete analysis results
            output_path: Optional path to save JSON file

        Returns:
            JSON-serializable dictionary
        """
        if self.logger:
            self.logger.start_section("JSON_GENERATION")

        payload = {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_summary(results),
            'data_quality': self._generate_data_quality(results),
            'factor_analysis': self._generate_factor_analysis(results),
            'scenarios': self._generate_scenarios(results),
            'charts': self._generate_chart_data(results)
        }

        if self.include_raw_data and 'enriched_trades' in results:
            payload['raw_data'] = self._generate_raw_data(results['enriched_trades'])

        # Save to file if path provided
        if output_path:
            path = Path(output_path)
            if not path.suffix:
                path = path.with_suffix('.json')

            with open(path, 'w') as f:
                json.dump(payload, f, cls=NumpyEncoder, indent=2)

            if self.logger:
                self.logger.info("JSON payload saved", {'path': str(path)})

        if self.logger:
            self.logger.end_section()

        return payload

    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary section."""
        summary = {
            'trade_counts': {},
            'key_metrics': {},
            'key_findings': [],
            'top_factors': [],
            'best_scenarios': [],
            'worst_scenarios': []
        }

        # Trade counts
        if 'data_summary' in results:
            data_sum = results['data_summary']
            summary['trade_counts'] = {
                'total': data_sum.get('total_trades', 0),
                'good': data_sum.get('good_trades', 0),
                'bad': data_sum.get('bad_trades', 0),
                'indeterminate': data_sum.get('indeterminate_trades', 0)
            }

            # Win rate
            total = summary['trade_counts']['total']
            if total > 0:
                summary['key_metrics']['good_trade_rate'] = summary['trade_counts']['good'] / total
                summary['key_metrics']['bad_trade_rate'] = summary['trade_counts']['bad'] / total

        # Key findings
        if 'key_findings' in results:
            summary['key_findings'] = results['key_findings'][:10]

        # Top factors from Tier 2 regression
        if 'tier2' in results and results['tier2'].get('logistic_regression'):
            reg = results['tier2']['logistic_regression']
            sig_factors = reg.get_significant_factors()[:5]
            for factor in sig_factors:
                summary['top_factors'].append({
                    'name': factor.factor_name,
                    'odds_ratio': float(factor.odds_ratio),
                    'p_value': float(factor.p_value),
                    'direction': 'positive' if factor.coefficient > 0 else 'negative'
                })

        # Best/worst scenarios
        if 'scenarios' in results:
            for scenario in results['scenarios'].get('best_scenarios', [])[:3]:
                summary['best_scenarios'].append({
                    'name': scenario.name,
                    'conditions': scenario.get_condition_string(),
                    'lift': float(scenario.lift),
                    'n_trades': scenario.n_trades
                })

            for scenario in results['scenarios'].get('worst_scenarios', [])[:3]:
                summary['worst_scenarios'].append({
                    'name': scenario.name,
                    'conditions': scenario.get_condition_string(),
                    'lift': float(scenario.lift),
                    'n_trades': scenario.n_trades
                })

        return summary

    def _generate_data_quality(self, results: Dict) -> Dict:
        """Generate data quality section."""
        quality = {
            'overall_score': None,
            'completeness': {},
            'temporal_alignment': {},
            'outliers': {},
            'warnings': []
        }

        if 'quality_score' in results:
            qs = results['quality_score']
            quality['overall_score'] = qs.get('overall_score')
            quality['completeness'] = qs.get('completeness', {})

        if 'temporal_alignment' in results:
            quality['temporal_alignment'] = results['temporal_alignment']

        if 'outlier_result' in results:
            outlier = results['outlier_result']
            quality['outliers'] = {
                'total': outlier.total_outliers,
                'by_column': outlier.outliers_by_column,
                'trades_affected': outlier.trades_with_outliers
            }

        if 'warnings' in results:
            quality['warnings'] = results['warnings']

        return quality

    def _generate_factor_analysis(self, results: Dict) -> Dict:
        """Generate factor analysis section."""
        analysis = {
            'tier1': {},
            'tier2': {},
            'tier3': {},
            'multiple_testing': {}
        }

        # Tier 1: Exploratory
        if 'tier1' in results:
            tier1 = results['tier1']
            analysis['tier1'] = {
                'correlations': [],
                'descriptive_stats': {}
            }

            if 'point_biserial' in tier1:
                for corr in tier1['point_biserial']:
                    analysis['tier1']['correlations'].append({
                        'factor': corr.factor,
                        'correlation': float(corr.correlation),
                        'p_value': float(corr.p_value),
                        'significant': corr.significant
                    })

            if 'descriptive_stats' in tier1:
                analysis['tier1']['descriptive_stats'] = self._convert_to_json_safe(
                    tier1['descriptive_stats']
                )

        # Tier 2: Hypothesis testing
        if 'tier2' in results:
            tier2 = results['tier2']
            analysis['tier2'] = {
                'logistic_regression': None,
                'anova': [],
                'chi_square': {}
            }

            if tier2.get('logistic_regression'):
                reg = tier2['logistic_regression']
                analysis['tier2']['logistic_regression'] = {
                    'n_observations': reg.n_observations,
                    'pseudo_r2': float(reg.pseudo_r2),
                    'aic': float(reg.aic),
                    'bic': float(reg.bic),
                    'coefficients': [
                        {
                            'factor': f.factor_name,
                            'coefficient': float(f.coefficient),
                            'odds_ratio': float(f.odds_ratio),
                            'p_value': float(f.p_value),
                            'ci_lower': float(f.ci_lower),
                            'ci_upper': float(f.ci_upper)
                        }
                        for f in reg.factor_results
                    ]
                }

            if tier2.get('anova'):
                for anova in tier2['anova']:
                    analysis['tier2']['anova'].append({
                        'factor': anova.factor,
                        'statistic': float(anova.statistic),
                        'p_value': float(anova.p_value),
                        'effect_size': float(anova.effect_size or 0),
                        'significant': anova.significant
                    })

        # Tier 3: ML
        if 'tier3' in results and results['tier3'].get('enabled'):
            tier3 = results['tier3']
            analysis['tier3'] = {
                'rf_accuracy': tier3.get('rf_accuracy'),
                'rf_cv_accuracy': tier3.get('rf_cv_accuracy'),
                'feature_importance': [
                    {
                        'feature': f.feature_name,
                        'importance': float(f.importance),
                        'rank': f.rank
                    }
                    for f in tier3.get('rf_feature_importances', [])[:20]
                ],
                'shap_values': [
                    {
                        'feature': s.feature_name,
                        'mean_abs_shap': float(s.mean_abs_shap),
                        'direction': s.direction
                    }
                    for s in tier3.get('shap_results', [])[:20]
                ]
            }

        # Multiple testing correction
        if 'multiple_testing' in results:
            mt = results['multiple_testing']
            analysis['multiple_testing'] = {
                'method': mt.correction_method,
                'n_tests': mt.n_tests,
                'significant_before': mt.n_significant_original,
                'significant_after': mt.n_significant_corrected
            }

        return analysis

    def _generate_scenarios(self, results: Dict) -> Dict:
        """Generate scenarios section."""
        if 'scenarios' not in results:
            return {}

        scenarios_result = results['scenarios']

        return {
            'mode': scenarios_result.mode,
            'baseline': scenarios_result.baseline_metrics,
            'best': [
                {
                    'name': s.name,
                    'conditions': [
                        {
                            'factor': c.factor,
                            'operator': c.operator,
                            'threshold': float(c.threshold)
                        }
                        for c in s.conditions
                    ],
                    'n_trades': s.n_trades,
                    'good_trade_rate': float(s.good_trade_rate),
                    'lift': float(s.lift),
                    'confidence': float(s.confidence),
                    'support': float(s.support)
                }
                for s in scenarios_result.best_scenarios
            ],
            'worst': [
                {
                    'name': s.name,
                    'conditions': [
                        {
                            'factor': c.factor,
                            'operator': c.operator,
                            'threshold': float(c.threshold)
                        }
                        for c in s.conditions
                    ],
                    'n_trades': s.n_trades,
                    'good_trade_rate': float(s.good_trade_rate),
                    'lift': float(s.lift),
                    'confidence': float(s.confidence),
                    'support': float(s.support)
                }
                for s in scenarios_result.worst_scenarios
            ]
        }

    def _generate_chart_data(self, results: Dict) -> Dict:
        """Generate chart-ready data."""
        charts = {}

        # Correlation bar chart
        if 'tier1' in results and 'point_biserial' in results['tier1']:
            correlations = results['tier1']['point_biserial']
            sorted_corrs = sorted(correlations, key=lambda x: abs(x.correlation), reverse=True)[:15]
            charts['correlation_bar'] = {
                'labels': [c.factor for c in sorted_corrs],
                'values': [float(c.correlation) for c in sorted_corrs],
                'significant': [c.significant for c in sorted_corrs]
            }

        # Feature importance chart
        if 'tier3' in results and results['tier3'].get('rf_feature_importances'):
            importances = results['tier3']['rf_feature_importances'][:15]
            charts['feature_importance'] = {
                'labels': [f.feature_name for f in importances],
                'values': [float(f.importance) for f in importances],
                'errors': [float(f.importance_std) for f in importances]
            }

        # Scenario lift comparison
        if 'scenarios' in results:
            best = results['scenarios'].best_scenarios[:5]
            worst = results['scenarios'].worst_scenarios[:5]
            charts['scenario_lift'] = {
                'best': {
                    'labels': [s.name for s in best],
                    'values': [float(s.lift) for s in best]
                },
                'worst': {
                    'labels': [s.name for s in worst],
                    'values': [float(s.lift) for s in worst]
                }
            }

        # Trade class distribution
        if 'data_summary' in results:
            ds = results['data_summary']
            charts['trade_class_pie'] = {
                'labels': ['Good', 'Bad', 'Indeterminate'],
                'values': [
                    ds.get('good_trades', 0),
                    ds.get('bad_trades', 0),
                    ds.get('indeterminate_trades', 0)
                ]
            }

        return charts

    def _generate_raw_data(self, df: pd.DataFrame) -> Dict:
        """Generate raw data section."""
        # Limit rows
        if len(df) > self.max_raw_rows:
            df = df.head(self.max_raw_rows)

        # Convert to records
        records = df.to_dict(orient='records')

        # Clean NaN values
        for record in records:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None

        return {
            'columns': list(df.columns),
            'row_count': len(records),
            'data': records
        }

    def _convert_to_json_safe(self, obj: Any) -> Any:
        """Convert object to JSON-safe format."""
        if isinstance(obj, dict):
            return {k: self._convert_to_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_safe(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif pd.isna(obj):
            return None
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif is_dataclass(obj):
            return asdict(obj)
        return obj

    def generate_summary_only(self, results: Dict) -> Dict:
        """Generate summary-only payload for quick access."""
        return {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'summary': self._generate_summary(results)
        }

    def generate_charts_only(self, results: Dict) -> Dict:
        """Generate charts-only payload for visualization."""
        return {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'charts': self._generate_chart_data(results)
        }
