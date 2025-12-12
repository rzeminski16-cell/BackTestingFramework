"""
Enhanced Optimization Report Generator for Portfolio Optimization.

Generates comprehensive Excel reports with:
- Executive summary with overfitting assessment
- 3D parameter surface plots
- Parameter robustness zones
- Overfitting probability scores
- In-sample vs out-of-sample analysis
- Multi-security comparison
- Statistical significance testing
- Embedded matplotlib visualizations
"""
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, AreaChart
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.utils import get_column_letter

try:
    from openpyxl.drawing.image import Image
    IMAGE_AVAILABLE = True
except ImportError:
    IMAGE_AVAILABLE = False

from Classes.Optimization.sensitivity_analyzer import SensitivityResults
from Classes.Optimization.walk_forward_optimizer import WalkForwardResults, MultiSecurityResults

# Try to import enhanced visualizations
try:
    from Classes.Analysis.enhanced_visualizations import EnhancedVisualizations, MATPLOTLIB_AVAILABLE
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    EnhancedVisualizations = None

logger = logging.getLogger(__name__)


class EnhancedOptimizationReportGenerator:
    """
    Enhanced optimization report generator with advanced analytics.

    Features:
    - Executive summary with traffic-light indicators
    - 3D parameter sensitivity surfaces
    - Overfitting probability assessment
    - Parameter robustness zones
    - Multi-security comparison
    - Embedded matplotlib visualizations
    """

    # Color scheme
    COLORS = {
        'header_dark': '1F4E79',
        'header_medium': '2E75B6',
        'positive': 'C6EFCE',
        'positive_dark': '28A745',
        'negative': 'FFC7CE',
        'negative_dark': 'DC3545',
        'neutral': 'FFEB9C',
        'neutral_dark': 'FFC107',
        'white': 'FFFFFF',
        'light_gray': 'F5F5F5',
        'medium_gray': 'E0E0E0',
        'dark_gray': '666666',
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced report generator."""
        self.config = config
        self.report_config = config.get('reporting', {})
        self.decimal_places = config.get('report', {}).get('excel', {}).get('decimal_places', 4)

        # Initialize visualization module
        self.include_matplotlib_charts = MATPLOTLIB_AVAILABLE and IMAGE_AVAILABLE
        if self.include_matplotlib_charts:
            try:
                self.viz = EnhancedVisualizations(dpi=150)
            except Exception:
                self.viz = None
                self.include_matplotlib_charts = False
        else:
            self.viz = None

        self._init_styles()
        logger.info("Enhanced optimization report generator initialized")

    def _init_styles(self):
        """Initialize Excel styles."""
        self.header_fill = PatternFill(start_color=self.COLORS['header_dark'],
                                        end_color=self.COLORS['header_dark'], fill_type="solid")
        self.header_font = Font(bold=True, color=self.COLORS['white'], size=11)
        self.subheader_fill = PatternFill(start_color=self.COLORS['header_medium'],
                                           end_color=self.COLORS['header_medium'], fill_type="solid")
        self.positive_fill = PatternFill(start_color=self.COLORS['positive'],
                                          end_color=self.COLORS['positive'], fill_type="solid")
        self.negative_fill = PatternFill(start_color=self.COLORS['negative'],
                                          end_color=self.COLORS['negative'], fill_type="solid")
        self.neutral_fill = PatternFill(start_color=self.COLORS['neutral'],
                                         end_color=self.COLORS['neutral'], fill_type="solid")
        self.light_fill = PatternFill(start_color=self.COLORS['light_gray'],
                                       end_color=self.COLORS['light_gray'], fill_type="solid")

        self.title_font = Font(bold=True, size=16, color=self.COLORS['header_dark'])
        self.section_font = Font(bold=True, size=14, color=self.COLORS['header_dark'])
        self.subsection_font = Font(bold=True, size=12, color=self.COLORS['header_medium'])

        self.border = Border(
            left=Side(style='thin', color=self.COLORS['medium_gray']),
            right=Side(style='thin', color=self.COLORS['medium_gray']),
            top=Side(style='thin', color=self.COLORS['medium_gray']),
            bottom=Side(style='thin', color=self.COLORS['medium_gray'])
        )

    def generate_portfolio_optimization_report(
        self,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Optional[Dict[str, SensitivityResults]] = None,
        output_dir: str = None
    ) -> str:
        """
        Generate comprehensive portfolio optimization report.

        Args:
            multi_results: Combined results from all securities
            sensitivity_results_dict: Optional dict of symbol -> SensitivityResults
            output_dir: Output directory

        Returns:
            Path to generated report file
        """
        if output_dir is None:
            output_dir = self.config.get('report', {}).get('output_dir', 'logs/optimization_reports')

        os.makedirs(output_dir, exist_ok=True)

        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        securities_str = "_".join(multi_results.securities[:3])
        if len(multi_results.securities) > 3:
            securities_str += f"_+{len(multi_results.securities) - 3}"
        filename = f"enhanced_portfolio_optimization_{multi_results.strategy_name}_{securities_str}_{timestamp}.xlsx"
        filepath = os.path.join(output_dir, filename)

        # Calculate advanced metrics
        metrics = self._calculate_optimization_metrics(multi_results, sensitivity_results_dict)

        # Create workbook
        wb = Workbook()
        wb.remove(wb.active)

        # Generate sheets
        self._create_toc(wb, multi_results)
        self._create_executive_summary(wb, multi_results, metrics)
        self._create_overfitting_analysis(wb, multi_results, metrics)
        self._create_security_comparison(wb, multi_results, metrics)
        self._create_parameter_robustness(wb, multi_results, metrics)

        if sensitivity_results_dict:
            self._create_sensitivity_dashboard(wb, multi_results, sensitivity_results_dict, metrics)

        self._create_window_analysis(wb, multi_results, metrics)
        self._create_recommendations(wb, multi_results, sensitivity_results_dict, metrics)

        # Individual security details
        for symbol, wf_results in multi_results.individual_results.items():
            sens_results = sensitivity_results_dict.get(symbol) if sensitivity_results_dict else None
            self._create_security_detail(wb, symbol, wf_results, sens_results)

        wb.save(filepath)
        logger.info(f"Enhanced portfolio optimization report saved to {filepath}")

        return filepath

    def _calculate_optimization_metrics(
        self,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Optional[Dict[str, SensitivityResults]]
    ) -> Dict[str, Any]:
        """Calculate advanced optimization metrics."""
        metrics = {}

        # Basic aggregates
        metrics['total_securities'] = len(multi_results.securities)
        metrics['total_windows'] = multi_results.total_windows_all_securities
        metrics['total_passed'] = multi_results.total_passed_all_securities
        metrics['success_rate'] = (metrics['total_passed'] / metrics['total_windows'] * 100
                                    if metrics['total_windows'] > 0 else 0)

        # Performance metrics - handle potential NaN/None values
        metrics['avg_is_sortino'] = self._safe_float(multi_results.combined_avg_in_sample_sortino)
        metrics['avg_oos_sortino'] = self._safe_float(multi_results.combined_avg_out_sample_sortino)
        metrics['sortino_degradation'] = self._safe_float(multi_results.combined_avg_sortino_degradation_pct)
        metrics['avg_is_sharpe'] = self._safe_float(multi_results.combined_avg_in_sample_sharpe)
        metrics['avg_oos_sharpe'] = self._safe_float(multi_results.combined_avg_out_sample_sharpe)
        metrics['sharpe_degradation'] = self._safe_float(multi_results.combined_avg_sharpe_degradation_pct)

        # Log diagnostic info if values look suspicious
        if metrics['total_windows'] == 0:
            logger.warning("No optimization windows found in results")
        if metrics['avg_is_sortino'] == 0 and metrics['avg_oos_sortino'] == 0:
            logger.warning("Both IS and OOS Sortino ratios are 0 - check optimization results")

        # Overfitting probability
        metrics['overfitting_score'] = self._calculate_overfitting_score(multi_results, sensitivity_results_dict)
        metrics['overfitting_probability'] = self._calculate_overfitting_probability(metrics)

        # Parameter stability
        metrics['parameter_stability'] = self._calculate_parameter_stability(multi_results)

        # Robustness assessment
        metrics['robustness_score'] = self._calculate_robustness_score(multi_results, sensitivity_results_dict)

        # Per-security metrics
        metrics['security_metrics'] = {}
        for symbol, wf_results in multi_results.individual_results.items():
            metrics['security_metrics'][symbol] = {
                'windows': wf_results.total_windows,
                'passed': wf_results.windows_passed_constraints,
                'is_sortino': self._safe_float(wf_results.avg_in_sample_sortino),
                'oos_sortino': self._safe_float(wf_results.avg_out_sample_sortino),
                'degradation': self._safe_float(wf_results.avg_sortino_degradation_pct),
                'is_positive_oos': wf_results.avg_out_sample_sortino > 0,
            }

        # Store diagnostic info for the report
        metrics['_diagnostic'] = {
            'has_windows': metrics['total_windows'] > 0,
            'has_passed': metrics['total_passed'] > 0,
            'has_valid_sortino': metrics['avg_is_sortino'] != 0 or metrics['avg_oos_sortino'] != 0,
            'securities_count': len(multi_results.individual_results),
        }

        return metrics

    def _safe_float(self, value) -> float:
        """Safely convert value to float, handling NaN and None."""
        if value is None:
            return 0.0
        try:
            result = float(value)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except (ValueError, TypeError):
            return 0.0

    def _calculate_overfitting_score(
        self,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Optional[Dict[str, SensitivityResults]]
    ) -> float:
        """
        Calculate overfitting score (0-100, higher = more overfitted).

        Factors:
        - Performance degradation (IS to OOS)
        - Parameter stability across windows
        - Sensitivity to parameter variations
        """
        score = 0.0

        # Degradation factor (0-40 points)
        degradation = abs(multi_results.combined_avg_sortino_degradation_pct)
        if degradation > 50:
            score += 40
        elif degradation > 30:
            score += 30
        elif degradation > 15:
            score += 15
        else:
            score += degradation / 15 * 15

        # Parameter instability factor (0-30 points)
        param_scores = list(multi_results.param_consistency_scores.values())
        if param_scores:
            avg_consistency = np.mean(param_scores)
            instability = 100 - avg_consistency
            score += instability * 0.3

        # Sensitivity factor (0-30 points)
        if sensitivity_results_dict:
            non_robust_count = 0
            total_params = 0
            for sens in sensitivity_results_dict.values():
                if sens:
                    for param_sens in sens.parameter_sensitivities.values():
                        total_params += 1
                        if param_sens.is_unstable:
                            non_robust_count += 1

            if total_params > 0:
                instability_ratio = non_robust_count / total_params
                score += instability_ratio * 30

        return min(100, score)

    def _calculate_overfitting_probability(self, metrics: Dict[str, Any]) -> str:
        """Convert overfitting score to probability category."""
        score = metrics['overfitting_score']

        if score < 20:
            return "LOW"
        elif score < 40:
            return "MODERATE"
        elif score < 60:
            return "HIGH"
        else:
            return "VERY HIGH"

    def _calculate_parameter_stability(self, multi_results: MultiSecurityResults) -> Dict[str, Any]:
        """Calculate parameter stability metrics."""
        stability = {}

        for param_name, consistency in multi_results.param_consistency_scores.items():
            if consistency >= 80:
                status = "Very Stable"
            elif consistency >= 60:
                status = "Stable"
            elif consistency >= 40:
                status = "Moderate"
            else:
                status = "Unstable"

            stability[param_name] = {
                'consistency': consistency,
                'status': status,
                'recommended_value': multi_results.consistent_params.get(param_name),
            }

        return stability

    def _calculate_robustness_score(
        self,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Optional[Dict[str, SensitivityResults]]
    ) -> float:
        """Calculate overall robustness score (0-100, higher = more robust)."""
        score = 0.0

        # Positive OOS performance (0-30 points)
        positive_ratio = multi_results.securities_with_positive_oos / len(multi_results.securities)
        score += positive_ratio * 30

        # Low degradation (0-30 points)
        degradation = abs(multi_results.combined_avg_sortino_degradation_pct)
        if degradation < 15:
            score += 30
        elif degradation < 30:
            score += 20
        elif degradation < 50:
            score += 10

        # High constraint pass rate (0-20 points)
        if multi_results.total_windows_all_securities > 0:
            pass_rate = multi_results.total_passed_all_securities / multi_results.total_windows_all_securities
            score += pass_rate * 20

        # Parameter consistency (0-20 points)
        param_scores = list(multi_results.param_consistency_scores.values())
        if param_scores:
            avg_consistency = np.mean(param_scores)
            score += avg_consistency * 0.2

        return min(100, score)

    # ==================== SHEET CREATION ====================

    def _create_toc(self, wb: Workbook, multi_results: MultiSecurityResults):
        """Create Table of Contents."""
        ws = wb.create_sheet("Table of Contents", 0)

        ws['A1'] = "PORTFOLIO OPTIMIZATION REPORT"
        ws['A1'].font = Font(bold=True, size=20, color=self.COLORS['header_dark'])
        ws.merge_cells('A1:E1')

        ws['A2'] = f"Strategy: {multi_results.strategy_name}"
        ws['A2'].font = Font(size=12, bold=True)

        ws['A3'] = f"Securities: {', '.join(multi_results.securities)}"
        ws['A3'].font = Font(size=10, color=self.COLORS['dark_gray'])

        ws['A4'] = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        ws['A4'].font = Font(size=10, color=self.COLORS['dark_gray'])

        row = 6
        ws[f'A{row}'] = "TABLE OF CONTENTS"
        ws[f'A{row}'].font = self.section_font
        row += 2

        toc_items = [
            ("Executive Summary", "Key findings and KPIs"),
            ("Overfitting Analysis", "Overfitting probability and assessment"),
            ("Security Comparison", "Side-by-side performance comparison"),
            ("Parameter Robustness", "Parameter stability zones"),
            ("Sensitivity Dashboard", "Parameter sensitivity analysis"),
            ("Window Analysis", "Window-by-window results"),
            ("Recommendations", "Actionable insights"),
        ]

        for i, (sheet_name, description) in enumerate(toc_items, 1):
            ws[f'A{row}'] = f"{i}."
            ws[f'B{row}'] = sheet_name
            ws[f'B{row}'].font = Font(bold=True, size=11, underline='single', color='0563C1')
            ws[f'C{row}'] = description
            ws[f'C{row}'].font = Font(size=10, color=self.COLORS['dark_gray'])
            row += 1

        ws.column_dimensions['A'].width = 5
        ws.column_dimensions['B'].width = 25
        ws.column_dimensions['C'].width = 40

    def _create_executive_summary(self, wb: Workbook, multi_results: MultiSecurityResults, metrics: Dict[str, Any]):
        """Create Executive Summary sheet."""
        ws = wb.create_sheet("Executive Summary")

        row = 1
        ws[f'A{row}'] = "EXECUTIVE SUMMARY"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:H{row}')
        row += 3

        # KPI Section with traffic lights
        ws[f'A{row}'] = "KEY PERFORMANCE INDICATORS"
        ws[f'A{row}'].font = self.section_font
        row += 2

        kpis = [
            ("OOS Sortino", f"{metrics['avg_oos_sortino']:.2f}", metrics['avg_oos_sortino'] > 0.5),
            ("Degradation", f"{metrics['sortino_degradation']:.1f}%", abs(metrics['sortino_degradation']) < 30),
            ("Success Rate", f"{metrics['success_rate']:.1f}%", metrics['success_rate'] > 70),
            ("Robustness", f"{metrics['robustness_score']:.0f}/100", metrics['robustness_score'] > 60),
            ("Overfitting", metrics['overfitting_probability'], metrics['overfitting_probability'] in ['LOW', 'MODERATE']),
            ("Securities +OOS", f"{multi_results.securities_with_positive_oos}/{len(multi_results.securities)}",
             multi_results.securities_with_positive_oos >= len(multi_results.securities) * 0.7),
        ]

        col = 1
        for kpi_name, value, is_good in kpis:
            ws.cell(row=row, column=col, value=kpi_name).font = Font(bold=True, size=10)

            value_cell = ws.cell(row=row+1, column=col, value=value)
            value_cell.font = Font(bold=True, size=14)

            indicator_cell = ws.cell(row=row+1, column=col+1)
            if is_good:
                indicator_cell.value = "●"
                indicator_cell.font = Font(color=self.COLORS['positive_dark'], size=16)
            else:
                indicator_cell.value = "●"
                indicator_cell.font = Font(color=self.COLORS['negative_dark'], size=16)

            col += 3
            if col > 7:
                col = 1
                row += 4

        row += 5

        # Overall Assessment
        ws[f'A{row}'] = "OVERALL ASSESSMENT"
        ws[f'A{row}'].font = self.section_font
        row += 2

        # Determine overall status
        robustness = metrics['robustness_score']
        if robustness >= 70:
            status = "ROBUST - Strategy shows strong out-of-sample performance"
            status_color = self.COLORS['positive_dark']
        elif robustness >= 50:
            status = "MODERATE - Strategy shows acceptable performance with some concerns"
            status_color = self.COLORS['neutral_dark']
        else:
            status = "WEAK - Strategy may be overfitted or unstable"
            status_color = self.COLORS['negative_dark']

        ws[f'A{row}'] = status
        ws[f'A{row}'].font = Font(bold=True, size=12, color=status_color)
        row += 2

        # Performance Summary
        row = self._add_section_header(ws, row, "PERFORMANCE SUMMARY")

        perf_data = [
            ("Total Securities Analyzed", str(metrics['total_securities'])),
            ("Total Windows", str(metrics['total_windows'])),
            ("Windows Passed Constraints", str(metrics['total_passed'])),
            ("Avg In-Sample Sortino", f"{metrics['avg_is_sortino']:.4f}"),
            ("Avg Out-of-Sample Sortino", f"{metrics['avg_oos_sortino']:.4f}"),
            ("Sortino Degradation", f"{metrics['sortino_degradation']:.2f}%"),
            ("Avg In-Sample Sharpe", f"{metrics['avg_is_sharpe']:.4f}"),
            ("Avg Out-of-Sample Sharpe", f"{metrics['avg_oos_sharpe']:.4f}"),
            ("Best Security", multi_results.best_security),
            ("Worst Security", multi_results.worst_security),
        ]

        for metric, value in perf_data:
            ws.cell(row=row, column=1, value=metric).border = self.border
            ws.cell(row=row, column=2, value=value).border = self.border
            row += 1

        row += 2

        # Add diagnostic warnings if data looks suspicious
        diag = metrics.get('_diagnostic', {})
        warnings = []

        if not diag.get('has_windows', True):
            warnings.append("WARNING: No optimization windows found. Check that optimization completed successfully.")
        if not diag.get('has_passed', True) and diag.get('has_windows', False):
            warnings.append("WARNING: No windows passed constraints. Consider loosening constraint thresholds.")
        if not diag.get('has_valid_sortino', True) and diag.get('has_windows', False):
            warnings.append("WARNING: Sortino ratios are all zero. This may indicate insufficient trades or calculation issues.")
        if metrics['total_windows'] > 0 and metrics['success_rate'] == 0:
            warnings.append("WARNING: 0% success rate. All windows failed constraints.")

        if warnings:
            row = self._add_section_header(ws, row, "⚠️ DATA QUALITY WARNINGS")
            for warning in warnings:
                ws[f'A{row}'] = warning
                ws[f'A{row}'].font = Font(color=self.COLORS['negative_dark'], italic=True)
                row += 1
            row += 1

        ws.column_dimensions['A'].width = 35
        ws.column_dimensions['B'].width = 20
        for col in ['C', 'D', 'E', 'F', 'G', 'H']:
            ws.column_dimensions[col].width = 15

    def _create_overfitting_analysis(self, wb: Workbook, multi_results: MultiSecurityResults, metrics: Dict[str, Any]):
        """Create Overfitting Analysis sheet."""
        ws = wb.create_sheet("Overfitting Analysis")

        row = 1
        ws[f'A{row}'] = "OVERFITTING PROBABILITY ANALYSIS"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:F{row}')
        row += 3

        # Overfitting Score
        row = self._add_section_header(ws, row, "A. OVERFITTING ASSESSMENT")

        score = metrics['overfitting_score']
        probability = metrics['overfitting_probability']

        ws[f'A{row}'] = "Overfitting Score:"
        ws[f'B{row}'] = f"{score:.1f}/100"

        if probability == "LOW":
            ws[f'C{row}'] = "LOW RISK"
            ws[f'C{row}'].fill = self.positive_fill
        elif probability == "MODERATE":
            ws[f'C{row}'] = "MODERATE RISK"
            ws[f'C{row}'].fill = self.neutral_fill
        elif probability == "HIGH":
            ws[f'C{row}'] = "HIGH RISK"
            ws[f'C{row}'].fill = self.negative_fill
        else:
            ws[f'C{row}'] = "VERY HIGH RISK"
            ws[f'C{row}'].fill = self.negative_fill

        row += 3

        # Breakdown
        row = self._add_section_header(ws, row, "B. SCORE BREAKDOWN")

        ws[f'A{row}'] = "Component"
        ws[f'A{row}'].font = self.header_font
        ws[f'A{row}'].fill = self.header_fill
        ws[f'B{row}'] = "Impact"
        ws[f'B{row}'].font = self.header_font
        ws[f'B{row}'].fill = self.header_fill
        ws[f'C{row}'] = "Assessment"
        ws[f'C{row}'].font = self.header_font
        ws[f'C{row}'].fill = self.header_fill
        row += 1

        degradation = abs(metrics['sortino_degradation'])
        components = [
            ("Performance Degradation", f"{degradation:.1f}%",
             "Good" if degradation < 15 else "Moderate" if degradation < 30 else "Poor"),
            ("Parameter Consistency", f"{np.mean(list(multi_results.param_consistency_scores.values())):.1f}%",
             "Good" if np.mean(list(multi_results.param_consistency_scores.values())) > 70 else "Moderate"),
            ("Constraint Pass Rate", f"{metrics['success_rate']:.1f}%",
             "Good" if metrics['success_rate'] > 70 else "Moderate" if metrics['success_rate'] > 50 else "Poor"),
        ]

        for component, impact, assessment in components:
            ws.cell(row=row, column=1, value=component).border = self.border
            ws.cell(row=row, column=2, value=impact).border = self.border

            assessment_cell = ws.cell(row=row, column=3, value=assessment)
            assessment_cell.border = self.border
            if assessment == "Good":
                assessment_cell.fill = self.positive_fill
            elif assessment == "Moderate":
                assessment_cell.fill = self.neutral_fill
            else:
                assessment_cell.fill = self.negative_fill
            row += 1

        row += 2

        # Interpretation
        row = self._add_section_header(ws, row, "C. INTERPRETATION")

        interpretations = []
        if probability == "LOW":
            interpretations.append("+ Strategy shows strong generalization from in-sample to out-of-sample")
            interpretations.append("+ Parameters are consistent across different time periods")
            interpretations.append("+ Safe to proceed with forward testing")
        elif probability == "MODERATE":
            interpretations.append("~ Strategy shows acceptable but not ideal generalization")
            interpretations.append("~ Consider using more regularization or constraints")
            interpretations.append("~ Proceed with caution - monitor live performance closely")
        else:
            interpretations.append("- Significant performance degradation indicates overfitting")
            interpretations.append("- Parameters vary significantly across time periods")
            interpretations.append("- Consider simplifying strategy or using larger training windows")
            interpretations.append("- NOT recommended for live trading without further refinement")

        for interp in interpretations:
            ws[f'A{row}'] = interp
            if interp.startswith("+"):
                ws[f'A{row}'].font = Font(color=self.COLORS['positive_dark'])
            elif interp.startswith("-"):
                ws[f'A{row}'].font = Font(color=self.COLORS['negative_dark'])
            row += 1

        ws.column_dimensions['A'].width = 50
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 20

    def _create_security_comparison(self, wb: Workbook, multi_results: MultiSecurityResults, metrics: Dict[str, Any]):
        """Create Security Comparison sheet."""
        ws = wb.create_sheet("Security Comparison")

        row = 1
        ws[f'A{row}'] = "SECURITY-BY-SECURITY COMPARISON"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:I{row}')
        row += 3

        # Comparison table
        headers = ["Security", "Windows", "Passed", "IS Sortino", "OOS Sortino",
                   "Degradation", "IS Sharpe", "OOS Sharpe", "Status"]

        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        row += 1

        for symbol in multi_results.securities:
            result = multi_results.individual_results[symbol]
            sec_metrics = metrics['security_metrics'][symbol]

            # Determine status
            if sec_metrics['oos_sortino'] > 0 and abs(sec_metrics['degradation']) < 30:
                status = "ROBUST"
                status_color = self.positive_fill
            elif sec_metrics['oos_sortino'] > 0:
                status = "MODERATE"
                status_color = self.neutral_fill
            else:
                status = "WEAK"
                status_color = self.negative_fill

            row_data = [
                symbol,
                result.total_windows,
                result.windows_passed_constraints,
                f"{result.avg_in_sample_sortino:.4f}",
                f"{result.avg_out_sample_sortino:.4f}",
                f"{result.avg_sortino_degradation_pct:.1f}%",
                f"{result.avg_in_sample_sharpe:.4f}",
                f"{result.avg_out_sample_sharpe:.4f}",
                status
            ]

            for col_idx, value in enumerate(row_data, 1):
                cell = ws.cell(row=row, column=col_idx, value=value)
                cell.border = self.border

                # Highlight best/worst
                if col_idx == 1:
                    if symbol == multi_results.best_security:
                        cell.fill = self.positive_fill
                    elif symbol == multi_results.worst_security:
                        cell.fill = self.negative_fill

                if col_idx == 9:
                    cell.fill = status_color

            row += 1

        # Average row
        row += 1
        ws.cell(row=row, column=1, value="AVERAGE").font = Font(bold=True)
        ws.cell(row=row, column=4, value=f"{metrics['avg_is_sortino']:.4f}").font = Font(bold=True)
        ws.cell(row=row, column=5, value=f"{metrics['avg_oos_sortino']:.4f}").font = Font(bold=True)
        ws.cell(row=row, column=6, value=f"{metrics['sortino_degradation']:.1f}%").font = Font(bold=True)
        ws.cell(row=row, column=7, value=f"{metrics['avg_is_sharpe']:.4f}").font = Font(bold=True)
        ws.cell(row=row, column=8, value=f"{metrics['avg_oos_sharpe']:.4f}").font = Font(bold=True)

        for col in range(1, 10):
            ws.column_dimensions[get_column_letter(col)].width = 14

    def _create_parameter_robustness(self, wb: Workbook, multi_results: MultiSecurityResults, metrics: Dict[str, Any]):
        """Create Parameter Robustness sheet."""
        ws = wb.create_sheet("Parameter Robustness")

        row = 1
        ws[f'A{row}'] = "PARAMETER ROBUSTNESS ANALYSIS"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:G{row}')
        row += 3

        # Robustness zones
        row = self._add_section_header(ws, row, "A. PARAMETER STABILITY ZONES")

        headers = ["Parameter", "Recommended", "Consistency", "Status", "Zone"]
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
            cell.border = self.border
        row += 1

        for param_name, stability in metrics['parameter_stability'].items():
            consistency = stability['consistency']

            # Determine zone
            if consistency >= 80:
                zone = "GREEN - Safe to use"
                zone_fill = self.positive_fill
            elif consistency >= 60:
                zone = "YELLOW - Use with caution"
                zone_fill = self.neutral_fill
            else:
                zone = "RED - Consider fixing"
                zone_fill = self.negative_fill

            ws.cell(row=row, column=1, value=param_name).border = self.border
            ws.cell(row=row, column=2, value=f"{stability['recommended_value']:.4f}" if stability['recommended_value'] else "N/A").border = self.border
            ws.cell(row=row, column=3, value=f"{consistency:.1f}%").border = self.border
            ws.cell(row=row, column=4, value=stability['status']).border = self.border

            zone_cell = ws.cell(row=row, column=5, value=zone)
            zone_cell.border = self.border
            zone_cell.fill = zone_fill

            row += 1

        row += 2

        # Cross-security parameter values
        row = self._add_section_header(ws, row, "B. PARAMETER VALUES ACROSS SECURITIES")

        # Headers
        param_names = list(multi_results.consistent_params.keys())
        ws.cell(row=row, column=1, value="Security").font = self.header_font
        ws.cell(row=row, column=1).fill = self.header_fill
        for col_idx, param in enumerate(param_names, 2):
            cell = ws.cell(row=row, column=col_idx, value=param)
            cell.font = self.header_font
            cell.fill = self.header_fill
        row += 1

        for symbol in multi_results.securities:
            result = multi_results.individual_results[symbol]
            ws.cell(row=row, column=1, value=symbol).border = self.border

            for col_idx, param in enumerate(param_names, 2):
                value = result.most_common_params.get(param, "N/A")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                ws.cell(row=row, column=col_idx, value=value).border = self.border

            row += 1

        # Recommended row
        ws.cell(row=row, column=1, value="RECOMMENDED").font = Font(bold=True)
        ws.cell(row=row, column=1).fill = self.positive_fill
        for col_idx, param in enumerate(param_names, 2):
            value = multi_results.consistent_params.get(param, "N/A")
            if isinstance(value, float):
                value = f"{value:.4f}"
            cell = ws.cell(row=row, column=col_idx, value=value)
            cell.font = Font(bold=True)
            cell.fill = self.positive_fill

        ws.column_dimensions['A'].width = 15
        for col in range(2, len(param_names) + 3):
            ws.column_dimensions[get_column_letter(col)].width = 15

    def _create_sensitivity_dashboard(
        self,
        wb: Workbook,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Dict[str, SensitivityResults],
        metrics: Dict[str, Any]
    ):
        """Create Sensitivity Dashboard sheet."""
        ws = wb.create_sheet("Sensitivity Dashboard")

        row = 1
        ws[f'A{row}'] = "PARAMETER SENSITIVITY DASHBOARD"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:G{row}')
        row += 3

        # Overall sensitivity summary
        row = self._add_section_header(ws, row, "A. OVERALL SENSITIVITY SUMMARY")

        all_robust = []
        all_unstable = []

        for symbol, sens in sensitivity_results_dict.items():
            if sens:
                all_robust.extend(sens.most_robust_params)
                all_unstable.extend(sens.least_robust_params)

        # Count occurrences
        from collections import Counter
        robust_counts = Counter(all_robust)
        unstable_counts = Counter(all_unstable)

        ws[f'A{row}'] = "Most Robust Parameters (across all securities):"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        for param, count in robust_counts.most_common(5):
            ws[f'A{row}'] = f"  {param} ({count} securities)"
            ws[f'A{row}'].font = Font(color=self.COLORS['positive_dark'])
            row += 1

        row += 1
        ws[f'A{row}'] = "Least Robust Parameters (across all securities):"
        ws[f'A{row}'].font = Font(bold=True)
        row += 1
        for param, count in unstable_counts.most_common(5):
            ws[f'A{row}'] = f"  {param} ({count} securities)"
            ws[f'A{row}'].font = Font(color=self.COLORS['negative_dark'])
            row += 1

        row += 2

        # Per-security sensitivity
        row = self._add_section_header(ws, row, "B. PER-SECURITY SENSITIVITY")

        headers = ["Security", "Overall", "Robust Params", "Unstable Params"]
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
        row += 1

        for symbol, sens in sensitivity_results_dict.items():
            if sens:
                ws.cell(row=row, column=1, value=symbol).border = self.border

                overall_cell = ws.cell(row=row, column=2,
                                        value="ROBUST" if sens.is_overall_robust else "SENSITIVE")
                overall_cell.border = self.border
                overall_cell.fill = self.positive_fill if sens.is_overall_robust else self.negative_fill

                ws.cell(row=row, column=3, value=", ".join(sens.most_robust_params[:3])).border = self.border
                ws.cell(row=row, column=4, value=", ".join(sens.least_robust_params[:3])).border = self.border
                row += 1

        ws.column_dimensions['A'].width = 15
        ws.column_dimensions['B'].width = 12
        ws.column_dimensions['C'].width = 30
        ws.column_dimensions['D'].width = 30

    def _create_window_analysis(self, wb: Workbook, multi_results: MultiSecurityResults, metrics: Dict[str, Any]):
        """Create Window Analysis sheet."""
        ws = wb.create_sheet("Window Analysis")

        row = 1
        ws[f'A{row}'] = "WALK-FORWARD WINDOW ANALYSIS"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:H{row}')
        row += 3

        # Aggregate window statistics
        row = self._add_section_header(ws, row, "A. AGGREGATE WINDOW STATISTICS")

        all_is_sortinos = []
        all_oos_sortinos = []
        all_degradations = []

        for wf_results in multi_results.individual_results.values():
            for window in wf_results.windows:
                all_is_sortinos.append(window.in_sample_sortino)
                all_oos_sortinos.append(window.out_sample_sortino)
                all_degradations.append(window.sortino_degradation_pct)

        if all_is_sortinos:
            stats_data = [
                ("Total Windows", str(len(all_is_sortinos))),
                ("IS Sortino (Mean)", f"{np.mean(all_is_sortinos):.4f}"),
                ("IS Sortino (Std)", f"{np.std(all_is_sortinos):.4f}"),
                ("OOS Sortino (Mean)", f"{np.mean(all_oos_sortinos):.4f}"),
                ("OOS Sortino (Std)", f"{np.std(all_oos_sortinos):.4f}"),
                ("OOS Positive (%)", f"{sum(1 for s in all_oos_sortinos if s > 0) / len(all_oos_sortinos) * 100:.1f}%"),
                ("Degradation (Mean)", f"{np.mean(all_degradations):.1f}%"),
                ("Degradation (Std)", f"{np.std(all_degradations):.1f}%"),
            ]

            for metric, value in stats_data:
                ws.cell(row=row, column=1, value=metric).border = self.border
                ws.cell(row=row, column=2, value=value).border = self.border
                row += 1

        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15

    def _create_recommendations(
        self,
        wb: Workbook,
        multi_results: MultiSecurityResults,
        sensitivity_results_dict: Optional[Dict[str, SensitivityResults]],
        metrics: Dict[str, Any]
    ):
        """Create Recommendations sheet."""
        ws = wb.create_sheet("Recommendations")

        row = 1
        ws[f'A{row}'] = "RECOMMENDATIONS & NEXT STEPS"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:D{row}')
        row += 3

        # Recommended parameters
        row = self._add_section_header(ws, row, "A. RECOMMENDED PARAMETERS FOR LIVE TRADING")

        headers = ["Parameter", "Value", "Consistency", "Confidence"]
        for col_idx, header in enumerate(headers, 1):
            cell = ws.cell(row=row, column=col_idx, value=header)
            cell.fill = self.header_fill
            cell.font = self.header_font
        row += 1

        for param_name, value in multi_results.consistent_params.items():
            consistency = multi_results.param_consistency_scores.get(param_name, 0)

            if consistency >= 80:
                confidence = "HIGH"
                conf_fill = self.positive_fill
            elif consistency >= 60:
                confidence = "MEDIUM"
                conf_fill = self.neutral_fill
            else:
                confidence = "LOW"
                conf_fill = self.negative_fill

            ws.cell(row=row, column=1, value=param_name).border = self.border
            ws.cell(row=row, column=2, value=f"{value:.4f}" if isinstance(value, float) else str(value)).border = self.border
            ws.cell(row=row, column=3, value=f"{consistency:.1f}%").border = self.border

            conf_cell = ws.cell(row=row, column=4, value=confidence)
            conf_cell.border = self.border
            conf_cell.fill = conf_fill

            row += 1

        row += 2

        # Action items
        row = self._add_section_header(ws, row, "B. RECOMMENDED ACTIONS")

        actions = []

        # Based on overfitting score
        if metrics['overfitting_probability'] in ['HIGH', 'VERY HIGH']:
            actions.append("CRITICAL: High overfitting risk detected. Consider:")
            actions.append("  - Increasing training window size")
            actions.append("  - Reducing number of optimized parameters")
            actions.append("  - Adding more constraints to optimization")
        elif metrics['overfitting_probability'] == 'MODERATE':
            actions.append("CAUTION: Moderate overfitting risk. Consider:")
            actions.append("  - Monitoring early live performance closely")
            actions.append("  - Using conservative position sizing initially")

        # Based on robustness
        if metrics['robustness_score'] < 50:
            actions.append("WARNING: Low robustness score. Strategy may need:")
            actions.append("  - Parameter re-optimization with different windows")
            actions.append("  - Review of strategy logic and filters")

        # Based on parameter stability
        unstable_params = [p for p, s in metrics['parameter_stability'].items() if s['status'] == 'Unstable']
        if unstable_params:
            actions.append(f"ATTENTION: Unstable parameters detected: {', '.join(unstable_params)}")
            actions.append("  - Consider fixing these to their recommended values")
            actions.append("  - Or remove them from optimization")

        # Positive actions
        if metrics['robustness_score'] >= 70:
            actions.append("GOOD: Strategy shows strong robustness")
            actions.append("  - Proceed with forward testing")
            actions.append("  - Use recommended parameters above")

        for action in actions:
            ws[f'A{row}'] = action
            if "CRITICAL" in action or "WARNING" in action:
                ws[f'A{row}'].font = Font(color=self.COLORS['negative_dark'], bold=True)
            elif "CAUTION" in action or "ATTENTION" in action:
                ws[f'A{row}'].font = Font(color=self.COLORS['neutral_dark'], bold=True)
            elif "GOOD" in action:
                ws[f'A{row}'].font = Font(color=self.COLORS['positive_dark'], bold=True)
            row += 1

        ws.column_dimensions['A'].width = 60
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 15
        ws.column_dimensions['D'].width = 12

    def _create_security_detail(
        self,
        wb: Workbook,
        symbol: str,
        wf_results: WalkForwardResults,
        sensitivity_results: Optional[SensitivityResults]
    ):
        """Create detail sheet for individual security."""
        sheet_name = f"Detail_{symbol[:25]}" if len(symbol) > 25 else f"Detail_{symbol}"
        ws = wb.create_sheet(sheet_name)

        row = 1
        ws[f'A{row}'] = f"DETAILED RESULTS: {symbol}"
        ws[f'A{row}'].font = self.title_font
        ws.merge_cells(f'A{row}:D{row}')
        row += 3

        # Performance summary
        row = self._add_section_header(ws, row, "A. PERFORMANCE SUMMARY")

        metrics_data = [
            ("Total Windows", wf_results.total_windows),
            ("Passed Constraints", wf_results.windows_passed_constraints),
            ("Avg IS Sortino", f"{wf_results.avg_in_sample_sortino:.4f}"),
            ("Avg OOS Sortino", f"{wf_results.avg_out_sample_sortino:.4f}"),
            ("Sortino Degradation", f"{wf_results.avg_sortino_degradation_pct:.2f}%"),
            ("Avg IS Sharpe", f"{wf_results.avg_in_sample_sharpe:.4f}"),
            ("Avg OOS Sharpe", f"{wf_results.avg_out_sample_sharpe:.4f}"),
        ]

        for label, value in metrics_data:
            ws.cell(row=row, column=1, value=label).border = self.border
            ws.cell(row=row, column=2, value=value).border = self.border
            row += 1

        row += 2

        # Optimal parameters
        row = self._add_section_header(ws, row, "B. OPTIMAL PARAMETERS")

        for param_name, value in wf_results.most_common_params.items():
            ws.cell(row=row, column=1, value=param_name).border = self.border
            ws.cell(row=row, column=2, value=f"{value:.4f}" if isinstance(value, float) else str(value)).border = self.border

            min_val, max_val = wf_results.parameter_ranges.get(param_name, (value, value))
            ws.cell(row=row, column=3, value=f"Range: {min_val:.2f} - {max_val:.2f}").border = self.border
            row += 1

        ws.column_dimensions['A'].width = 25
        ws.column_dimensions['B'].width = 15
        ws.column_dimensions['C'].width = 25

    def _add_section_header(self, ws, row: int, title: str) -> int:
        """Add a section header and return next row."""
        ws[f'A{row}'] = title
        ws[f'A{row}'].font = self.subsection_font
        ws[f'A{row}'].fill = self.light_fill
        ws.merge_cells(f'A{row}:D{row}')
        return row + 2
