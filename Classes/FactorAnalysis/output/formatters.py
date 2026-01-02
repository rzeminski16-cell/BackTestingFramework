"""
Formatting utilities for Factor Analysis output.

Provides formatters for:
- Tables (console, markdown, HTML)
- Charts data preparation
- Summary text generation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


@dataclass
class TableStyle:
    """Table formatting style."""
    header_separator: str = '-'
    column_separator: str = '|'
    precision: int = 4
    max_width: int = 120
    truncate_strings: int = 50


class TableFormatter:
    """Formats data as tables for various outputs."""

    def __init__(self, style: Optional[TableStyle] = None):
        """
        Initialize TableFormatter.

        Args:
            style: Table formatting style
        """
        self.style = style or TableStyle()

    def format_dataframe(
        self,
        df: pd.DataFrame,
        title: Optional[str] = None,
        format_type: str = 'console'
    ) -> str:
        """
        Format DataFrame for output.

        Args:
            df: DataFrame to format
            title: Optional title
            format_type: Output format ('console', 'markdown', 'html')

        Returns:
            Formatted string
        """
        if format_type == 'markdown':
            return self._format_markdown(df, title)
        elif format_type == 'html':
            return self._format_html(df, title)
        else:
            return self._format_console(df, title)

    def _format_console(self, df: pd.DataFrame, title: Optional[str]) -> str:
        """Format for console output."""
        lines = []

        if title:
            lines.append(title)
            lines.append('=' * len(title))
            lines.append('')

        # Format numeric columns
        df_formatted = df.copy()
        for col in df_formatted.columns:
            if pd.api.types.is_float_dtype(df_formatted[col]):
                df_formatted[col] = df_formatted[col].apply(
                    lambda x: f"{x:.{self.style.precision}f}" if pd.notna(x) else 'N/A'
                )

        # Use pandas to_string with custom width
        table_str = df_formatted.to_string(
            max_cols=10,
            max_rows=50,
            show_dimensions=True
        )
        lines.append(table_str)

        return '\n'.join(lines)

    def _format_markdown(self, df: pd.DataFrame, title: Optional[str]) -> str:
        """Format as Markdown table."""
        lines = []

        if title:
            lines.append(f"## {title}")
            lines.append('')

        # Header
        headers = list(df.columns)
        lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
        lines.append('|' + '|'.join(['---'] * len(headers)) + '|')

        # Rows
        for _, row in df.iterrows():
            values = []
            for val in row:
                if pd.isna(val):
                    values.append('N/A')
                elif isinstance(val, float):
                    values.append(f"{val:.{self.style.precision}f}")
                else:
                    str_val = str(val)
                    if len(str_val) > self.style.truncate_strings:
                        str_val = str_val[:self.style.truncate_strings-3] + '...'
                    values.append(str_val)
            lines.append('| ' + ' | '.join(values) + ' |')

        return '\n'.join(lines)

    def _format_html(self, df: pd.DataFrame, title: Optional[str]) -> str:
        """Format as HTML table."""
        lines = ['<div class="factor-analysis-table">']

        if title:
            lines.append(f'<h3>{title}</h3>')

        lines.append('<table class="table table-striped">')

        # Header
        lines.append('<thead><tr>')
        for col in df.columns:
            lines.append(f'<th>{col}</th>')
        lines.append('</tr></thead>')

        # Body
        lines.append('<tbody>')
        for _, row in df.iterrows():
            lines.append('<tr>')
            for val in row:
                if pd.isna(val):
                    cell = 'N/A'
                elif isinstance(val, float):
                    cell = f"{val:.{self.style.precision}f}"
                else:
                    cell = str(val)
                lines.append(f'<td>{cell}</td>')
            lines.append('</tr>')
        lines.append('</tbody>')

        lines.append('</table></div>')

        return '\n'.join(lines)

    def format_correlation_table(
        self,
        correlations: List[Any],
        format_type: str = 'console'
    ) -> str:
        """Format correlation results as table."""
        data = []
        for corr in correlations:
            data.append({
                'Factor': corr.factor,
                'Correlation': corr.correlation,
                'P-Value': corr.p_value,
                'Significant': 'Yes' if corr.significant else 'No'
            })

        df = pd.DataFrame(data)
        return self.format_dataframe(df, 'Factor Correlations', format_type)

    def format_regression_table(
        self,
        regression_result: Any,
        format_type: str = 'console'
    ) -> str:
        """Format regression results as table."""
        data = []
        for factor in regression_result.factor_results:
            data.append({
                'Factor': factor.factor_name,
                'Coefficient': factor.coefficient,
                'Odds Ratio': factor.odds_ratio,
                'P-Value': factor.p_value,
                '95% CI': f"[{factor.ci_lower:.3f}, {factor.ci_upper:.3f}]"
            })

        df = pd.DataFrame(data)
        title = f"Logistic Regression (N={regression_result.n_observations}, R²={regression_result.pseudo_r2:.3f})"
        return self.format_dataframe(df, title, format_type)

    def format_scenario_table(
        self,
        scenarios: List[Any],
        scenario_type: str = 'best',
        format_type: str = 'console'
    ) -> str:
        """Format scenarios as table."""
        data = []
        for scenario in scenarios:
            data.append({
                'Scenario': scenario.name,
                'Conditions': scenario.get_condition_string()[:50],
                'Trades': scenario.n_trades,
                'Good Rate': f"{scenario.good_trade_rate:.1%}",
                'Lift': f"{scenario.lift:.2f}",
                'Confidence': f"{scenario.confidence:.1%}"
            })

        df = pd.DataFrame(data)
        title = f"{'Best' if scenario_type == 'best' else 'Worst'} Trading Scenarios"
        return self.format_dataframe(df, title, format_type)


class ChartFormatter:
    """Prepares data for chart visualization."""

    def __init__(self):
        """Initialize ChartFormatter."""
        pass

    def format_bar_chart_data(
        self,
        labels: List[str],
        values: List[float],
        title: str = '',
        colors: Optional[List[str]] = None
    ) -> Dict:
        """
        Format data for bar chart.

        Args:
            labels: Category labels
            values: Bar values
            title: Chart title
            colors: Optional bar colors

        Returns:
            Chart data dictionary
        """
        return {
            'type': 'bar',
            'title': title,
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': values,
                    'backgroundColor': colors or self._generate_colors(len(labels))
                }]
            }
        }

    def format_line_chart_data(
        self,
        x_values: List[Any],
        y_values: List[float],
        title: str = '',
        x_label: str = '',
        y_label: str = ''
    ) -> Dict:
        """
        Format data for line chart.

        Args:
            x_values: X-axis values
            y_values: Y-axis values
            title: Chart title
            x_label: X-axis label
            y_label: Y-axis label

        Returns:
            Chart data dictionary
        """
        return {
            'type': 'line',
            'title': title,
            'data': {
                'labels': x_values,
                'datasets': [{
                    'data': y_values,
                    'borderColor': '#4472C4',
                    'fill': False
                }]
            },
            'options': {
                'scales': {
                    'x': {'title': {'display': True, 'text': x_label}},
                    'y': {'title': {'display': True, 'text': y_label}}
                }
            }
        }

    def format_scatter_data(
        self,
        x_values: List[float],
        y_values: List[float],
        labels: Optional[List[str]] = None,
        title: str = ''
    ) -> Dict:
        """
        Format data for scatter plot.

        Args:
            x_values: X coordinates
            y_values: Y coordinates
            labels: Point labels
            title: Chart title

        Returns:
            Chart data dictionary
        """
        points = [{'x': x, 'y': y} for x, y in zip(x_values, y_values)]

        return {
            'type': 'scatter',
            'title': title,
            'data': {
                'datasets': [{
                    'data': points,
                    'labels': labels,
                    'backgroundColor': '#4472C4'
                }]
            }
        }

    def format_heatmap_data(
        self,
        matrix: pd.DataFrame,
        title: str = ''
    ) -> Dict:
        """
        Format data for heatmap.

        Args:
            matrix: Correlation or other matrix
            title: Chart title

        Returns:
            Chart data dictionary
        """
        return {
            'type': 'heatmap',
            'title': title,
            'data': {
                'x_labels': list(matrix.columns),
                'y_labels': list(matrix.index),
                'values': matrix.values.tolist()
            }
        }

    def format_pie_chart_data(
        self,
        labels: List[str],
        values: List[float],
        title: str = ''
    ) -> Dict:
        """
        Format data for pie chart.

        Args:
            labels: Slice labels
            values: Slice values
            title: Chart title

        Returns:
            Chart data dictionary
        """
        return {
            'type': 'pie',
            'title': title,
            'data': {
                'labels': labels,
                'datasets': [{
                    'data': values,
                    'backgroundColor': self._generate_colors(len(labels))
                }]
            }
        }

    def _generate_colors(self, n: int) -> List[str]:
        """Generate n distinct colors."""
        base_colors = [
            '#4472C4', '#ED7D31', '#A5A5A5', '#FFC000', '#5B9BD5',
            '#70AD47', '#9E480E', '#636363', '#997300', '#1F4E79',
            '#C5E0B4', '#BDD7EE', '#F8CBAD', '#FFE699', '#DBDBDB'
        ]
        return (base_colors * ((n // len(base_colors)) + 1))[:n]


class ResultFormatter:
    """Formats complete analysis results."""

    def __init__(self):
        """Initialize ResultFormatter."""
        self.table_formatter = TableFormatter()
        self.chart_formatter = ChartFormatter()

    def format_summary_text(self, results: Dict) -> str:
        """
        Generate text summary of analysis results.

        Args:
            results: Complete analysis results

        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 60)
        lines.append("FACTOR ANALYSIS SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        # Data summary
        if 'data_summary' in results:
            ds = results['data_summary']
            lines.append("DATA OVERVIEW")
            lines.append("-" * 40)
            lines.append(f"Total Trades: {ds.get('total_trades', 'N/A')}")
            lines.append(f"  Good: {ds.get('good_trades', 'N/A')}")
            lines.append(f"  Bad: {ds.get('bad_trades', 'N/A')}")
            lines.append(f"  Indeterminate: {ds.get('indeterminate_trades', 'N/A')}")
            lines.append("")

        # Key findings
        if 'key_findings' in results:
            lines.append("KEY FINDINGS")
            lines.append("-" * 40)
            for finding in results['key_findings'][:5]:
                lines.append(f"  * {finding}")
            lines.append("")

        # Top significant factors
        if 'tier2' in results and results['tier2'].get('logistic_regression'):
            lines.append("TOP SIGNIFICANT FACTORS")
            lines.append("-" * 40)
            reg = results['tier2']['logistic_regression']
            for factor in reg.get_significant_factors()[:5]:
                direction = "increases" if factor.coefficient > 0 else "decreases"
                lines.append(
                    f"  * {factor.factor_name}: {direction} P(good) "
                    f"(OR={factor.odds_ratio:.2f}, p={factor.p_value:.4f})"
                )
            lines.append("")

        # Best scenarios
        if 'scenarios' in results and results['scenarios'].get('best_scenarios'):
            lines.append("BEST TRADING SCENARIOS")
            lines.append("-" * 40)
            for scenario in results['scenarios']['best_scenarios'][:3]:
                lines.append(f"  * {scenario.name}")
                lines.append(f"    Conditions: {scenario.get_condition_string()}")
                lines.append(f"    Lift: {scenario.lift:.2f}x ({scenario.n_trades} trades)")
            lines.append("")

        # Worst scenarios
        if 'scenarios' in results and results['scenarios'].get('worst_scenarios'):
            lines.append("WORST TRADING SCENARIOS")
            lines.append("-" * 40)
            for scenario in results['scenarios']['worst_scenarios'][:3]:
                lines.append(f"  * {scenario.name}")
                lines.append(f"    Conditions: {scenario.get_condition_string()}")
                lines.append(f"    Lift: {scenario.lift:.2f}x ({scenario.n_trades} trades)")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)

    def format_full_report(
        self,
        results: Dict,
        format_type: str = 'console'
    ) -> str:
        """
        Generate full formatted report.

        Args:
            results: Complete analysis results
            format_type: Output format ('console', 'markdown', 'html')

        Returns:
            Formatted report string
        """
        sections = []

        # Summary
        sections.append(self.format_summary_text(results))

        # Correlation table
        if 'tier1' in results and 'point_biserial' in results['tier1']:
            sections.append(self.table_formatter.format_correlation_table(
                results['tier1']['point_biserial'],
                format_type
            ))

        # Regression table
        if 'tier2' in results and results['tier2'].get('logistic_regression'):
            sections.append(self.table_formatter.format_regression_table(
                results['tier2']['logistic_regression'],
                format_type
            ))

        # Scenario tables
        if 'scenarios' in results:
            if results['scenarios'].get('best_scenarios'):
                sections.append(self.table_formatter.format_scenario_table(
                    results['scenarios']['best_scenarios'],
                    'best',
                    format_type
                ))

            if results['scenarios'].get('worst_scenarios'):
                sections.append(self.table_formatter.format_scenario_table(
                    results['scenarios']['worst_scenarios'],
                    'worst',
                    format_type
                ))

        return "\n\n".join(sections)
