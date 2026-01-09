"""
Excel Report Generator for Optimization Results.

Generates Excel files with:
- Summary tab with control values and configuration
- One tab per optimized parameter with line charts for each metric
"""

from typing import TYPE_CHECKING, List, Optional
import pandas as pd

if TYPE_CHECKING:
    from backtesting.optimization.optimizer import FullOptimizationResults

from backtesting.metrics.performance import AVAILABLE_METRICS


class ExcelReportGenerator:
    """
    Generate Excel reports with charts from optimization results.

    Creates a comprehensive Excel workbook with:
    1. Summary tab - Control values, configuration, and best values
    2. Parameter tabs - One per optimized parameter with:
       - Data table showing parameter values and metrics
       - Line charts for each metric showing how it varies with parameter

    Example:
        generator = ExcelReportGenerator(results)
        generator.generate("optimization_report.xlsx")
    """

    # Chart configuration
    CHART_WIDTH = 480
    CHART_HEIGHT = 300
    CHARTS_PER_ROW = 2
    CHART_START_ROW = 3  # Row offset from data table

    def __init__(self, results: "FullOptimizationResults"):
        """
        Initialize generator.

        Args:
            results: FullOptimizationResults from Optimizer.run()
        """
        self.results = results

    def generate(self, filepath: str) -> None:
        """
        Generate the Excel report.

        Args:
            filepath: Path to save the Excel file
        """
        import xlsxwriter

        workbook = xlsxwriter.Workbook(filepath)

        # Define formats
        formats = self._create_formats(workbook)

        # Create summary sheet
        self._create_summary_sheet(workbook, formats)

        # Create sheet for each parameter
        for param_name, param_results in self.results.parameter_results.items():
            self._create_parameter_sheet(workbook, param_name, param_results, formats)

        workbook.close()

    def _create_formats(self, workbook) -> dict:
        """Create cell formats for the workbook."""
        return {
            "title": workbook.add_format({
                "bold": True,
                "font_size": 14,
                "font_color": "#2c3e50",
            }),
            "header": workbook.add_format({
                "bold": True,
                "bg_color": "#3498db",
                "font_color": "white",
                "border": 1,
                "align": "center",
            }),
            "subheader": workbook.add_format({
                "bold": True,
                "bg_color": "#ecf0f1",
                "border": 1,
            }),
            "cell": workbook.add_format({
                "border": 1,
                "align": "center",
            }),
            "number": workbook.add_format({
                "border": 1,
                "align": "center",
                "num_format": "#,##0.00",
            }),
            "percent": workbook.add_format({
                "border": 1,
                "align": "center",
                "num_format": "0.00%",
            }),
            "integer": workbook.add_format({
                "border": 1,
                "align": "center",
                "num_format": "0",
            }),
            "highlight": workbook.add_format({
                "bold": True,
                "bg_color": "#27ae60",
                "font_color": "white",
                "border": 1,
                "align": "center",
            }),
        }

    def _create_summary_sheet(self, workbook, formats: dict) -> None:
        """Create the summary sheet with control values and configuration."""
        sheet = workbook.add_worksheet("Summary")

        # Set column widths
        sheet.set_column("A:A", 25)
        sheet.set_column("B:B", 20)
        sheet.set_column("C:Z", 15)

        row = 0

        # Title
        sheet.write(row, 0, "Optimization Results Summary", formats["title"])
        row += 2

        # Strategy info
        sheet.write(row, 0, "Strategy:", formats["subheader"])
        sheet.write(row, 1, self.results.config.strategy_class.name, formats["cell"])
        row += 1

        sheet.write(row, 0, "Run Date:", formats["subheader"])
        sheet.write(row, 1, self.results.timestamp.strftime("%Y-%m-%d %H:%M:%S"), formats["cell"])
        row += 2

        # Configuration section
        sheet.write(row, 0, "Configuration", formats["title"])
        row += 1

        sheet.write(row, 0, "Initial Capital:", formats["subheader"])
        sheet.write(row, 1, f"${self.results.config.initial_capital:,.2f}", formats["cell"])
        row += 1

        sheet.write(row, 0, "Commission:", formats["subheader"])
        sheet.write(row, 1, f"{self.results.config.commission * 100:.2f}%", formats["cell"])
        row += 1

        sheet.write(row, 0, "Slippage:", formats["subheader"])
        sheet.write(row, 1, f"{self.results.config.slippage * 100:.3f}%", formats["cell"])
        row += 2

        # Control values section
        sheet.write(row, 0, "Control Values", formats["title"])
        row += 1

        sheet.write(row, 0, "Parameter", formats["header"])
        sheet.write(row, 1, "Control Value", formats["header"])
        sheet.write(row, 2, "Description", formats["header"])
        row += 1

        param_defs = self.results.config.strategy_class.get_parameter_definitions()
        for param_name, value in self.results.control_values.items():
            param_def = param_defs.get(param_name)
            description = param_def.description if param_def else ""

            sheet.write(row, 0, param_name, formats["cell"])
            sheet.write(row, 1, str(value), formats["cell"])
            sheet.write(row, 2, description, formats["cell"])
            row += 1

        row += 1

        # Parameters being optimized
        sheet.write(row, 0, "Parameters Optimized", formats["title"])
        row += 1

        sheet.write(row, 0, "Parameter", formats["header"])
        sheet.write(row, 1, "Values Tested", formats["header"])
        sheet.write(row, 2, "Range", formats["header"])
        row += 1

        for param_name, param_results in self.results.parameter_results.items():
            values = param_results.values_tested
            value_range = f"{min(values)} - {max(values)}" if values else "N/A"

            sheet.write(row, 0, param_name, formats["cell"])
            sheet.write(row, 1, len(values), formats["integer"])
            sheet.write(row, 2, value_range, formats["cell"])
            row += 1

        row += 1

        # Metrics being tracked
        sheet.write(row, 0, "Metrics Tracked", formats["title"])
        row += 1

        sheet.write(row, 0, "Metric", formats["header"])
        sheet.write(row, 1, "Description", formats["header"])
        sheet.write(row, 2, "Higher is Better", formats["header"])
        row += 1

        for metric in self.results.config.metrics:
            metric_def = AVAILABLE_METRICS.get(metric)
            if metric_def:
                sheet.write(row, 0, metric_def.name, formats["cell"])
                sheet.write(row, 1, metric_def.description, formats["cell"])
                sheet.write(row, 2, "Yes" if metric_def.higher_is_better else "No", formats["cell"])
                row += 1

        row += 1

        # Best values summary
        sheet.write(row, 0, "Best Values Summary", formats["title"])
        row += 1

        # Header row
        sheet.write(row, 0, "Parameter", formats["header"])
        col = 1
        for metric in self.results.config.metrics:
            metric_def = AVAILABLE_METRICS.get(metric)
            name = metric_def.name if metric_def else metric
            sheet.write(row, col, f"Best for {name}", formats["header"])
            col += 1
        row += 1

        # Data rows
        for param_name, param_results in self.results.parameter_results.items():
            sheet.write(row, 0, param_name, formats["cell"])
            col = 1
            for metric in self.results.config.metrics:
                best_value = param_results.get_best_value(metric)
                sheet.write(row, col, str(best_value) if best_value is not None else "N/A", formats["highlight"])
                col += 1
            row += 1

    def _create_parameter_sheet(
        self,
        workbook,
        param_name: str,
        param_results,
        formats: dict,
    ) -> None:
        """Create a sheet for a single parameter with data and charts."""
        # Clean sheet name (Excel limits to 31 chars)
        sheet_name = param_name[:31]
        sheet = workbook.add_worksheet(sheet_name)

        # Set column widths
        sheet.set_column("A:A", 15)
        sheet.set_column("B:Z", 15)

        row = 0

        # Title
        param_def = param_results.parameter_definition
        title = f"Parameter: {param_name}"
        if param_def.description:
            title += f" - {param_def.description}"
        sheet.write(row, 0, title, formats["title"])
        row += 1

        sheet.write(row, 0, f"Control Value: {param_results.control_value}", formats["subheader"])
        row += 2

        # Write data table
        data_start_row = row

        # Header row
        sheet.write(row, 0, param_name, formats["header"])
        for col, metric in enumerate(self.results.config.metrics, 1):
            metric_def = AVAILABLE_METRICS.get(metric)
            name = metric_def.name if metric_def else metric
            sheet.write(row, col, name, formats["header"])
        row += 1

        # Data rows
        data_rows = []
        for result in param_results.results:
            data_row = [result.parameter_value]
            for metric in self.results.config.metrics:
                value = result.metrics.get(metric, float("nan"))
                data_row.append(value)
            data_rows.append(data_row)

            # Write to sheet
            sheet.write(row, 0, result.parameter_value, formats["cell"])
            for col, metric in enumerate(self.results.config.metrics, 1):
                value = result.metrics.get(metric, float("nan"))
                if pd.notna(value):
                    sheet.write(row, col, value, formats["number"])
                else:
                    sheet.write(row, col, "N/A", formats["cell"])
            row += 1

        data_end_row = row - 1

        # Add charts below the data
        row += 2
        chart_row = row

        num_metrics = len(self.results.config.metrics)
        num_values = len(param_results.values_tested)

        for i, metric in enumerate(self.results.config.metrics):
            metric_def = AVAILABLE_METRICS.get(metric)
            metric_name = metric_def.name if metric_def else metric

            # Create line chart
            chart = workbook.add_chart({"type": "line"})

            # Add data series
            chart.add_series({
                "name": metric_name,
                "categories": [sheet_name, data_start_row + 1, 0, data_end_row, 0],
                "values": [sheet_name, data_start_row + 1, i + 1, data_end_row, i + 1],
                "line": {"width": 2.5},
                "marker": {"type": "circle", "size": 6},
            })

            # Configure chart
            chart.set_title({"name": f"{metric_name} vs {param_name}"})
            chart.set_x_axis({
                "name": param_name,
                "major_gridlines": {"visible": True, "line": {"color": "#D9D9D9"}},
            })
            chart.set_y_axis({
                "name": metric_name,
                "major_gridlines": {"visible": True, "line": {"color": "#D9D9D9"}},
            })
            chart.set_legend({"position": "bottom"})
            chart.set_size({"width": self.CHART_WIDTH, "height": self.CHART_HEIGHT})

            # Position chart
            chart_col = (i % self.CHARTS_PER_ROW) * 8
            chart_row_offset = (i // self.CHARTS_PER_ROW) * 15

            sheet.insert_chart(chart_row + chart_row_offset, chart_col, chart)

        # Add a summary row showing best values
        summary_row = data_end_row + 2 + (((num_metrics - 1) // self.CHARTS_PER_ROW) + 1) * 15

        sheet.write(summary_row, 0, "Best Values:", formats["title"])
        summary_row += 1

        for metric in self.results.config.metrics:
            metric_def = AVAILABLE_METRICS.get(metric)
            metric_name = metric_def.name if metric_def else metric
            best_value = param_results.get_best_value(metric)

            sheet.write(summary_row, 0, metric_name, formats["subheader"])
            sheet.write(summary_row, 1, str(best_value) if best_value is not None else "N/A", formats["highlight"])
            summary_row += 1


def generate_optimization_report(
    results: "FullOptimizationResults",
    filepath: str,
) -> None:
    """
    Convenience function to generate an Excel report.

    Args:
        results: FullOptimizationResults from Optimizer.run()
        filepath: Path to save the Excel file
    """
    generator = ExcelReportGenerator(results)
    generator.generate(filepath)
