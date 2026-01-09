"""GUI components for the backtesting framework."""

from backtesting.gui.main_window import MainWindow
from backtesting.gui.optimization_gui import OptimizationWindow

__all__ = ["MainWindow", "OptimizationWindow"]


def launch():
    """Launch the main GUI application."""
    app = MainWindow()
    app.run()
