"""
Factor Analysis Dashboard GUI Launcher.

A modern CustomTkinter GUI for strategy performance factor analysis.

Features:
- Multi-view dashboard with navigation
- Data upload and preparation
- Configuration management
- Three-tier statistical analysis visualization
- Scenario analysis
- Export and reporting

Usage:
    python ctk_factor_analysis_gui.py [--config] [--upload]

Arguments:
    --config    Launch the Configuration Manager instead of Dashboard
    --upload    Launch the Data Upload interface instead of Dashboard
"""

import sys
import argparse


def main():
    """Main entry point for Factor Analysis GUI."""
    parser = argparse.ArgumentParser(
        description="Factor Analysis GUI for strategy performance analysis"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Launch the Configuration Manager"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Launch the Data Upload interface"
    )

    args = parser.parse_args()

    try:
        if args.config:
            from Classes.GUI.factor_analysis import FactorConfigManagerGUI
            print("Launching Factor Analysis Configuration Manager...")
            app = FactorConfigManagerGUI()
            app.run()
        elif args.upload:
            from Classes.GUI.factor_analysis import FactorDataUploadGUI
            print("Launching Factor Analysis Data Upload...")
            app = FactorDataUploadGUI()
            app.run()
        else:
            from Classes.GUI.factor_analysis import FactorAnalysisDashboard
            print("Launching Factor Analysis Dashboard...")
            app = FactorAnalysisDashboard()
            app.run()

    except ImportError as e:
        print(f"Error: Could not import GUI modules: {e}")
        print("\nMake sure customtkinter is installed:")
        print("  pip install customtkinter")
        sys.exit(1)
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
