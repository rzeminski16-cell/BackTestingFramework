"""
Factor Analysis GUI Package.

This package provides three GUI applications for the Factor Analysis module:
1. Factor Analysis Dashboard - Main analysis interface and visualization
2. Configuration Manager - Profile creation and parameter configuration
3. Data Upload & Preparation - Trade log and factor data ingestion

Usage:
    # Launch the main dashboard
    from Classes.GUI.factor_analysis import FactorAnalysisDashboard
    app = FactorAnalysisDashboard()
    app.run()

    # Launch configuration manager standalone
    from Classes.GUI.factor_analysis import FactorConfigManagerGUI
    app = FactorConfigManagerGUI()
    app.run()

    # Launch data upload standalone
    from Classes.GUI.factor_analysis import FactorDataUploadGUI
    app = FactorDataUploadGUI()
    app.run()
"""

# Import components - with error handling for missing dependencies
try:
    from .components import (
        FactorListPanel,
        StatisticsPanel,
        DataQualityIndicator,
        AnalysisProgressPanel,
        ScenarioCard,
        TierResultsPanel,
        NavigationPanel,
        ProfileSelector,
        ConfigSection
    )
    _components_available = True
except ImportError as e:
    _components_available = False
    import warnings
    warnings.warn(f"Could not import factor analysis GUI components: {e}")

# Import main GUIs
try:
    from .config_manager import FactorConfigManagerGUI
    from .data_upload import FactorDataUploadGUI
    from .dashboard import FactorAnalysisDashboard
    _guis_available = True
except ImportError as e:
    _guis_available = False
    import warnings
    warnings.warn(f"Could not import factor analysis GUIs: {e}")

__all__ = []

if _components_available:
    __all__.extend([
        'FactorListPanel',
        'StatisticsPanel',
        'DataQualityIndicator',
        'AnalysisProgressPanel',
        'ScenarioCard',
        'TierResultsPanel',
        'NavigationPanel',
        'ProfileSelector',
        'ConfigSection',
    ])

if _guis_available:
    __all__.extend([
        'FactorConfigManagerGUI',
        'FactorDataUploadGUI',
        'FactorAnalysisDashboard',
    ])
