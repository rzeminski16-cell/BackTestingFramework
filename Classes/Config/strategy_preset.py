"""
Strategy Parameter Preset Management

Allows users to save and load different parameter combinations for strategies.
Presets are stored as JSON files in the config/strategy_presets directory.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


class StrategyParameterPreset:
    """
    Manages saving and loading of strategy parameter presets.

    Each preset includes:
    - Strategy name
    - Parameter values
    - Preset name
    - Description (optional)
    - Created/modified timestamps
    """

    def __init__(self, presets_dir: str = "config/strategy_presets"):
        """
        Initialize preset manager.

        Args:
            presets_dir: Directory to store preset JSON files
        """
        self.presets_dir = Path(presets_dir)
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def save_preset(
        self,
        strategy_name: str,
        preset_name: str,
        parameters: Dict[str, Any],
        description: str = ""
    ) -> Path:
        """
        Save a parameter preset to disk.

        Args:
            strategy_name: Name of the strategy (e.g., "AlphaTrendStrategy")
            preset_name: Name for this preset (e.g., "Aggressive", "Conservative")
            parameters: Dictionary of parameter names and values
            description: Optional description of the preset

        Returns:
            Path to the saved preset file
        """
        preset_data = {
            "strategy_name": strategy_name,
            "preset_name": preset_name,
            "description": description,
            "parameters": parameters,
            "created_at": datetime.now().isoformat(),
            "modified_at": datetime.now().isoformat()
        }

        # Create filename: strategy_name__preset_name.json
        filename = f"{strategy_name}__{preset_name}.json"
        filepath = self.presets_dir / filename

        # If file exists, preserve creation time
        if filepath.exists():
            try:
                existing_data = json.loads(filepath.read_text())
                preset_data["created_at"] = existing_data.get("created_at", preset_data["created_at"])
            except (json.JSONDecodeError, IOError):
                pass

        # Save to file
        with open(filepath, 'w') as f:
            json.dump(preset_data, f, indent=2)

        return filepath

    def load_preset(self, strategy_name: str, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a parameter preset from disk.

        Args:
            strategy_name: Name of the strategy
            preset_name: Name of the preset to load

        Returns:
            Dictionary containing preset data, or None if not found
        """
        filename = f"{strategy_name}__{preset_name}.json"
        filepath = self.presets_dir / filename

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def list_presets(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all available presets, optionally filtered by strategy.

        Args:
            strategy_name: If provided, only return presets for this strategy

        Returns:
            List of preset metadata dictionaries
        """
        presets = []

        for filepath in self.presets_dir.glob("*.json"):
            try:
                with open(filepath, 'r') as f:
                    preset_data = json.load(f)

                # Filter by strategy if specified
                if strategy_name and preset_data.get("strategy_name") != strategy_name:
                    continue

                # Add file path to metadata
                preset_data["filepath"] = str(filepath)
                presets.append(preset_data)
            except (json.JSONDecodeError, IOError):
                continue

        # Sort by strategy name, then preset name
        presets.sort(key=lambda x: (x.get("strategy_name", ""), x.get("preset_name", "")))
        return presets

    def delete_preset(self, strategy_name: str, preset_name: str) -> bool:
        """
        Delete a parameter preset.

        Args:
            strategy_name: Name of the strategy
            preset_name: Name of the preset to delete

        Returns:
            True if deleted successfully, False if not found
        """
        filename = f"{strategy_name}__{preset_name}.json"
        filepath = self.presets_dir / filename

        if filepath.exists():
            filepath.unlink()
            return True
        return False

    def get_preset_parameters(self, strategy_name: str, preset_name: str) -> Optional[Dict[str, Any]]:
        """
        Get just the parameters from a preset (convenience method).

        Args:
            strategy_name: Name of the strategy
            preset_name: Name of the preset

        Returns:
            Dictionary of parameters, or None if preset not found
        """
        preset = self.load_preset(strategy_name, preset_name)
        if preset:
            return preset.get("parameters")
        return None

    def rename_preset(
        self,
        strategy_name: str,
        old_preset_name: str,
        new_preset_name: str
    ) -> bool:
        """
        Rename an existing preset.

        Args:
            strategy_name: Name of the strategy
            old_preset_name: Current name of the preset
            new_preset_name: New name for the preset

        Returns:
            True if renamed successfully, False otherwise
        """
        preset = self.load_preset(strategy_name, old_preset_name)
        if not preset:
            return False

        # Update preset name
        preset["preset_name"] = new_preset_name
        preset["modified_at"] = datetime.now().isoformat()

        # Save with new name
        self.save_preset(
            strategy_name,
            new_preset_name,
            preset["parameters"],
            preset.get("description", "")
        )

        # Delete old file
        self.delete_preset(strategy_name, old_preset_name)

        return True

    def export_preset(self, strategy_name: str, preset_name: str, export_path: Path) -> bool:
        """
        Export a preset to a specific file path.

        Args:
            strategy_name: Name of the strategy
            preset_name: Name of the preset
            export_path: Path where to export the preset

        Returns:
            True if exported successfully, False otherwise
        """
        preset = self.load_preset(strategy_name, preset_name)
        if not preset:
            return False

        try:
            with open(export_path, 'w') as f:
                json.dump(preset, f, indent=2)
            return True
        except IOError:
            return False

    def import_preset(self, import_path: Path) -> Optional[Dict[str, Any]]:
        """
        Import a preset from a file path.

        Args:
            import_path: Path to the preset file to import

        Returns:
            The imported preset data, or None if import failed
        """
        try:
            with open(import_path, 'r') as f:
                preset_data = json.load(f)

            # Validate required fields
            required_fields = ["strategy_name", "preset_name", "parameters"]
            if not all(field in preset_data for field in required_fields):
                return None

            # Save to presets directory
            self.save_preset(
                preset_data["strategy_name"],
                preset_data["preset_name"],
                preset_data["parameters"],
                preset_data.get("description", "")
            )

            return preset_data
        except (json.JSONDecodeError, IOError):
            return None
