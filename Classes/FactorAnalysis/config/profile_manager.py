"""
Profile Manager for Factor Analysis configurations.

Handles loading, saving, and versioning of analysis profiles.
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .factor_config import FactorAnalysisConfig


class ProfileError(Exception):
    """Exception raised for profile-related errors."""
    pass


class ProfileManager:
    """
    Manages Factor Analysis configuration profiles.

    Provides functionality to:
    - Load profiles from YAML or JSON files
    - Save profiles with automatic versioning
    - List available profiles
    - Track profile history and changes

    Example:
        manager = ProfileManager(config_dir="./configs/factor_analysis")
        config = manager.load("momentum_value_2025")
        manager.save(config, "momentum_value_2025_v2")
    """

    SUPPORTED_EXTENSIONS = ['.yaml', '.yml', '.json']
    VERSION_HISTORY_DIR = ".versions"

    def __init__(self, config_dir: str = "./configs/factor_analysis"):
        """
        Initialize ProfileManager.

        Args:
            config_dir: Directory containing profile configurations
        """
        self.config_dir = Path(config_dir)
        self._ensure_directory_exists()

    def _ensure_directory_exists(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        version_dir = self.config_dir / self.VERSION_HISTORY_DIR
        version_dir.mkdir(exist_ok=True)

    def _find_profile_file(self, profile_name: str) -> Optional[Path]:
        """Find profile file with any supported extension."""
        for ext in self.SUPPORTED_EXTENSIONS:
            path = self.config_dir / f"{profile_name}{ext}"
            if path.exists():
                return path
        return None

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        if not YAML_AVAILABLE:
            raise ProfileError(
                "PyYAML is required to load YAML files. "
                "Install with: pip install pyyaml"
            )
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file."""
        with open(path, 'r') as f:
            return json.load(f)

    def _save_yaml(self, path: Path, data: Dict[str, Any]) -> None:
        """Save to YAML file."""
        if not YAML_AVAILABLE:
            raise ProfileError(
                "PyYAML is required to save YAML files. "
                "Install with: pip install pyyaml"
            )
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    def _save_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Save to JSON file."""
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, profile_name: str) -> FactorAnalysisConfig:
        """
        Load a profile by name.

        Args:
            profile_name: Name of the profile (without extension)

        Returns:
            FactorAnalysisConfig loaded from file

        Raises:
            ProfileError: If profile not found or invalid
        """
        path = self._find_profile_file(profile_name)

        if path is None:
            available = self.list_profiles()
            raise ProfileError(
                f"Profile '{profile_name}' not found. "
                f"Available profiles: {available}"
            )

        try:
            if path.suffix in ['.yaml', '.yml']:
                data = self._load_yaml(path)
            else:
                data = self._load_json(path)

            return FactorAnalysisConfig.from_dict(data)

        except Exception as e:
            raise ProfileError(f"Error loading profile '{profile_name}': {e}")

    def save(
        self,
        config: FactorAnalysisConfig,
        profile_name: Optional[str] = None,
        format: str = "yaml",
        create_version_backup: bool = True
    ) -> Path:
        """
        Save a profile to file.

        Args:
            config: Configuration to save
            profile_name: Name for the profile (uses config.profile_name if None)
            format: Output format ('yaml' or 'json')
            create_version_backup: Whether to backup existing version

        Returns:
            Path to saved file
        """
        name = profile_name or config.profile_name

        if format not in ['yaml', 'json']:
            raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'")

        ext = '.yaml' if format == 'yaml' else '.json'
        path = self.config_dir / f"{name}{ext}"

        # Create version backup if file exists
        if create_version_backup and path.exists():
            self._create_version_backup(path, name)

        # Update timestamp and save
        updated_config = config.with_updated_timestamp()
        data = updated_config.to_dict()

        if format == 'yaml':
            self._save_yaml(path, data)
        else:
            self._save_json(path, data)

        return path

    def _create_version_backup(self, path: Path, profile_name: str) -> None:
        """Create a versioned backup of existing profile."""
        version_dir = self.config_dir / self.VERSION_HISTORY_DIR
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{profile_name}_{timestamp}{path.suffix}"
        backup_path = version_dir / backup_name
        shutil.copy2(path, backup_path)

    def list_profiles(self) -> List[str]:
        """
        List all available profiles.

        Returns:
            List of profile names (without extensions)
        """
        profiles = set()
        for ext in self.SUPPORTED_EXTENSIONS:
            for path in self.config_dir.glob(f"*{ext}"):
                if path.stem != self.VERSION_HISTORY_DIR:
                    profiles.add(path.stem)
        return sorted(profiles)

    def exists(self, profile_name: str) -> bool:
        """Check if a profile exists."""
        return self._find_profile_file(profile_name) is not None

    def delete(self, profile_name: str, keep_versions: bool = True) -> None:
        """
        Delete a profile.

        Args:
            profile_name: Name of profile to delete
            keep_versions: If True, version history is preserved
        """
        path = self._find_profile_file(profile_name)
        if path is None:
            raise ProfileError(f"Profile '{profile_name}' not found")

        if keep_versions:
            self._create_version_backup(path, profile_name)

        path.unlink()

    def get_version_history(self, profile_name: str) -> List[Dict[str, Any]]:
        """
        Get version history for a profile.

        Returns:
            List of version info dicts with 'filename', 'timestamp', 'path'
        """
        version_dir = self.config_dir / self.VERSION_HISTORY_DIR
        versions = []

        for ext in self.SUPPORTED_EXTENSIONS:
            pattern = f"{profile_name}_*{ext}"
            for path in version_dir.glob(pattern):
                # Extract timestamp from filename
                parts = path.stem.rsplit('_', 2)
                if len(parts) >= 3:
                    date_str = parts[-2]
                    time_str = parts[-1]
                    try:
                        timestamp = datetime.strptime(
                            f"{date_str}_{time_str}",
                            "%Y%m%d_%H%M%S"
                        )
                        versions.append({
                            'filename': path.name,
                            'timestamp': timestamp,
                            'path': str(path)
                        })
                    except ValueError:
                        pass

        return sorted(versions, key=lambda x: x['timestamp'], reverse=True)

    def restore_version(self, profile_name: str, version_filename: str) -> FactorAnalysisConfig:
        """
        Restore a specific version of a profile.

        Args:
            profile_name: Name of the profile
            version_filename: Filename of version to restore

        Returns:
            The restored configuration
        """
        version_path = self.config_dir / self.VERSION_HISTORY_DIR / version_filename

        if not version_path.exists():
            raise ProfileError(f"Version file not found: {version_filename}")

        try:
            if version_path.suffix in ['.yaml', '.yml']:
                data = self._load_yaml(version_path)
            else:
                data = self._load_json(version_path)

            config = FactorAnalysisConfig.from_dict(data)

            # Save as current version
            self.save(config, profile_name)

            return config

        except Exception as e:
            raise ProfileError(f"Error restoring version: {e}")

    def duplicate(self, source_name: str, target_name: str) -> FactorAnalysisConfig:
        """
        Duplicate a profile with a new name.

        Args:
            source_name: Name of profile to duplicate
            target_name: Name for the new profile

        Returns:
            The duplicated configuration
        """
        config = self.load(source_name)

        # Update the profile name in config
        config_dict = config.to_dict()
        config_dict['profile_name'] = target_name
        config_dict['created_date'] = datetime.now().strftime("%Y-%m-%d")
        config_dict['last_modified'] = datetime.now().strftime("%Y-%m-%d")

        new_config = FactorAnalysisConfig.from_dict(config_dict)
        self.save(new_config, target_name, create_version_backup=False)

        return new_config

    def compare_profiles(self, name1: str, name2: str) -> Dict[str, Any]:
        """
        Compare two profiles and return differences.

        Args:
            name1: First profile name
            name2: Second profile name

        Returns:
            Dictionary with 'same', 'different', and 'details' keys
        """
        config1 = self.load(name1)
        config2 = self.load(name2)

        dict1 = config1.to_dict()
        dict2 = config2.to_dict()

        def compare_dicts(d1: Dict, d2: Dict, path: str = "") -> Dict[str, Any]:
            same = []
            different = []
            details = {}

            all_keys = set(d1.keys()) | set(d2.keys())

            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                v1 = d1.get(key)
                v2 = d2.get(key)

                if key not in d1:
                    different.append(current_path)
                    details[current_path] = {'profile1': None, 'profile2': v2}
                elif key not in d2:
                    different.append(current_path)
                    details[current_path] = {'profile1': v1, 'profile2': None}
                elif isinstance(v1, dict) and isinstance(v2, dict):
                    sub_result = compare_dicts(v1, v2, current_path)
                    same.extend(sub_result['same'])
                    different.extend(sub_result['different'])
                    details.update(sub_result['details'])
                elif v1 == v2:
                    same.append(current_path)
                else:
                    different.append(current_path)
                    details[current_path] = {'profile1': v1, 'profile2': v2}

            return {'same': same, 'different': different, 'details': details}

        return compare_dicts(dict1, dict2)

    def create_example_profile(self) -> Path:
        """
        Create an example profile with documentation.

        Returns:
            Path to created example file
        """
        config = FactorAnalysisConfig.create_default(
            profile_name="example_momentum_value",
            strategy_name="momentum_value_mix"
        )

        # Add a descriptive description
        config_dict = config.to_dict()
        config_dict['description'] = (
            "Example factor analysis profile for momentum+value strategy. "
            "Analyzes technical indicators, fundamental ratios, insider activity, "
            "and options data to identify best/worst trading scenarios."
        )

        config = FactorAnalysisConfig.from_dict(config_dict)
        return self.save(config, "example_momentum_value", format="yaml")
