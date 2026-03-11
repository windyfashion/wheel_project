"""Configuration manager for YAML-based configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class ConfigManager:
    """YAML configuration file manager with hot reload support.
    
    Example
    -------
    >>> config = ConfigManager("config")
    >>> robot_config = config.load("robot")
    >>> max_v = config.get("robot.max_v", default=1.5)
    """
    
    def __init__(self, config_dir: str | Path = "config"):
        """Initialize configuration manager.
        
        Parameters
        ----------
        config_dir : str or Path
            Directory containing YAML configuration files
        """
        self.config_dir = Path(config_dir)
        self._cache: dict[str, dict] = {}
        self._file_mtimes: dict[str, float] = {}
    
    def load(self, name: str, reload: bool = False) -> dict:
        """Load a configuration file.
        
        Parameters
        ----------
        name : str
            Configuration file name (without .yaml extension)
            Can include subdirectory, e.g., "controller/lqr"
        reload : bool
            Force reload from file even if cached
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        # Check if we need to reload
        config_path = self.config_dir / f"{name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        current_mtime = config_path.stat().st_mtime
        
        if reload or name not in self._cache or self._file_mtimes.get(name, 0) != current_mtime:
            with open(config_path, "r", encoding="utf-8") as f:
                self._cache[name] = yaml.safe_load(f) or {}
            self._file_mtimes[name] = current_mtime
        
        return self._cache[name]
    
    def reload(self, name: str) -> dict:
        """Force reload a configuration file.
        
        Parameters
        ----------
        name : str
            Configuration file name
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        return self.load(name, reload=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key.
        
        Parameters
        ----------
        key : str
            Dot-separated key, e.g., "robot.max_v"
        default : Any
            Default value if key not found
        
        Returns
        -------
        Any
            Configuration value or default
        """
        parts = key.split(".")
        config_name = parts[0]
        
        try:
            config = self.load(config_name)
            value = config
            
            for part in parts[1:]:
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    return default
            
            return value if value is not None else default
        
        except (FileNotFoundError, KeyError, AttributeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value in cache (does not write to file).
        
        Parameters
        ----------
        key : str
            Dot-separated key
        value : Any
            Value to set
        """
        parts = key.split(".")
        config_name = parts[0]
        
        if config_name not in self._cache:
            self._cache[config_name] = {}
        
        obj = self._cache[config_name]
        for part in parts[1:-1]:
            if part not in obj:
                obj[part] = {}
            obj = obj[part]
        
        obj[parts[-1]] = value
    
    def merge(self, base_name: str, override_name: str) -> dict:
        """Merge two configurations, with override taking precedence.
        
        Parameters
        ----------
        base_name : str
            Base configuration name
        override_name : str
            Override configuration name
        
        Returns
        -------
        dict
            Merged configuration
        """
        base = self.load(base_name).copy()
        override = self.load(override_name)
        
        def deep_merge(base_dict: dict, override_dict: dict) -> dict:
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_merge(base_dict[key], value)
                else:
                    base_dict[key] = value
            return base_dict
        
        return deep_merge(base, override)
    
    def save(self, name: str, config: dict) -> None:
        """Save configuration to YAML file.
        
        Parameters
        ----------
        name : str
            Configuration file name
        config : dict
            Configuration dictionary to save
        """
        config_path = self.config_dir / f"{name}.yaml"
        
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        self._cache[name] = config
        self._file_mtimes[name] = config_path.stat().st_mtime
