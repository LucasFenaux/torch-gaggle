from dataclasses import dataclass


@dataclass
class GlobalConfigs:
    CACHE_DIR = "./.cache"  # Path to the local cache. This is where we persist hidden files.


global_configs = GlobalConfigs()
