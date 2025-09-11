from .build import build_loader as _build_loader
from .build import load_constantmask as _load_constantmask
from .build import load_mean_std as _load_mean_std
from .build import load_daily_mean_std as _load_daily_mean_std

def build_loader(config):
    return _build_loader(config)

def load_constantmask(config):
    return _load_constantmask(config)

def load_mean_std(config):
    return _load_mean_std(config)

def load_daily_mean_std(config):
    return _load_daily_mean_std(config)