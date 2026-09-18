"""配置加载。所有脚本共用，路径统一相对项目根目录解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "data.yaml"
MODEL_CONFIG = CONFIG_DIR / "model.yaml"


def abs_path(p: str | Path) -> Path:
    """相对路径按项目根目录解析；绝对路径原样返回。"""
    path = Path(p).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class Config(dict):
    """dict 的薄封装，额外提供点号取值和相对路径解析。"""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def path(self, dotted: str) -> Path:
        """把配置里的相对路径解析成绝对路径，例如 cfg.path("data.qlib_dir")。"""
        node: Any = self
        for part in dotted.split("."):
            node = node[part]
        p = Path(node).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    """加载 configs/data.yaml（数据 / 股票池 / 切分）。"""
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def load_model_config(path: str | Path | None = None) -> Config:
    """加载 configs/model.yaml（因子 / 模型 / walk-forward）。"""
    with open(path or MODEL_CONFIG, encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))
