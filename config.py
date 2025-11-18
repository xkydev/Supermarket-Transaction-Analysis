"""
Configuraciones y constantes del proyecto.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paths:
    """Configuración de rutas del proyecto."""
    DATA_RAW: Path = Path("data/raw")
    DATA_PROCESSED: Path = Path("data/processed")
    MODELS: Path = Path("models")


@dataclass
class ClusteringConfig:
    """Configuración para el análisis de clustering."""
    MIN_CLUSTERS: int = 2
    MAX_CLUSTERS: int = 10
    RANDOM_STATE: int = 42
    N_INIT: int = 10


@dataclass
class RecommenderConfig:
    """Configuración para el sistema de recomendación."""
    TOP_N: int = 10
    MIN_SUPPORT: float = 0.01
    MIN_CONFIDENCE: float = 0.3
    MIN_LIFT: float = 1.0


@dataclass
class VisualizationConfig:
    """Configuración para visualizaciones."""
    PLOT_TEMPLATE: str = "plotly_white"
    HEIGHT: int = 500
    WIDTH: int = 800
