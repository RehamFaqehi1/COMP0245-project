"""Shared results root and directory names; join names to a results root."""

from pathlib import Path

RESULTS = Path(__file__).resolve().parent / 'results'
DATASETS = 'datasets'
EXPERIMENTS = 'experiments'
EVALUATIONS = 'evaluations'
FIGURES = 'figures'
METADATA = 'metadata'
HISTORICAL = Path(EVALUATIONS) / 'historical'
COLLECTIONS = Path(DATASETS) / 'collections'
