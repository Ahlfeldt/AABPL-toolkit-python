import builtins as _builtins
from importlib.metadata import version
from . import config
from . import main
from .main import (
    radius_search,
    radius_sum,
    radius_count,
    radius_mean,
    radius_variance,
    radius_std,
    radius_cv,
    radius_skewness,
    radius_kurtosis,
    detect_cluster_pts,
    detect_cluster_cells,
    detect_cluster_cells_from_labeled_pts,
)
from .search.grid_class import Grid
from .search.study_area import infer_study_area_from_pts, infer_sample_area_from_pts
from .search.null_distribution import draw_random_coords

__version__ = version('aabpl')
if not getattr(_builtins, '_aabpl_imported', False):
    print(f"aabpl v{__version__}.")
    _builtins._aabpl_imported = True
