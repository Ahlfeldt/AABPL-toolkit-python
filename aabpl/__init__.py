import builtins as _builtins
from importlib.metadata import version
from . import config
from . import main
from .main import (
    radius_search,
    radius_sum,
    radius_count,
    radius_mean,
    detect_cluster_pts,
    detect_cluster_cells,
)
from .radius_search.grid_class import Grid
from .radius_search.sample_area import infer_sample_area_from_pts
from .utils.grid_aggregate import aggregate_to_grid

__version__ = version('aabpl')
if not getattr(_builtins, '_aabpl_imported', False):
    print(f"aabpl v{__version__} is under active development.")
    _builtins._aabpl_imported = True
