"""
Backward-compatibility shim.  This package has been renamed to aabpl.search.

Re-exports the most commonly used public names so that code using the old
import path continues to work with a deprecation warning.
"""
import warnings
warnings.warn(
    "aabpl.radius_search has been renamed to aabpl.search. "
    "Update your imports to silence this warning.",
    DeprecationWarning,
    stacklevel=2,
)
from aabpl.search import *  # noqa: F401, F403
from aabpl.search.grid_class import Grid  # noqa: F401
from aabpl.search.point_assignment import assign_points_to_cells  # noqa: F401
from aabpl.search.spacing_topology import *  # noqa: F401, F403
