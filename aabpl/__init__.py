from importlib.metadata import version
# TODO replace * with more expliced exports for the final version of the package 
# # from .radius_search import *
# import utils
# import testing
# import illustrations
# import radius_search
from . import main
# from .main import *
# from .radius_search.optimal_grid_spacing import *
# from .radius_search.offset_regions import *
# from ..utils.general import *
# from ..illustrations.illustrate_nested_grid import *
# from ..illustrations.illustrate_optimal_grid_spacing import *
# from ..illustrations.illustrate_point_to_cell_region_assignment import *
# # from .nested_search import *
# from .radius_search.radius_search_class import *
# from ..illustrations.plot_utils import *
# from .radius_search.two_dimensional_weak_ordering_class import *
# from .main import *
# from ..illustrations.plot_utils import *
# TODO ensure to no exports imports (like np.array)?

__version__ = version('primelocations')
