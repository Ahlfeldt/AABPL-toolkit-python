from importlib.metadata import version
# TODO replace * with more expliced exports for the final version of the package 
# # from .radius_search import *
# import utils
# import testing
# import radius_search
from . import main
# from .main import *
# from .radius_search.optimal_grid_spacing import *
# from .radius_search.offset_regions import *
# from .radius_search.radius_search_class import *

__version__ = version('aabpl')
print("Please note that the package 'aabpl' is under active development. Your currently using version "+str(__version__))
