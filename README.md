# AABPL-toolkit-python
(c) Gabriel M. Ahlfeldt, Thilo N. H. Albers, Kristian Behrens Version 0.1.0, 2024-10


## About
Python version of the prime locations delineation algorithm

## Installation
To install the Python package of the ABRSQOL-toolkit, run the following command in your python environment in your terminal. 

```console
pip install primelocations
```
Alternatively you can also install it from within your python script:
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'primelocations'])
```
<details>
<summary>In case an error occurs at the installation...</summary>

with an erorr message like 'metadata-generation-failed', it is likely caused by incompatabile versions of `setuptools` and `packaging`. 
This can be fixed by upgrading `setuptools` and `packaging` to compatible versions:
```console
pip install --upgrade setuptools>=74.1.1
pip install --upgrade packaging>=22.0
```
Or by downgrading `setuptools`:
```console
pip install --upgrade setuptools==70.0.0
```

</details>



## Usage
You may then load the package by running:
```python
import primelocations
```
Or if you prefer alternatively import the function and testdata explicitly:
```python
from primelocations import *
```

### Examples
#### Example 1: load testdata, run QoL inversion with default parameters, store result as 'QoL' variable, view result
```python
testdata = ABRSQOL.testdata
testdata['QoL'] = ABRSQOL.invert_quality_of_life(df=testdata)
testdata.head()
```

#### Example 2: load your data from csv, run inversion, save result as csv
```python
my_dataframe = read.csv("path/to/your/csv_filename.csv")
my_dataframe['quality_of_life'] = ABRSQOL.invert_quality_of_life(
  # supply your dataset as a dataframe
  df=my_dataframe,
  # specify the corresponding variable name for your dataset
  w = 'wage',
  p_H = 'floor_space_price',
  P_t = 'tradable_goods_price',
  p_n = 'local_services_price',
  L = 'residence_pop',
  L_b = 'L_b',
  # freely adjust remaining parameters
  alpha = 0.7,
  beta = 0.5,
  gamma = 3,
  xi = 5.5,
  conv = 0.3,
  tolerance = 1e-11,
  maxiter = 50000
)
# Write output to target folder (just replace the path)
from pandas import write_csv
write_csv(my_dataframe, 'C:/FOLDER/my_data_with_qol.csv')
```

#### Example 3: Reference variables in your dataset by using the column index
```python
my_dataframe['QoL'] = ABRSQOL.invert_quality_of_life(
  df=my_dataframe,
  w = 1,
  p_H = 3,
  P_t = 4,
  p_n = 2,
  L = 6,
  L_b = 5
)
```

#### Example 4: Having named the variables in your data according to the default parameters, you can omit specifying variable names
```python
my_dataframe['QoL'] = ABRSQOL.invert_quality_of_life(
  df=my_dataframe,
  alpha = 0.7,
  beta = 0.5,
  gamma = 3,
  xi = 5.5,
  conv = 0.5
)
```

### Ready-to-use script

If you are new to Python, you may find it useful to execute the [`Example.py`](https://github.com/Ahlfeldt/ABRSQOL-toolkit-python/blob/main/Example.py) (or [`Example.ipynb`](https://github.com/Ahlfeldt/ABRSQOL-toolkit-python/blob/main/Example.ipynb)) script saved in this folder. It will install the package, load the testing data set, generate a quality-of-life index, and save it to your working directory.  It should be straightforward to adapt the script to your data and preferred parameter values.

## Folder Structure and Files (OUTDATED)

Folder | Name  | Description |
|:------------------------|:-----------------------|:----------------------------------------------------------------------------------|
| primelocations | `main.py` | Contains main functions for user: radius_search and detect_clusters   |
| primelocations | `disk_search.py` | Performs radius search in multiple steps: 
(1): 
    (a) Assigns each point to a grid cell. 
    (b) store pt ids for search target in grid cells and precalucaltes sums per grid cell. 
(2): divides cell into cell regions that define which of the surrounding cells are fully included in search radius and which cell are partly overapped by search radius. It assign each point to such a relative search region avoiding unnecessary checks on cells (through methods from 2d_weak_ordering). 
(3): loops over all search source points sorted based on cell id and cell region and 
    (a) sums precalculated sums of non empty cells that are fully within cell radius (or reuses this sum from last source point if same cells were relevant). 
    (b) retrieves all search target points from partly overlapped cells (or reuses them from last source point if same cell were relevant).
    (c) checks bilateral distance from source point to target points and sums values for target points within search radius |
| primelocations | `grid_class.py` | Mostly implemented. Creates class for Grid  |
| primelocations | `2d_weak_ordering.py` | Complex logic that helps to reduce the number of cells that need to be checked if they overlap with the search radius. Relative to origing cell (0,0) it creates a hierarchical weak ordering for surrounding cells. E.g. cell(1,1) is always closer to any point within cell(0,0) than cell(2,2). But its unclear whether a point within cell(0,0) is closer to cell(1,0) or to cell(0,1) |
| primelocations | `random_distribution.py` | functions to draw random points and optain cutoff value for k-th percentile |
| primelocations | `valid_area.py` | Not implemented yet. Will include functions to allow the user to provide a (in)valid area by either providing a geometry or provide a list of (in)valid of cell ids  |
| primelocations | `distances_to_cell.py` | Includes helper functions to calculate smallest/largest distance from a cell to (1) another cell, (2) to a triangle, (3) to points. Also contains other functions related to cell distance checks. |
| primelocations | `general.py` | Contains small helper functions unrelated to radius search methods |
| primelocations/illustrations | `*.py*` | Contains method for illustrating methods (mainly used for testing but can remain in final version to allow user to illustrate the algorithm) |
| plots | `opt_grid` | Created in first days of project to get a feeling for the importance of the relative size of the grid spacing with respect to the search radius |
| primelocations | `optimal_grid_spacing` | Not fully implemented. Automatically choose optimal grid spacing to execute radius search as fast as possible |
| primelocations | `nested_search` | Not fully implemented. Nesting grid cell improves scaling for vary dense (relative to search radius) point data sets. |
| primelocations/documentation | `docstring.py` | Not implemented yet. Will include repetitive help text for functions |


## Acknowledgements
The Python version of the toolkit was developed by [Max von Mylius](https://github.com/maximylius) based on the Stata and MATLAB functions.
