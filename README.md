# !!! This site is under construction !!!

# AABPL-toolkit-python
(c) Gabriel M. Ahlfeldt, Thilo N. H. Albers, Kristian Behrens, [Max von Mylius](https://github.com/maximylius), Version 0.1.0, 2024-10


## About
This repository is part of the **[Toolkit of Prime Locations (AABPL)](https://github.com/Ahlfeldt/AABPL-toolkit/blob/main/README.md)**. It contains a Python version of the prime locations delineation algorithm developed by Ahlfeldt, Albers, and Behrens (2024). It is designed to be more readily accessible than the C++/Stata hybrid version used by Ahlfeldt, Albers, and Behrens (2024). The algorithm uses arbitrary spatial point patterns as input and returns a gridded version of the data along with polygons of the delineated spatial clusters as outputs.

Note that while this implementation of the algorithm follows the same basic steps as the one used by Ahlfeldt, Albers, and Behrens (2024), it will not necessarily generate exactly the same results. The Python package is designed to enhance usability. There are subtle differences in the way counterfactual distributions are generated, establishments are assigned to grid cells, clusters are aggregated, and convex hulls are generated. Importantly, the current version of the algorithm samples from a bounding box built around the establishments input into the algorithm, whereas Ahlfeldt, Albers, and Behrens (2024) condition on the presence of employment. Therefore, the parameter values that need to be defined in the program syntax cannot be directly transferred from Ahlfeldt, Albers, and Behrens (2024). 

We recommend that users find their own preferred values depending on the context and purpose of the clustering. We aim to allow for a user-specified sampling area so that users can, akin to Ahlfeldt, Albers, and Behrens (2024), exclude arbitrary areas when generating counterfactual establishment distributions. For replication of the results reported in Ahlfeldt, Albers, and Behrens (2024), we refer to the official replication directory.
 
When using the algorithm in your work, **please cite**: 

Ahlfeldt, Albers, Behrens (2024): Prime locations. American Economic Review: Insights, forthcoming.

## Installation
To install the Python package of the ABRSQOL-toolkit, run the following command in your python environment in your terminal. 

pip install aabpl

Alternatively you can also install it from within your python script:
```python
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", 'aabpl'])
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
import aabpl
```
Or if you prefer alternatively import the function and testdata explicitly:
```python
# imports 
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt 
from aabpl.main import detect_clusters, radius_search, convert_coords_to_local_crs
```

### Program syntax

Explain the syntax with its arguments here

### Examples
#### Example 1:
```python
path_to_your_csv = '../../cbsa_sample_data/plants_10180.txt'
crs_of_your_csv =  "EPSG:4326"

pts_df = read_csv(path_to_your_csv, sep=",", header=None)#[200:20000]
pts_df.columns = ["eid", "employment", "industry", "lat","lon","moved"]
convert_coords_to_local_crs(pts_df)

detect_clusters(
    pts_df=pts_df,
    radius=750,
    include_boundary=False,
    exclude_pt_itself=True,

    k_th_percentiles=[99.5],
    n_random_points=int(1e6),
    random_seed=0,

    sum_names=['employment'],
    plot_distribution={},
    plot_cluster_points={},
    silent = True,
)
```


### Ready-to-use script

If you are new to Python, you may find it useful to execute the [`Example.py`](https://github.com/Ahlfeldt/ABRSQOL-toolkit-python/blob/main/Example.py) (or [`Example.ipynb`](https://github.com/Ahlfeldt/ABRSQOL-toolkit-python/blob/main/Example.ipynb)) script saved in this folder. It will install the package, load the testing data set, generate a quality-of-life index, and save it to your working directory.  It should be straightforward to adapt the script to your data and preferred parameter values.

### Inputs

The **compulsory input** into the algorithm is a file containing spatial point pattern data. In the application by Ahlfeldt, Albers, and Behrens (2024), spatial points are establishments. However, these could also be individuals, buildings, or any other subjects or objects whose location can be referenced by geographic coordinates. The data file should contain geographic coordinates in standard decimal degrees and a variable that defines the importance of a subject or object. In the application by Ahlfeldt, Albers, and Behrens (2024), the importance is represented by the employment of an establishment. However, it could also be the productivity of a worker, the height of a building, or any weight that summarizes the importance of a data point. Of course, equal importance will be reflected by a uniform value.

In case you wish to use the above `Example.py` script without having to make any adjustments (except for setting your root directory), you should create a comma-separated file with exactly the same name and structure as the `plants.txt` file provided in this repository. Note that this exemplary input file **does not** include variable names. It includes variables in the following order (separated by commas):

- **identifier variable**: In our case, this is an establishment identifier. If you do not need this, you can set all values to 1.
- **importance weight**: In our case, this is predicted employment. If you want to use equal weights, you can set all values to 1.
- **category identifier**: In our case, this is the type of establishment (e.g., accounting, consulting, etc.). If you do not care, you can set all values to 1.
- **latitude**: Given in decimal degrees in the standard WGS1984 geographic coordinate system.
- **longitude**: Given in decimal degrees in the standard WGS1984 geographic coordinate system.

Variable names will then be assigned by the script. Of course, you can also import data sets that already contain variable names. Just make sure that latitudes and longitudes are defined by variables named `lat` and `lon`. You can define the name of the variable representing your importance weights in the program syntax.

For future versions of the package, we aim to allow for a shapefile that defines the sampling area of the counterfactual distribution as an **optional input**. This shapefile must be projected within the WGS1984 geographic coordinate system. Ahlfeldt, Albers, and Behrens (2024) exclude residential and undevelopable areas. Such a shapefile could also restrict the sampling area for counterfactual spatial distributions to inhabitable areas or to areas zoned for the development of tall buildings.

## Folder Structure and Files (OUTDATED)

Folder | Name  | Description |
|:------------------------|:-----------------------|:----------------------------------------------------------------------------------|
| aabpl | `main.py` | Contains main functions for user: radius_search and detect_clusters   |
| aabpl | `disk_search.py` | Performs radius search in multiple steps: 
(1): 
    (a) Assigns each point to a grid cell. 
    (b) store pt ids for search target in grid cells and precalucaltes sums per grid cell. 
(2): divides cell into cell regions that define which of the surrounding cells are fully included in search radius and which cell are partly overapped by search radius. It assign each point to such a relative search region avoiding unnecessary checks on cells (through methods from 2d_weak_ordering). 
(3): loops over all search source points sorted based on cell id and cell region and 
    (a) sums precalculated sums of non empty cells that are fully within cell radius (or reuses this sum from last source point if same cells were relevant). 
    (b) retrieves all search target points from partly overlapped cells (or reuses them from last source point if same cell were relevant).
    (c) checks bilateral distance from source point to target points and sums values for target points within search radius |
| aabpl | `grid_class.py` | Mostly implemented. Creates class for Grid  |
| aabpl | `2d_weak_ordering.py` | Complex logic that helps to reduce the number of cells that need to be checked if they overlap with the search radius. Relative to origing cell (0,0) it creates a hierarchical weak ordering for surrounding cells. E.g. cell(1,1) is always closer to any point within cell(0,0) than cell(2,2). But its unclear whether a point within cell(0,0) is closer to cell(1,0) or to cell(0,1) |
| aabpl | `random_distribution.py` | functions to draw random points and optain cutoff value for k-th percentile |
| aabpl | `valid_area.py` | Not implemented yet. Will include functions to allow the user to provide a (in)valid area by either providing a geometry or provide a list of (in)valid of cell ids  |
| aabpl | `distances_to_cell.py` | Includes helper functions to calculate smallest/largest distance from a cell to (1) another cell, (2) to a triangle, (3) to points. Also contains other functions related to cell distance checks. |
| aabpl | `general.py` | Contains small helper functions unrelated to radius search methods |
| aabpl/illustrations | `*.py*` | Contains method for illustrating methods (mainly used for testing but can remain in final version to allow user to illustrate the algorithm) |
| plots | `opt_grid` | Created in first days of project to get a feeling for the importance of the relative size of the grid spacing with respect to the search radius |
| aabpl | `optimal_grid_spacing` | Not fully implemented. Automatically choose optimal grid spacing to execute radius search as fast as possible |
| aabpl | `nested_search` | Not fully implemented. Nesting grid cell improves scaling for vary dense (relative to search radius) point data sets. |
| aabpl/documentation | `docstring.py` | Not implemented yet. Will include repetitive help text for functions |

### Selected Files

Folder | Name  | Description |
|:-------------------|:-------------------------------------|:-------------------------------------------------------------------------|
| [-](https://github.com/Ahlfeldt/ABRSQOL-toolkit) | `AABPL-Codebook.pdf` | **Codebook** laying out the **structure of the deliniation algorithm in pseduo code** |

# References 

Ahlfeldt, Albers, Behrens (2024): Prime locations. American Economic Review: Insights, forthcoming.
