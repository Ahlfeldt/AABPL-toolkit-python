from setuptools import setup, find_packages

name = 'aabpl'

extra_test = ['pytest>=4', 'pytest-cov>=2',]
extra_dev = [*extra_test,'twine>=4.0.2',]
extra_ci = [*extra_test,'python-coveralls',]

try:
    with open('./README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()
except (UnicodeDecodeError, FileNotFoundError):
    long_description = ''
setup(
    name=name,
    version="0.4.3",
    description=(
        'Fast radius search and spatial cluster detection for 2D point data '
        '— aggregate values within a fixed radius of every point.'
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ahlfeldt/AABPL-toolkit-python',
    author='Gabriel M Ahlfeldt',
    author_email='g.ahlfeldt@hu-berlin.de',
    license='MIT',
    keywords=[
        # The exact problems you solve
        'radius search', 'fixed-radius nearest neighbor', 'spatial aggregation', 
        'spatial clustering', 'spatial weights', 'buffer analysis', 'distance band',
        # The domain & technology
        'geospatial', 'gis', 'spatial analysis', 
        # Target audience / origin
        'spatial statistics', 'urban economics', 'economic geography',
        # Interoperability
        'geopandas', 'shapely' 
    ],
    install_requires=['numpy','pandas','geopandas','shapely','matplotlib','pyproj','concave_hull','mpmath'],
    python_requires='>=3.7',
    packages=find_packages(exclude=["tests*",]),
    extras_require={
        'test': extra_test,
        'dev': extra_dev,
        'ci': extra_ci,
    },
    entry_points={
        'console_scripts': [
        ],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'License :: OSI Approved :: MIT License',
    ],
)
