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
    version="0.3.6.11",
    description='.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ahlfeldt/AABPL-toolkit-python',
    author='Gabriel M Ahlfeldt',
    author_email='g.ahlfeldt@hu-berlin.de',
    license='MIT',
    install_requires=['numpy','pandas','geopandas','shapely','matplotlib','pyproj','concave_hull','mpmath'],
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

        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
    ],
)
