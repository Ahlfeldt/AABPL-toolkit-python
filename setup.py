from setuptools import setup, find_packages

name = 'primelocations'

extra_test = ['pytest>=4', 'pytest-cov>=2',]
extra_dev = [*extra_test,'twine>=4.0.2',]
extra_ci = [*extra_test,'python-coveralls',]

with open('./README.md', 'r') as f:
    long_description = f.read()

setup(
    name=name,
    version="0.1.0",
    description='.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Ahlfeldt/primelocations',
    author='Gabriel M Ahlfeldt',
    author_email='g.ahlfeldt@hu-berlin.de',
    license='MIT',
    install_requires=['numpy','pandas','geopandas','shapely','matplotlib','pyproj'],
    packages=[name],
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
