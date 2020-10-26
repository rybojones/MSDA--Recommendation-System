# KMeans and ALS

_kmeans.py_ and _als.py_ are python executables that read in .dat files containing information on movie ratings and runs ML logic to generate recommendation systems.

## Installation

You need to have python version 3.8.5 installed, as well as pyspark version 3.0.0 and matplotlib.

```bash
pip install python=3.8.5 matplotlib=3.3.2
```
or, if using miniconda.

```bash
conda install python=3.8.5 matplotlib=3.3.2
```

## Usage

Navigate to the folder with the _kmeans.py_ script and type the following into the bash terminal (ensure that _movies.dat_, _ratings.dat_ and _users.dat_ are in the same location as _kmeans.py_.):

```bash
/spark/bin/spark-submit kmeans.py
```

The program will run with only one line of output at the end. KMeans clustering will be performed and a ratings matrix will produced for use in a user-movie recommendation system.

Navigate to the folder with the _als.py_ script and type the following into the bash terminal (ensure that _movies.dat_ and _ratings.dat_ are in the same location as _als.py_.):

```bash
/spark/bin/spark-submit --driver-memory 4g als.py
```

The program will run with only one line of output at the end. ALS regression will be performed for use in a user-movie recommendation system.

## Contributing
The following links contained helpful code that was adapted for the purposes of this program:
- https://github.com/databricks/spark-training/tree/master/data/movielens/medium

## License
None at this time.
