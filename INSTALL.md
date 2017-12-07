# Instacart Market Basket Analysis
Team ZLABS

## Prerequisites

- Python 3.0+
- numpy
- Pandas
- LightGBM
- XGBoost

if you have the above packages installed, please ignore this guide
## Installing

A step by step series of examples that tell you have to get a development env running

> This guide assumes you have python 3.0+ and is written for Linux/macOS ONLY (sorry Windows).

If you have [Anaconda ](https://www.anaconda.com/what-is-anaconda/)(python 3.6 verision), please skip the steps for installing [Numpy ](https://pandas.pydata.org) and [Pandas ](https://www.anaconda.com/what-is-anaconda/)

##### Numpy
> Numpy comes in many third-party and operating system vendor packages. You should have it already installed in If you don't have it, we recommend installing Numpy with [Pypl](https://pypi.python.org/pypi/numpy)


##### Pandas
We recommend installing Pandas using Anaconda. But if you don't want to, you can install it using pip:
```
pip install pandas
```

##### LightGBM

>LightGBM is a gradient boosting framework that uses tree based learning algorithms. It is designed to be distributed and efficient with the following advantages

##### macOS

LightGBM depends on **OpenMP** for compiling, which isn't supported by Apple Clang.

Please install **gcc/g++** by using the following commands:

```

brew install cmake
brew install gcc
```

##### Linux

Install [wheel](https://pythonwheels.com) via pip install wheel first. After that download the wheel file and install from it:
```
pip install lightgbm
```

installing LightGBM as  a python package
