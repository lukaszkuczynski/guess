# Installation
For Windows environment I encourage to install using Anaconda distribution.  Installing `sklearn` and other `numpy` related packages using pip can be painful.

# Run

## Get files

## Create vectors
Before training a model we need to convert text into vectors. 
```bash
python vectorize.py
```
It should create two files: `df_train.pickle` containing data that our model will fit on, and `vectorizer.pickle` that will be used to convert coming text to a vectore.