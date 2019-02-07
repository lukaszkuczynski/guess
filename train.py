import pandas as pd
import pickle


def train_naivebayes(df):
    pass


def read_pickled_df(path):
    return pd.read_pickle(path)


def read_pickled_object(path):
    with open(path, 'rb') as fread:
        return pickle.load(fread)

if __name__ == "__main__":
    dataframe_path = '.\\df_train.pickle'
    vectorizer_path = '.\\vectorizer.pickle'
    df_train = read_pickled_df(dataframe_path)
    vectorizer = read_pickled_object(vectorizer_path)
    print(type(df_train))
    print(type(vectorizer))
    