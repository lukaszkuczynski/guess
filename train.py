import pandas as pd
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split


def train_naivebayes(X, y):
#     X = df['Xdense']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    bayes = MultinomialNB()
    bayes_model = bayes.fit(X_train, y_train)

    y_pred = bayes_model.predict(X_test)
    score = f1_score(y_test, y_pred, average='weighted')
    print("F1 score for model is %.2f" % score)

    return bayes_model


def read_pickled_df(path):
    return pd.read_pickle(path)


def read_pickled_object(path):
    with open(path, 'rb') as fread:
        return pickle.load(fread)


def save_model(path, model):
    with open(path, 'wb') as fout:
        pickle.dump(model, fout)


if __name__ == "__main__":
    dataframe_path = '.\\df.pickle'
    vectorizer_path = '.\\vectorizer.pickle'
    df = read_pickled_df(dataframe_path)
    X = read_pickled_object('.\\X.pickle')
    vectorizer = read_pickled_object(vectorizer_path)
    y = df['tag']
    bayes_model = train_naivebayes(X, y)
    save_model('.\\bayes.pickle', bayes_model)
    
    