import pickle


class Predictor():

    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer

    def predict(self, text):
        X = self.vectorizer.transform([text])
        prediction = self.model.predict_proba(X)
        return prediction


def from_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


if __name__ == "__main__":
    model_path = 'bayes.pickle'
    vectorizer_path = 'vectorizer.pickle'
    model = from_pickle(model_path)
    vectorizer = from_pickle(vectorizer_path)
    predictor = Predictor(model, vectorizer)
    text = "nice app in javascript with gatsby"
    prediction = predictor.predict(text)
    print(prediction)

