import pickle
import sys


class Predictor():

    def __init__(self, model, vectorizer, label_encoder):
        self.model = model
        self.vectorizer = vectorizer
        self.index_to_label = self.create_labels_dict(label_encoder)

    def create_labels_dict(self,label_encoder):
        labels = label_encoder.classes_
        tuples = zip(range(0,len(labels)), labels)
        return dict(tuples)

    def predict(self, text):
        X = self.vectorizer.transform([text])
        prediction_proba = self.model.predict_proba(X)
        prediction_dict = dict(zip(self.index_to_label.values(), prediction_proba[0]))
        return prediction_dict


def from_pickle(path):
    with open(path, 'rb') as fin:
        return pickle.load(fin)


if __name__ == "__main__":
    params = sys.argv[1:]
    if not params:
        print("No text to predict!")
        exit(1)
    text_to_predict = params[0]
    model = from_pickle('bayes.pickle')
    vectorizer = from_pickle('vectorizer.pickle')
    label_encoder = from_pickle('label_encoder.pickle')
    predictor = Predictor(model, vectorizer, label_encoder)
    prediction = predictor.predict(text_to_predict)
    print(prediction)

