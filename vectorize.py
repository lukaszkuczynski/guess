import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pickle
from sklearn.preprocessing import LabelEncoder


def df_from_files(path, file_tags):
    frames = []
    for tag, filename in file_tags.items():
        df = pd.read_csv(path + filename)
        df['tag_text'] = tag
        frames.append(df)
    df = pd.concat(frames)
    df = df.dropna()
    return df


def tokenize(df):
    tokenizer = RegexpTokenizer(r'\w+')

    def tokenize_row(row):
        tokens = tokenizer.tokenize(row['content'])
        row['tokens'] = [str.lower(token) for token in tokens]
        return row

    df = df.apply(tokenize_row, axis=1)
    return df


def clean(df):
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords')

    def remove_stopwords(row):
        stop_words = stopwords.words('english')
        row['cleaned'] = [w for w in row['tokens'] if not w in stop_words] 
        return row
    
    df = df.apply(remove_stopwords, axis=1)
    return df


def label(df):
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df['tag_text'])
    df['label'] = labels
    return df, label_encoder


def fit_vectorizer(df):
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=4, use_idf=True)
    df['features'] = df.apply(lambda row: ' '.join(row['cleaned']), axis=1)
    vectorizer = vectorizer.fit(df['features'])
    X = vectorizer.transform(df['features'])
    return vectorizer, X


def save_vectorizer(save_path, vectorizer):
    vectorizer_path = save_path+'vectorizer.pickle'
    with open(vectorizer_path, 'wb') as fout:
        pickle.dump(vectorizer, fout)
        return vectorizer_path


def save_pickle(save_path, object_to_save):
    dataframe_path = save_path
    with open(save_path, 'wb') as fout:
        pickle.dump(object_to_save, fout)


if __name__ == "__main__":
    data_path = 'c:\\data\\devto_data\\'
    save_path = '.\\'
    file_tags = {
        'python' : 'python_posts.csv',
        'java' : 'java_posts.csv',
        'javascript' : 'javascript_posts.csv',
        'devops' : 'devops_posts.csv',
    }
    df_read = df_from_files(data_path, file_tags)
    df_tokenized = tokenize(df_read)
    df_cleaned = clean(df_tokenized)
    df_labeled, label_encoder = label(df_cleaned)
    vectorizer, X = fit_vectorizer(df_labeled)
    vectorizer_path = save_vectorizer(save_path, vectorizer)
    save_pickle(save_path+'X.pickle', X)
    save_pickle(save_path+'y.pickle', df_cleaned['label'])
    save_pickle(save_path+'label_encoder.pickle', label_encoder)
    print("Data fitted, vectorizer saved as file to path '%s'" % vectorizer_path)
    print("X and y saved as file to path '%s'" % save_path)