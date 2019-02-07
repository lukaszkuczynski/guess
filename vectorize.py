import pandas as pd
from nltk.tokenize import RegexpTokenizer
import pickle


def df_from_files(path, file_tags):
    tag_no = 0
    frames = []
    for tag, filename in file_tags.items():
        tag_no += 1
        df = pd.read_csv(path + filename)
        df['tag'] = tag_no
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


def fit_vectorizer(df):
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=4, use_idf=True)
    df['features'] = df.apply(lambda row: ' '.join(row['cleaned']), axis=1)
    vectorizer = vectorizer.fit(df['features'])
    X = vectorizer.transform(df['features'])
    df['X'] = X
    return vectorizer


def save_vectorizer(save_path, vectorizer):
    vectorizer_path = save_path+'vectorizer.pickle'
    with open(vectorizer_path, 'wb') as fout:
        pickle.dump(vectorizer, fout)
        return vectorizer_path


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
    vectorizer = fit_vectorizer(df_cleaned)
    vectorizer_path = save_vectorizer(save_path, vectorizer)
    print("Data fitted, vectorizer save as file to path '%s'" % vectorizer_path)    