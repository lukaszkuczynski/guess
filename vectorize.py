import pandas as pd
from nltk.tokenize import RegexpTokenizer


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


def to_vector(df):
    from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
    vectorizer = TfidfVectorizer(max_df=0.5, min_df=4, use_idf=True)
    df['features'] = df.apply(lambda row: ' '.join(row['cleaned']), axis=1)
    vectorizer = vectorizer.fit(df['features'])
    X = vectorizer.transform(df['features'])
    df['X'] = X
    return df


if __name__ == "__main__":
    path = 'c:\\data\\devto_data\\'
    file_tags = {
        'python' : 'python_posts.csv',
        'java' : 'java_posts.csv',
        'javascript' : 'javascript_posts.csv',
        'devops' : 'devops_posts.csv',
    }
    df_read = df_from_files(path, file_tags)
    df_tokenized = tokenize(df_read)
    df_cleaned = clean(df_tokenized)
    df_featurized = to_vector(df_cleaned)

    df = df_featurized
    print(df.head())
    