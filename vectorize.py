import pandas as pd


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


def clean_docs(df):
    pass


def to_vector(df):
    pass


if __name__ == "__main__":
    path = 'c:\\data\\devto_data\\'
    file_tags = {
        'python' : 'python_posts.csv',
        'java' : 'java_posts.csv',
        'javascript' : 'javascript_posts.csv',
        'devops' : 'devops_posts.csv',
    }
    df = df_from_files(path, file_tags)
    print(df.describe())
    