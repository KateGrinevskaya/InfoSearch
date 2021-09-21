# импорты

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import os
import re
import nltk
import pickle
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
mystem = Mystem()
russian_stopwords = stopwords.words("russian")


# Функция препроцессинга данных. Включите туда лемматизацию,
# приведение к одному регистру, удаление пунктуации и стоп-слов.


def get_preproc(dir_name):
    corpus = []
    curr_dir = os.getcwd()
    files_dir = os.path.join(curr_dir, dir_name)
    for root, dirs, files in os.walk(files_dir):
        for name in files:
            with open('files_list.txt', 'a', encoding='utf-8') as file:
                file.write(name + '\n')
            fpath = os.path.join(root, name)
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().replace('\n\n', ' ').replace('\n', ' ')
                text = re.sub(r'\d+|[a-zA-Z]+', ' ', text)
                tokens = mystem.lemmatize(text.lower())
                puncts = '''!@#$%^&"*()«»_+.—!!!\,|/,...:;?-!.'''
                tokens = [token for token in tokens if token not in russian_stopwords\
                          and token.isalpha()]
                clean_text = " ".join(tokens)
                corpus.append(clean_text)
    return corpus


# функция индексации корпуса, на выходе которой посчитанная матрица Document-Term
# сохранение векторайзера в tfidf.pickle


def get_index(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    pickle.dump(vectorizer.vocabulary_, open('tfidf.pickle', 'wb'))
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    return df

# сохранение матрицы Document-Term в corpus_df.csv


if __name__ == "__main__":
    corpus = get_preproc('friends-data')
    df = get_index(corpus)
    df.to_csv('corpus_df.csv')