# Я ориентировалась на то что, что в папке,
# в которой лежит этот код, также находится
# папка `friends-data`, в которой лежат папки по сезонам.

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
import os
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
mystem = Mystem()
russian_stopwords = stopwords.words("russian")
from collections import Counter

# Функция препроцессинга данных. Включите туда лемматизацию,
# приведение к одному регистру, удаление пунктуации и стоп-слов.
def get_preproc(dir_name):
    corpus = []
    curr_dir = os.getcwd()
    files_dir = os.path.join(curr_dir, dir_name)
    for root, dirs, files in os.walk(files_dir):
        for name in files:
            fpath = os.path.join(root, name)
            with open(fpath, 'r', encoding='utf-8') as f:
                text = f.read().replace('\n\n', ' ').replace('\n', ' ')
                text = re.sub(r'\d+|[a-zA-Z]+', ' ', text)
                tokens = mystem.lemmatize(text.lower())
                puncts = '''!@#$%^&"*()«»_+.—!!!\,|/,...:;?-!.'''
                tokens = [token for token in tokens if
                          token not in russian_stopwords\
                          and token.isalpha()]
                clean_text = " ".join(tokens)
                corpus.append(clean_text)
    return corpus

# Функция индексирования данных.
# На выходе создает обратный индекс, он же матрица Term-Document.
def get_index(corpus):
    vectorizer = CountVectorizer(analyzer='word')
    X = vectorizer.fit_transform(corpus)
    df = pd.DataFrame(X.todense(), columns=vectorizer.get_feature_names())
    df.loc['TOTAL'] = df.sum()
    return df

# a) какое слово является самым частотным?
def most_common_word(df):
    words_dict = {}
    for col in df.columns:
        words_dict[col] = df[col]['TOTAL']
    return Counter(words_dict).most_common(1)[0][0]

# b) какое самым редким
def least_common_word(df):
    words_dict = {}
    for col in df.columns:
        words_dict[col] = df[col]['TOTAL']
    return Counter(words_dict).most_common()[-1][0]

# c) какой набор слов есть во всех документах коллекции
def words_in_all_docs(df):
    words_all_docs = []
    for col in df.columns:
        if 0 not in df[col].to_list():
            words_all_docs.append(col)
    return words_all_docs

# d) кто из главных героев статистически самый
# популярный (упоминается чаще всего)?
def most_popular_person(df):
    names = {'моника': 0, 'рэйчел': 0, 'чендлер': 0, 'фиби': 0, 'росс': 0,
           'джоуи': 0, 'мон': 0, 'рейч': 0, 'чэндлер': 0, 'чен': 0, 'фибс': 0,
           'джои': 0, 'джо': 0}
    persons = {}
    for name in names.keys():
        try:
            names[name] = df[name]['TOTAL']
        except:
            continue
    persons['моника'] = names['моника'] + names['мон']
    persons['рэйчел'] = names['рэйчел'] + names['рейч']
    persons['чендлер'] = names['чендлер'] + names['чэндлер'] + names['чен']
    persons['фиби'] = names['фиби'] + names['фибс']
    persons['росс'] = names['росс']
    persons['джоуи'] = names['джоуи'] + names['джои'] + names['джо']
    return Counter(persons).most_common(1)[0][0].title()

if __name__ == "__main__":
    corpus = get_preproc('friends-data')
    df = get_index(corpus)
    print(f'Самое частотное слово - "{most_common_word(df)}"')
    print(f'Самое редкое слово - "{least_common_word(df)}"')
    print('Набор слов, которые есть во всех документах коллекции: ',
          words_in_all_docs(df))
    print('Самый популярный герой - ', most_popular_person(df))
