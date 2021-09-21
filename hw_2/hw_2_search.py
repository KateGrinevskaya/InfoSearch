# импорты

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
import re
import nltk
import pickle
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
mystem = Mystem()
russian_stopwords = stopwords.words("russian")
from sklearn.metrics.pairwise import cosine_similarity


# функция индексации запроса, на выходе которой посчитанный вектор запроса


def get_q_index(query):
    vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("tfidf.pickle", "rb")))
    text = re.sub(r'\d+|[a-zA-Z]+', ' ', query)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token.isalpha()]
    if len(tokens) == 0:
        return 'В вашем запросе только цифры, латиница и знаки препинания/пробелов'
    else:
        query_ready = ' '.join(tokens)
        q_array = vectorizer.fit_transform([query_ready]).toarray()
        return q_array


# функция с реализацией подсчета близости запроса и документов корпуса,
# на выходе которой вектор, i-й элемент которого обозначает близость
# запроса с i-м документом корпуса


def simil(q_array, corpus_matrix):
    c_s = cosine_similarity(q_array, corpus_matrix)
    return c_s


# главная функция, объединяющая все это вместе;
# на входе - запрос, на выходе - отсортированные по убыванию
# имена документов коллекции


def search(query):
    q_array = get_q_index(query)
    if isinstance(q_array, str):
        docs_col = q_array
    else:
        df = pd.read_csv('corpus_df.csv')
        if 'Unnamed: 0' in df.columns.tolist():
            df = df.drop('Unnamed: 0', axis='columns')
        c_s = simil(q_array, df.values)
        ranged_ind = np.argsort(c_s[0])[::-1]
        files_names = []
        docs_col = []
        with open('files_list.txt', 'r', encoding='utf-8') as file:
            for line in file:
                files_names.append(line)
        for n in ranged_ind:
            docs_col.append(files_names[n])
    return docs_col


if __name__ == "__main__":
    ans = 'да'
    while ans == 'да':
        query = str(input('Введите запрос: '))
        lines = search(query)
        with open('answer.txt', 'w', encoding='utf-8') as f:
            f.writelines("%s" % line for line in lines)
        print('Результат запроса сохранён в файл answer.txt')
        ans = str(input('Вы желаете искать снова (да/нет)? ')).lower()
    print('Результат последнего запроса сохранён в файл answer.txt')