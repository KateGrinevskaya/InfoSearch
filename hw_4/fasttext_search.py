# импорты

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from pymystem3 import Mystem
mystem = Mystem()
russian_stopwords = stopwords.words("russian")
import gensim


# функция индексации запроса, на выходе которой посчитанный вектор запроса


def get_q_index(query, model):
    text = re.sub(r'\d+|[a-zA-Z]+', ' ', query)
    tokens = mystem.lemmatize(text.lower())
    tokens = [token for token in tokens if token not in russian_stopwords\
              and token.isalpha()]
    if len(tokens) == 0:
        return 'В вашем запросе только цифры, латиница и знаки препинания/пробелов'
    else:
        if len(tokens) == 1:
            q_array = model[tokens[0]]
        else:
            q_array = np.mean([model[t] for t in tokens], axis=0)
        return q_array


# функция с реализацией подсчета близости запроса и документов корпуса, на выходе которой вектор,
# i-й элемент которого обозначает близость запроса с i-м документом корпуса


def similarity(matrix, q_array):
    return np.dot(matrix, q_array.T)


# главная функция, объединяющая все это вместе; на входе - запрос,
# на выходе - 5 отсортированных по убыванию имён документов коллекции


def search(query):
    q_array = get_q_index(query, model)
    scores = similarity(matrix, q_array)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    corpus_sorted = corpus_doc_names[sorted_scores_indx.ravel()]
    return corpus_sorted[:5]


# функция, получающая запрос


def get_query():
    ans = 'да'
    while ans == 'да':
        query = str(input('Введите запрос: '))
        corpus_sorted = search(query)
        if len(corpus_sorted) != 0:
            with open('answer.txt', 'w', encoding='utf-8') as f:
                f.writelines("%s\n" % line for line in corpus_sorted)
            print('Результат запроса сохранён в файл answer.txt')
        else:
            print('По вашему запросу ничего не найдено :(')
        ans = str(input('Вы желаете искать снова (да/нет)? ')).lower()
    print('Результат последнего запроса сохранён в файл answer.txt')


if __name__ == "__main__":
    print('Getting model...')
    model = gensim.models.KeyedVectors.load("araneum_none_fasttextcbow_300_5_2018.model")
    with open ('docs_names.txt', 'r', encoding = 'utf-8') as fh:
            corpus_doc_names = np.array(fh.read().splitlines())
    print('Getting matrix...')
    matrix = np.load('term-doc.npy', allow_pickle=True)
    get_query()