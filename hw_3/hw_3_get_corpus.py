# импорты

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
import nltk
import pickle
import json
from scipy import sparse
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
from string import punctuation


# Функция получения ответа с самым большим рейтингом


def get_clever_answer(question):
    a = json.loads(question)
    if a['answers']:
        try:
            values = np.array([int(float(ans['author_rating']['value'])) for ans in a['answers']])
            answer = a['answers'][np.argmax(values)]['text']
        except ValueError:
            answer = None
        return answer
    else:
        answer = None


# Функция загрузки нужных документов из коллекции
# сохранение названий документов (в нашем случае просто ответов) в файл docs_names.txt


def get_docs(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        ask_reply = list(f)[:50000]
    corpus_with_None = list(map(get_clever_answer, ask_reply))
    corpus = list(filter(None.__ne__, corpus_with_None))
    with open ('docs_names.txt', 'w', encoding = 'utf-8') as fh:
         fh.writelines("%s\n" % line for line in corpus)
    return corpus


# Функция препроцессинга данных. Включите туда лемматизацию,
# приведение к одному регистру, удаление пунктуации и стоп-слов.


def preproc(raw_text):
    text = raw_text.lower().replace('\n\n', ' ').replace('\n', ' ').split()
    f = lambda w: morph.parse(w.strip(punctuation))[0].normal_form
    clean_text = ' '.join([f(w) for w in text])
    return clean_text


# Функция индексации корпуса, на выходе которой посчитанная sparse-матрица Document-Term,
# в ячейках которой находятся посчитанные bm25
# сохранение векторайзера в count_vect.pickle


def get_index(texts):
    tf_vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    x_count_vec = count_vectorizer.fit_transform(texts) # для индексации запроса
    pickle.dump(count_vectorizer.vocabulary_, open('count_vect.pickle', 'wb'))
    x_tf_vec = tf_vectorizer.fit_transform(texts) # матрица с tf
    x_tfidf_vec = tfidf_vectorizer.fit_transform(texts) # матрица для idf
    idf = tfidf_vectorizer.idf_
    idf = np.expand_dims(idf, axis=0)[0]
    tf = x_tf_vec
    k = 2
    b = 0.75
    len_d = x_count_vec.sum(axis=1)
    avdl = len_d.mean()
    values = []
    rows = []
    cols = []
    for i, j in zip(*tf.nonzero()):
        B_1 = (k * (1 - b + b * int(len_d[i]) / avdl))
        B = tf[i,j] + B_1
        A = (k + 1) * tf[i,j] * idf[j]
        bm25 = A/B
        rows.append(i)
        cols.append(j)
        values.append(bm25)
    sparse_tf = sparse.csr_matrix((values, (rows, cols)))
    return sparse_tf


# главная функция
# на всякий случай: запрепроцешенный корпус можно сохранить


def get_indexed_corpus(filename):
    docs = get_docs(filename)
    corpus = [preproc(d) for d in docs]
    #with open('preproc_corpus.txt', 'w', encoding='utf-8') as f:
        #f.writelines(c + '\n' for c in corpus)
    sparse_matrix = get_index(corpus)
    return sparse_matrix


if __name__ == "__main__":
    array = get_indexed_corpus('questions_about_love.jsonl')
    np.savez('sparsed_matrix', data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)