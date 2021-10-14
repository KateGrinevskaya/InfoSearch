# импорты

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
import re
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
import gensim
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


# Функция препроцессинга данных. Включите туда лемматизацию,
# приведение к одному регистру, удаление пунктуации и стоп-слов.


def preproc(raw_text):
    text = raw_text.lower().replace('\n\n', ' ').replace('\n', ' ').split()
    f = lambda w: morph.parse(w.strip(punctuation))[0].normal_form
    clean_text = ' '.join([f(w) for w in text if w.isalpha()])
    return clean_text


# функция получения cls токена


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]

# подсчёт близости для разных метрик


def bert_similarity(matrix, q_array):
    return cosine_similarity(np.array(matrix.cpu()), np.array(q_array.cpu()).T)

def fasttext_similarity(matrix, q_array):
    return np.dot(matrix, q_array.T)

def tfidf_similarity(q_array, corpus_matrix):
    c_s = cosine_similarity(corpus_matrix, q_array)
    return c_s


# функция получения вопросов к ответам
# в main'e не запускается, была запущена мной раньше


def get_questions():
    with open ('docs_names.txt', 'r', encoding='utf-8') as fh:
        corpus = fh.read().splitlines()[:10000]
    with open ('/content/drive/MyDrive/questions_about_love.jsonl', 'r', encoding='utf-8') as f:
        ask_reply = list(f)[:20000]
    ask_reply_dict = {}
    for q in ask_reply:
        q = json.loads(q)
        value = q['question']
        if q['answers']:
            try:
                values = np.array([int(float(ans['author_rating']['value'])) for ans in q['answers']])
                answer = q['answers'][np.argmax(values)]['text']
            except ValueError:
                answer = None
        else:
            answer = None
        key = answer
        ask_reply_dict[key] = value
    questions = []
    for doc in corpus:
        try:
            questions.append(ask_reply_dict[doc])
        except:
            continue
    with open ('questions.txt', 'w', encoding='utf-8') as file:
        file.writelines("%s\n" % line for line in questions)
    return questions

# подсчёт метрик для разных моделей


def bert_scoring():
    print('Getting model...')
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    model.cuda()
    print('Getting vectors...')
    q_arrays = []
    for q in questions:
        encoded_input = tokenizer(q, padding=True, truncation=True, max_length=24, return_tensors='pt')
        encoded_input.to('cuda')
        with torch.no_grad():
            model_output = model(**encoded_input)
        q_array = cls_pooling(model_output, encoded_input['attention_mask'])
        q_arrays.append(q_array)
    q_arrays = torch.stack(q_arrays).reshape(312, 10000)
    print('Making dot product...')
    matrix = torch.load('tensors_tiny.pt', map_location='cuda:0').reshape(10000, 312)
    scores = torch.from_numpy(bert_similarity(matrix, q_arrays))
    scores = torch.argsort(scores, axis=0, descending=True)
    return scores


def tfidf_scoring():
    vectorizer = TfidfVectorizer(vocabulary=pickle.load(open("tfidf.pickle", "rb")))
    query_ready = [preproc(q) for q in questions]
    q_arrays = vectorizer.fit_transform(query_ready)
    df = pd.read_csv('tf_idf.csv')
    if 'Unnamed: 0' in df.columns.tolist():
        df = df.drop('Unnamed: 0', axis='columns')
    scores = tfidf_similarity(q_arrays, df.values)
    scores = np.argsort(scores, axis=0)[::-1]
    print(scores.shape)
    return scores


def count_scoring():
    vectorizer = CountVectorizer(analyzer='word', vocabulary=pickle.load(open("count.pickle", "rb")))
    query_ready = [preproc(q) for q in questions]
    q_arrays = vectorizer.fit_transform(query_ready)
    df = pd.read_csv('count.csv')
    if 'Unnamed: 0' in df.columns.tolist():
        df = df.drop('Unnamed: 0', axis='columns')
    scores = tfidf_similarity(q_arrays, df.values)
    scores = np.argsort(scores, axis=0)[::-1]
    return scores


def bm25_scoring():
    loader = np.load('sparsed_matrix_10.npz')
    bm_matrix = sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
    query_ready = [preproc(q) for q in questions]
    vectorizer = CountVectorizer(vocabulary=pickle.load(open("bm25.pickle", "rb")))
    q_arrays = vectorizer.fit_transform(query_ready)
    scores = fasttext_similarity(bm_matrix, q_arrays).toarray()
    scores = np.argsort(scores, axis=0)[::-1]
    return scores


def fasttext_scoring():
    model = gensim.models.KeyedVectors.load("araneum_none_fasttextcbow_300_5_2018.model")
    matrix = np.load('term-doc-10.npy', allow_pickle=True)
    q_arrays = []
    for q in questions:
        q = preproc(q)
        tokens = q.split()
        if len(tokens) == 0:
            q_array = np.zeros(300)
        else:
            if len(tokens) == 1:
                q_array = model[tokens[0]]
            else:
                q_array = np.mean([model[t] for t in tokens], axis=0)
        q_arrays.append(q_array)
    q_arrays = np.vstack(q_arrays)
    scores = fasttext_similarity(matrix, q_arrays)
    scores = np.argsort(scores, axis=0)[::-1]
    return scores


# функция, которая считает итоговую метрику на топ-5 результатов по логике, описанной в лекции


def scoring(method):
    if method=='bert':
        scores = bert_scoring()
    if method=='fasttext':
        scores = fasttext_scoring()
    if method=='bm25':
        scores = bm25_scoring()
    if method=='tf-idf':
        scores = tfidf_scoring()
    if method=='count':
        scores = count_scoring()
    scores = scores.T
    counter = 0
    for i, raw in enumerate(scores):
        if i in raw[:5]:
            counter += 1
    metrics = counter/(scores.shape[0])
    return metrics


if __name__ == "__main__":
    with open ('docs_names.txt', 'r', encoding='utf-8') as fh:
        answers = fh.read().splitlines()[:10000]
    #questions = get_questions()
    with open ('questions.txt', 'r', encoding='utf-8') as fh:
        questions = fh.read().splitlines()
    methods = ['count', 'tf-idf', 'bm25', 'fasttext', 'bert']
    scorings = {}
    for m in methods:
        print('Doing ', m, '...')
        s = scoring(m)
        with open('scorings.txt', 'a', encoding='utf-8') as f:
            f.write(m+' '+str(s))