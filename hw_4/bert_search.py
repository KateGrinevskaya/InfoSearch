# импорты

import numpy as np
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


# функция получения cls токена


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


# функция индексации запроса, на выходе которой посчитанный вектор запроса


def get_q_index(query, model):
    encoded_input = tokenizer(query, padding=True, truncation=True, max_length=24, return_tensors='pt')
    encoded_input.to('cuda')
    with torch.no_grad():
        model_output = model(**encoded_input)
    q_array = cls_pooling(model_output, encoded_input['attention_mask'])
    return q_array.reshape(312, 1)


# функция с реализацией подсчета близости запроса и документов корпуса, на выходе которой вектор,
# i-й элемент которого обозначает близость запроса с i-м документом корпуса


def similarity(matrix, q_array):
    return cos(matrix, q_array.t())


# главная функция, объединяющая все это вместе; на входе - запрос,
# на выходе - отсортированные по убыванию имена документов коллекции


def search(query):
    q_array = get_q_index(query, model)
    scores = similarity(matrix, q_array)
    sorted_scores_indx = torch.ravel(torch.argsort(scores, axis=0, descending=True)).cpu()
    corpus_sorted = corpus_doc_names[sorted_scores_indx]
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
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    model.cuda()
    with open ('docs_names.txt', 'r', encoding = 'utf-8') as fh:
            corpus_doc_names = np.array(fh.read().splitlines())[:10000]
    print('Getting matrix...')
    matrix = torch.load('tensors_tiny.pt', map_location='cuda:0').reshape(10000, 312)
    get_query()