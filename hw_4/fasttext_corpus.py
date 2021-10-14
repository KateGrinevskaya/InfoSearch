# импорты

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
import nltk
import json
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
from pymorphy2 import MorphAnalyzer
morph = MorphAnalyzer()
from string import punctuation
import gensim


# Функция получения ответа с самым большим рейтингом

def get_clever_answer(question):
    a = json.loads(question)
    if a['answers']:
        try:
            values = np.array([int(float(ans['author_rating']['value'])) for ans in a['answers']])
            answer = a['answers'][np.argmax(values)]['text']
        except ValueError:
            answer = None
    else:
        answer = None
    return answer


# Функция загрузки нужных документов из коллекции

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
    clean_text = ' '.join([f(w) for w in text if w.isalpha()])
    return clean_text


# Функция индексации корпуса, на выходе которой посчитанная матрица Document-Term,
# где каждая строка - вектор усреднённых значений векторов слов этого документа


def get_index(texts, model):
    corpus_vec = []
    for text in texts:
        if len(text.split()) > 1:
            doc_vec = np.mean([model[w] for w in text.split()], axis=0)
        if len(text.slit()) == 1:
            doc_vec = model[text]
        else:
            doc_vec = np.zeros(300)
        corpus_vec.append(np.expand_dims(doc_vec, axis=0))
        if np.expand_dims(doc_vec, axis=0).shape != corpus_vec[0].shape:
            print(text)
    return corpus_vec


def get_indexed_corpus(filename):
    with open ('docs_names.txt', 'r', encoding='utf-8') as fh:
        docs = fh.read().splitlines()[:10000]
    corpus = [preproc(d) for d in docs]
    #with open('preproc_corpus.txt', 'r', encoding='utf-8') as fh:
        #corpus = fh.read().splitlines()
    print('Getting model...')
    model = gensim.models.KeyedVectors.load("araneum_none_fasttextcbow_300_5_2018.model")
    print('Getting vectors...')
    corpus_vec = get_index(corpus, model)
    print('Vectorization is done!')
    return corpus_vec


if __name__ == "__main__":
    print("Let's start vectorization!")
    vecs = get_indexed_corpus('questions_about_love.jsonl')
    vecs = np.vstack(vecs)
    #with open ('docs_names.txt', 'r', encoding = 'utf-8') as fh:
        #corpus_doc_names = np.array(fh.read().splitlines())
    print('Saving to npy...')
    np.save('term-doc-10', np.array(vecs))
    print('Done!')