# импорты


import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def cls_pooling(model_output, attention_mask):
    return model_output[0][:,0]


def get_indexed_corpus(texts, model):
    vectors = []
    for text in texts:
        encoded_input = tokenizer(text, padding=True, truncation=True, max_length=24, return_tensors='pt')
        encoded_input.to('cuda')
        with torch.no_grad():
            model_output = model(**encoded_input)
            vector = cls_pooling(model_output, encoded_input['attention_mask'])
        vectors.append(vector)
    vectors = torch.stack(vectors)
    return vectors


if __name__ == "__main__":
    print("Getting corpus...")
    with open ('docs_names.txt', 'r', encoding='utf-8') as fh:
        corpus = fh.read().splitlines()[:10000]
    print('Getting model...')
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
    model = AutoModel.from_pretrained("cointegrated/rubert-tiny")
    model.cuda()
    print("Let's start vectorization!")
    vecs = get_indexed_corpus(corpus, model)
    torch.save(vecs, 'tensors_tiny.pt')
    print('Done!')