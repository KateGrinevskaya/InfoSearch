- `bert_corpus.py` - файл для составления Term-Document матрицы с помощью BERT (только с *cuda*).
- `bert_search.py` - файл для поиска с помощью BERT (только с *cuda*).
- `fasttext_corpus.py` - файл для составления Term-Document матрицы с помощью FastText (файл с моделью *fasttext* должен находиться в папке с кодом).
- `fasttext_search.py` - файл для поиска с помощью FastText (файл с моделью *fasttext* должен находиться в папке с кодом).
- `scoring.py` - расчёт метрик качества по всем 5 методам. С ним всё достаточно сложно. BM-25 работает как угодно. TF-IDF и Count работают только если скачать вот эти два файла ([count](https://drive.google.com/file/d/1AS7gSslWDWkfLQqxTLWDiXLJRzg20Uhs/view?usp=sharing), [tf-idf](https://drive.google.com/file/d/12wY1YZF7WYcAeNkKz-ISYgU9wc7a8vEL/view?usp=sharing)), для гитхаба они слишком большие. Мне уже было лень переделывать в спарс-матрицы, переделаю для проекта. Bert запускается только с *cuda*, а для FastText нужна модель в папке с кодом.
- `scorings.txt` - файл с полученными значениями метрик качества для всех 5 моделей.
- остальные файлы - технические (матрицы корпусов, вопросы, ответы и т.д.)
__важно:__ в папке с кодом так же должен лежать файл `questions_about_love.jsonl`
