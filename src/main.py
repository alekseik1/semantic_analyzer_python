import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import itertools
import pymorphy2
import operator
from TextMetrics import *
from scipy.spatial.distance import cosine
from WordsTokenizer import WordsTokenizer

morph = pymorphy2.MorphAnalyzer()
def normed_word(x):
    p = morph.parse(x)[0]
    return p.normal_form

df1 = pd.read_csv('keyword_names.csv')
df2 = pd.read_csv('keywords_data.csv')

# ТОКЕНИЗАЦИЯ ЗАПРОСОВ ПО УНИКАЛЬНЫМ СЛОВАМ
tokenizer = WordsTokenizer()
tokenizer.fit(df1['keyword_name'].values)

cos_matrix = tokenizer.transform(df2['search'].values)
sem_cos_matrix = tokenizer.transform(df1['keyword_name'])

def check_sem(req, sem):

    req = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in req.split(' ')])
    n_sem = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in sem.split(' ')])

    vec_req = tokenizer.transform(np.array([req]))
    vec_sem = tokenizer.transform(np.array([n_sem]))

    if '-' in sem:
        minus_words = []
        for part in sem.split('-'):
            minus_words.append(part.split(' ')[0])
        for bad_word in minus_words:
            if bad_word in req.split(' '):
                return 1
    if '+' in sem:
        plus_words = []
        for part in sem.split('+'):
            plus_words.append(part.split(' ')[0])
        for required_word in plus_words:
            if required_word not in req.split(' '):
                return 1
    if '"' in sem:
        if np.count_nonzero(vec_req - vec_sem) == 0:
            return 0
        else:
            return 1
    elif '[' in sem:
        # TODO: здесь пока будет заглушка (`[]` -- это отстой)
        return 1
    else:
        # Пытаемся примерно предсказать
        # самая подгонистая часть
        a = 0.7
        b = 1-a
        s1 = ' '.join(sorted(req.split(' ')))
        s2 = ' '.join(sorted(n_sem.split(' ')))
        return a*cosine(vec_req, vec_sem) + b*norm_lev(s1, s2)


# Наверно, я все сделал через жопу, но результаты есть!
predictions = []
# Бегаем по запросам пользователя
for i, element in enumerate(df2['search'].values):
    elem_distances = {}
    # Бегаем по сематич. ядру
    for j, sem in enumerate(df1['keyword_name'].values):
        elem_distances[sem] = check_sem(element, sem)
    nearest_sem = min(elem_distances, key=elem_distances.get)
    predictions.append(nearest_sem)
    #print('"{}" has nearest {} --- elem_distances={}'
    #    .format(element, nearest_sem, round(elem_distances[nearest_sem], 3)))

df2['predictions'] = predictions
df2.to_csv('result.csv')
