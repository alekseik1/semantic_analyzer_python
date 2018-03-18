from scipy.spatial.distance import cosine
from TextMetrics import normed_word, norm_lev
import re
from WordsTokenizer import WordsTokenizer
import numpy as np


class SemanticsClassifier:
    """
    Classifier for requests <--> semantic kernel
    """

    def check_sem(self, req, sem, vec_req, vec_sem):

        req = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in req.split(' ')])
        n_sem = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in sem.split(' ')])

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
            s1 = ' '.join(sorted(req.split(' ')))
            s2 = ' '.join(sorted(n_sem.split(' ')))
            return self.a * cosine(vec_req, vec_sem) + self.b * norm_lev(s1, s2)

    def __init__(self, a=0.7, tokenizer=WordsTokenizer()):
        """
        Classifier for requests <--> semantic kernel
        @param a: Weight for cosine distance
        """
        self.a = a
        self.b = 1-a
        self.tokenizer = tokenizer

    def train(self, data):
        """
        Trains at array for STRINGS for our purposes
        @param data: data to train at
        @return: nothing
        """
        self.vec_sem = self.tokenizer.fit_transform(data, data)
        self.sem = data

    def predict(self, data):
        """
        Predict semantics for array of STRING requests
        @param data: data to be grouped
        @return: Semantics for each data
        """
        vec_req = self.tokenizer.transform(data)
        # Наверно, я все сделал через жопу, но результаты есть!
        predictions = []
        # Бегаем по запросам пользователя
        percent = 0
        for i, element in enumerate(data):
            elem_distances = {}
            # Бегаем по сематич. ядру
            for j, sem in enumerate(self.sem):
                elem_distances[sem] = self.check_sem(element, sem, vec_req[i], self.vec_sem[j])
            nearest_sem = min(elem_distances, key=elem_distances.get)
            predictions.append(nearest_sem)
            # Будем отображать прогресс
            if i / data.shape[0] >= percent + 0.05:
                percent = round((i/data.shape[0]), 3)
                print("{}% is done".format(percent*100))
        print("ALL DONE!")
        return np.array(predictions)
