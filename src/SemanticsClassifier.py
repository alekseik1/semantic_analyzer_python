from scipy.spatial.distance import cosine
from TextMetrics import normed_word, norm_lev
import re
from WordsTokenizer import WordsTokenizer
import numpy as np
from joblib import Parallel, delayed
import multiprocessing


class SemanticsClassifier:
    """
    Classifier for requests <--> semantic kernel
    """

    def check_sem(self, req, sem, vec_req, vec_sem):
        """
        Проверяет, хороша ли семантика. АХТУНГ! Надо слова нормировать (смотрите исходники ниже)
        @param req: Фраза из запроса
        @param sem: Фраза из семантики
        @param vec_req: Векторизированная фраза из запроса
        @param vec_sem: Векторизированная фраза из семантики
        @return: Процент схожести (от 0 до 1)
        """
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
            s2 = ' '.join(sorted(sem.split(' ')))
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

    def _check(self, element, vec_req, i, sem, j):
        n_sem = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in sem.split(' ')])
        return self.check_sem(element, n_sem, vec_req[i], self.vec_sem[j])

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
            element = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in element.split(' ')])
            # Бегаем по сематич. ядру
            num_cores = multiprocessing.cpu_count()
            elem_distances = dict(zip(self.sem,
                                      Parallel(n_jobs=num_cores)(delayed(self._check)(element, vec_req, i, sem, j)
                                                                 for j, sem in enumerate(self.sem))))
            nearest_sem = min(elem_distances, key=elem_distances.get)
            predictions.append(nearest_sem)
            # Будем отображать прогресс
            if i / data.shape[0] >= percent + 0.01:
                percent = round((i/data.shape[0]), 3)
                print("{}% is done".format(percent*100))
        print("ALL DONE!")
        return np.array(predictions)
