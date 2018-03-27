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

    def __init__(self, a=0.7, tokenizer=WordsTokenizer(), n_jobs=multiprocessing.cpu_count(), be_verbose=False):
        """
        Classifier for requests <--> semantic kernel
        @param a: Weight for cosine distance
        """
        self.a = a
        self.b = 1-a
        self.tokenizer = tokenizer
        self.n_jobs = n_jobs
        self.sem = None
        self.vec_sem = None
        self.n_sem = None
        self.verbose = be_verbose

    def _normalize_semantics(self, sem):
        return ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in sem.split(' ')])

    def train(self, data):
        """
        Trains at array for STRINGS for our purposes
        @param data: data to train at
        @return: nothing
        """
        if len(self.tokenizer.uniq_words) == 0:
            self.vec_sem = self.tokenizer.fit_transform(data, data)
        else:
            self.vec_sem = self.tokenizer.transform(data)
        self.sem = data
        self.n_sem = Parallel(n_jobs=self.n_jobs)(delayed(self._normalize_semantics)(sem) for sem in data)
        if self.verbose:
            print("Sematics classifier is trained!")

    # ПЕРЕДАВАТЬ ЕМУ ТОЛЬКО НОРМАЛИЗОВАННУЮ СЕМАНТИКУ!!!
    def _check(self, element, vec_req, i, n_sem, j):
        return self.check_sem(element, n_sem, vec_req[i], self.vec_sem[j])

    def _make_predictions_multithread(self, i, element, vec_req):
        element = ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in element.split(' ')])
        # Бегаем по сематич. ядру
        elem_distances = dict(zip(self.sem, [self._check(element, vec_req, i, n_sem, j) for j, n_sem in enumerate(self.n_sem)]))
        nearest_sem = min(elem_distances, key=elem_distances.get)
        if elem_distances[nearest_sem] < 0.3:
            return nearest_sem
        else:
            return 'Unknown'

    def predict(self, data):
        """
        Predict semantics for array of STRING requests
        @param data: data to be grouped
        @return: Semantics for each data
        """
        vec_req = self.tokenizer.transform(data)
        predictions = []
        for i in range(0, data.shape[0], 1000):
            # Если выпираем, то
            if 1000 + i > data.shape[0]:
                pred_data = data[i:]
            else:
                pred_data = data[i:1000 + i]

            tmp = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(self._make_predictions_multithread)(i, element, vec_req)
                                                   for i, element in enumerate(pred_data))
            predictions += tmp
            if self.verbose:
                print("--------------------------")
                print("Made from {} to {}".format(i, i + 1000))
                print("--------------------------")
        return np.array(predictions)
