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
            for i, part in enumerate(sem.split('-')):
                if i == 0:
                    continue
                minus_words.append(part.split(' ')[0])
            for bad_word in minus_words:
                if bad_word in req.split(' '):
                    return 10
        if '+' in sem:
            plus_words = []
            for i, part in enumerate(sem.split('+')):
                if i == 0:
                    continue
                plus_words.append(part.split(' ')[0])
            for required_word in plus_words:
                if required_word not in req.split(' '):
                    return 10
        if '"' in sem:
            sem = self._normalize_sentence(sem)
            if sem == req:
                return 0
            else:
                return 10
        elif '[' in sem:
            # TODO: здесь пока будет заглушка (`[]` -- это отстой)
            return 1
        else:
            # Пытаемся примерно предсказать
            # самая подгонистая часть
            s1 = ' '.join(sorted(req.split(' ')))
            s2 = ' '.join(sorted(sem.split(' ')))
            return self.a * cosine(vec_req, vec_sem) + self.b * norm_lev(s1, s2)

    def __init__(self, a=0.7, tokenizer=WordsTokenizer(), n_jobs=multiprocessing.cpu_count(), be_verbose=False, p=1):
        """
        Classifier for requests <--> semantic kernel
        @param a: Weight for cosine distance
        @param tokenizer: Pass existing tokenizer for words, if exists. It would speed up computation
        @param n_jobs: Number of threads to use in execution
        @param be_verbose: Write more output info about progress
        @param p: Minimum threshold for predictions. If none of sentence in semantics kernel has probability more than p,
        then requested phrase is considered as 'Unknown'
        """
        self.a = a
        self.b = 1-a
        self.tokenizer = tokenizer
        self.n_jobs = n_jobs
        self.sem = None
        self.vec_sem = None
        self.n_sem = None
        self.verbose = be_verbose
        self.p = p

    @staticmethod
    def _normalize_sentence(sem):
        return ' '.join([normed_word(re.sub("\W", "", tmp_word).lower()) for tmp_word in sem.split(' ')])

    def train(self, data):
        """
        Trains at array for STRINGS for our purposes
        @param data: data to train at
        @return: nothing
        """
        # Check whether the tokenizer is already trained
        if len(self.tokenizer.uniq_words) == 0:
            self.vec_sem = self.tokenizer.fit_transform(data, data)
        else:
            self.vec_sem = self.tokenizer.transform(data)
        self.sem = data
        # Сохраним в поле класса еще и нормализованную семантику
        self.n_sem = Parallel(n_jobs=self.n_jobs)(delayed(self._normalize_sentence)(sem) for sem in data)
        if self.verbose:
            print("Semantics classifier is trained!")

    def _make_predictions_multithread(self, i, element, vec_req):
        # Нормируем слово
        element = self._normalize_sentence(element)
        # Бегаем по сематич. ядру
        elem_distances = dict(
            zip(self.sem, [self.check_sem(element, sem, vec_req[i], self.vec_sem[j]) for j, sem in enumerate(self.sem)]))
        # Находим ближаюшую к запросу семантику
        nearest_sem = min(elem_distances, key=elem_distances.get)
        # Проверяем: если расстояние меньше порогового, то выдаем найденную сем-ку, иначе 'Unknown'
        if elem_distances[nearest_sem] < self.p:
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
        # Разделим исходные данные порциями по 1000, так лучше параллелится
        for i in range(0, data.shape[0], 1000):
            # Если выпираем, то
            if 1000 + i > data.shape[0]:
                pred_data = data[i:]
            else:
                pred_data = data[i:1000 + i]
            # Распараллелим задачу поиска оптимальной семантики для текущего запроса
            tmp = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(self._make_predictions_multithread)(i, element, vec_req)
                                                   for i, element in enumerate(pred_data))
            predictions += tmp
            if self.verbose:
                print("--------------------------")
                print("Made from {} to {}".format(i, i + 1000))
                print("--------------------------")
        return np.array(predictions)
