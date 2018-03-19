from TextMetrics import *
import re
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


class WordsTokenizer:
    """
    Class for tokenizing words with handling special characters (e.g. '-', '+')
    """

    def __init__(self, p=0.1):
        """
        Create tokenizer object
        @param p: Threshold in transform() method's levenstein distance. Should be about 0.05-0.25
        """
        self._cos_matrix = None
        self.uniq_words = set()
        self._p = p

    def _get_words(self, sentence):
        _uniq_words = set()
        for word in sentence.split():
            word = normed_word(re.sub("\W", "", word)).lower()
            _uniq_words.add(word)
        return _uniq_words

    def fit(self, data):
        """
        Fit tokenizer with given semantic kernel
        @param data: Semantic kernel to train at
        @return: Nothing
        """
        self.uniq_words = set()
        num_cores = multiprocessing.cpu_count()
        results = Parallel(n_jobs=num_cores)(delayed(self._get_words)(sentence) for sentence in data)
        for s in results:
            self.uniq_words = self.uniq_words.union(s)
        self.uniq_words = list(self.uniq_words)
        self.uniq_words.append('Unknown')

    def transform(self, data):
        """
        Encode
        @param data: Data to be vectorized
        @return: numpy n-d array with vectorized sentences in data
        """
        if self.uniq_words is None:
            raise str("Надо сначала вызвать fit() !!!")

        _cos_matrix = np.zeros((data.shape[0], len(self.uniq_words)+1))
        for i, sentence in enumerate(data):
            for word_in_sentence in sentence.split(" "):
                # Уберем всякие какашки из слов в запросе (вдруг кто-то решил это делать)
                word_in_sentence = re.sub("\W", "", word_in_sentence)
                word_in_sentence = normed_word(word_in_sentence.lower())

                is_found = False
                for j, word in enumerate(self.uniq_words):
                    if norm_lev(word, word_in_sentence) < self._p:
                        _cos_matrix[i][j] += 1
                        is_found = True
                        break
                if not is_found:
                    _cos_matrix[i][-1] += 1
        return _cos_matrix

    def fit_transform(self, train_words, data):
        """
        Call combo of fit() and transform() methods
        @param train_words: Semantic kernel to train at
        @param data: Data to be vectorized
        @return: numpy n-d array with vectorized sentences in data
        """
        self.fit(train_words)
        return self.transform(data)
