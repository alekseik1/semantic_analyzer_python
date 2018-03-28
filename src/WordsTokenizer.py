from TextMetrics import *
import re
import numpy as np
import multiprocessing
from joblib import Parallel, delayed


class WordsTokenizer:
    """
    Class for tokenizing words with handling special characters (e.g. '-', '+')
    """

    def __init__(self, p=0.1, n_jobs=multiprocessing.cpu_count()):
        """
        Create tokenizer object
        @param p: Threshold in transform() method's levenstein distance. Should be about 0.05-0.25
        """
        # NOTE! uniq_words WILL contain 'Unknown'!!!
        self.uniq_words = set()
        self._p = p
        self.n_jobs = n_jobs

    def _get_words(self, sentence):
        """
        Gets unique words from sentence (removing the \W symbols)
        @param sentence: Sentence
        @return: Set of unique words
        """
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
        results = Parallel(n_jobs=self.n_jobs)(delayed(self._get_words)(sentence) for sentence in data)
        for s in results:
            self.uniq_words = self.uniq_words.union(s)
        self.uniq_words = list(self.uniq_words)
        self.uniq_words.append('Unknown')

    def _transfrom_helper(self, sentence, i):
        """
        Main magic for transform() method, used only in class and only for multithreading purposes (don't call it directly)
        @param sentence: one sentence from transform() data
        @param i: number of sentence in data
        @return: list of tuples (i, j). cos_matrix[i][j] should be added by 1 afterwards!!
        """
        to_return = []
        for word_in_sentence in sentence.split(" "):
            # Уберем всякие какашки из слов в запросе (вдруг кто-то решил это делать)
            word_in_sentence = re.sub("\W", "", word_in_sentence)
            word_in_sentence = normed_word(word_in_sentence.lower())
            is_found = False

            for j, word in enumerate(self.uniq_words):
                if norm_lev(word, word_in_sentence) < self._p:
                    # В случае совпадения увеличиваем соответствующее значение в матрице слов на 1
                    to_return.append((i, j))
                    is_found = True
            # Если совсем не нашли значений, то увеличиваем 'Unknown' на единицу
            if not is_found:
                to_return.append((i, -1))
        return to_return

    def transform(self, data):
        """
        Tokenize given data. This data can contain any symbols (e.g. '+', '-') -- they will be removed
        @param data: Data to be vectorized
        @return: numpy n-d array with vectorized sentences in data
        """

        if self.uniq_words is None:
            raise str("Надо сначала вызвать fit() !!!")

        token_matrix = np.zeros((data.shape[0], len(self.uniq_words)), dtype=np.int)
        tmp = Parallel(n_jobs=self.n_jobs)(delayed(self._transfrom_helper)(sentence, i) for i, sentence in enumerate(data))
        # Now, let's fill token_matrix
        for l in tmp:
            for pair in l:
                token_matrix[pair[0]][pair[1]] += 1
        return token_matrix

    def fit_transform(self, train_words, data):
        """
        Call combo of fit() and transform() methods
        @param train_words: Semantic kernel to train at
        @param data: Data to be vectorized
        @return: numpy n-d array with vectorized sentences in data
        """
        self.fit(train_words)
        return self.transform(data)
