from TextMetrics import *
import re
import numpy as np

class WordsTokenizer:


    def __init__(self):
        self._cos_matrix = None
        self.uniq_words = None

    def fit(self, data):
        _uniq_words = set()
        for sentence in data:
            for word in sentence.split():
                word = normed_word(re.sub("\W", "", word)).lower()
                _uniq_words.add(word)
        _uniq_words = list(_uniq_words)
        self.uniq_words = _uniq_words
        self.uniq_words += ['Unknown']

    def transform(self, data):
        if self.uniq_words is None:
            raise "Надо сначала вызвать fit() !!!"

        _cos_matrix = np.zeros((data.shape[0], len(self.uniq_words)+1))
        for i, sentence in enumerate(data):
            for word_in_sentence in sentence.split(" "):
                # Уберем всякие какашки из слов в запросе (вдруг кто-то решил это делать)
                word_in_sentence = re.sub("\W", "", word_in_sentence)
                word_in_sentence = normed_word(word_in_sentence.lower())

                p = 0.1
                is_found = False
                for j, word in enumerate(self.uniq_words):
                    if norm_lev(word, word_in_sentence) < p:
                        _cos_matrix[i][j] += 1
                        is_found = True
                        break
                if not is_found:
                    _cos_matrix[i][-1] += 1
        return _cos_matrix

    def fit_transform(self, train_words, data):
        self.fit(train_words)
        return self.transform(data)
