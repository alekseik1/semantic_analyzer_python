import pymorphy2
from leven import levenshtein


 # Левенштейн
def damerau_levenshtein_distance(s1, s2):
    return levenshtein(s1, s2)


# Нормированный Левенштейн
def norm_lev(s1, s2):
    return damerau_levenshtein_distance(s1, s2)/max(len(s1), len(s2))


def tanimoto(s1, s2):
    """
    Метрика Танимото. Не очень хорошая
    @param s1: Строка 1
    @param s2: Строка 2
    @return: Расстоние по этой метрике
    """
    a, b, c = len(s1.split()), len(s2.split()), 0.0

    for sym in s1.split():
        if sym in s2.split():
            c += 1

    return c / (a + b - c)


morph = pymorphy2.MorphAnalyzer()
def normed_word(x):
    p = morph.parse(x)[0]
    return p.normal_form
