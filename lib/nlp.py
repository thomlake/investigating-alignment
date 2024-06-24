import string
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize as _word_tokenize


class Context:
    _stop_words = None
    _stemmer = None

    @property
    def stop_words(self) -> set[str]:
        if self._stop_words is None:
            self._stop_words = set(stopwords.words('english'))

        return self._stop_words

    @property
    def stemmer(self) -> PorterStemmer:
        if self._stemmer is None:
            self._stemmer = PorterStemmer()

        return self._stemmer


_CONTEXT = Context()


@lru_cache
def word_tokenize(string: str):
    return _word_tokenize(string)


def ngrams(s: list[str], n: int = 1) -> list[tuple[str, ...]]:
    if n < 1:
        raise ValueError('n must be >= 1')
    if n == 1:
        return s
    if len(s) < n:
        return [tuple(s)]
    return list(zip(*[s[i:] for i in range(n)]))


def is_stop_word(w: str) -> str:
    if all(c in string.punctuation for c in w):
        return True

    return w in _CONTEXT.stop_words


def get_stem(w: str) -> str:
    return _CONTEXT.stemmer.stem(w)


def word_bag(
        text: str,
        n: int = 1,
        stem: bool = False,
        remove_stop_words: bool = True,
) -> list[str]:
    words = word_tokenize(text)
    if remove_stop_words:
        words = [w for w in words if not is_stop_word(w)]

    if stem:
        words = [get_stem(w) for w in words]

    if n > 1:
        words = ngrams(words, n=n)

    return list(set(words))
