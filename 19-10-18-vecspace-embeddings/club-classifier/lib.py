from typing import Iterable, List, Any, Tuple, Dict, NamedTuple
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from vocab import Vocabulary
from collections import Counter, defaultdict, namedtuple
import gc
import json
import numpy as np


DocumentPart = NamedTuple(
    "Document",
    [('words', List[int]),
     ('weight', float)])


class QueryEngine(object):
    def __init__(self):
        # Read stoplist.
        with open('stoplist.txt') as f:
            self.stoplist = f.read().strip().split()
            self.stoplist = [w.strip() for w in self.stoplist]
            self.stoplist = set(self.stoplist)

        # Read GLOVE vectors.
        self.glove_vectors = np.zeros((400000, 50))
        self.glove_vocabulary = Vocabulary()
        with open('glove.6B.50d.txt') as f:
            for i, line in enumerate(f):
                args = line.split()
                word = args[0]
                self.glove_vectors[i] = [float(x) for x in args[1:]]
                i1 = self.glove_vocabulary.update(word)
                assert i1 == i

        self.word_vocabulary = Vocabulary()
        self.word_vocabulary.update('__OOV__')

        self.documents = []  # type: List[List[DocumentPart]]

    def _word_ids(self, query: str, update: bool=False) -> List[int]:
        words = [w.lower() for w in word_tokenize(query)
                 if w not in self.stoplist]
        if update:
            return [self.word_vocabulary.update(w) for w in words]
        else:
            words = [self.word_vocabulary.get(w) for w in words]
            words = [0 if w == -1 else w for w in words]
            return words

    def add_document(self, parts: Iterable[Tuple[str, float]]) -> int:
        words = [DocumentPart(self._word_ids(part, update=True), weight)
                 for part, weight in parts]
        self.documents.append(words)
        return len(self.documents) - 1

    def _tfidf(self, words: Iterable[int]) -> Dict[int, float]:
        counts = Counter(words)
        result = {}
        for w, tf in counts.items():
            df = len(self.word_documents[w]) + 1
            N = len(self.documents)
            result[w] = tf * np.log(N/df)

        z = np.sqrt(sum(w ** 2 for k, w in result.items()))
        for k in list(result.keys()):
            result[k] = result[k] / z
        return result

    def _tfidf_doc(self, document: List[DocumentPart]
                   ) -> Dict[int, float]:
        counts = defaultdict(float)  # type: Dict[int, float]

        for part, weight in document:
            for w in part:
                counts[w] += weight

        result = {}
        for w, tf in counts.items():
            df = len(self.word_documents[w]) + 1
            N = len(self.documents)
            result[w] = tf * np.log(N/df)

        z = np.sqrt(sum(w ** 2 for k, w in result.items()))
        for k in list(result.keys()):
            result[k] = result[k] / z
        return result

    def _tfidf_vec(self, tfidf_dict: Dict[int, float]) -> np.ndarray:
        vec = np.zeros(50)
        for k, w in tfidf_dict.items():
            wn = self.word_vocabulary.words[k]
            wi = self.glove_vocabulary.get(wn)
            if wi == -1:
                continue
            vec += w * self.glove_vectors[wi]
        return vec / np.linalg.norm(vec)

    def recompute_stats(self) -> None:
        self.word_documents = [set() for _ in range(len(self.word_vocabulary))]  # type: List[Set[int]]
        self.document_words_tfidf = []
        self.document_vectors = []

        for i, parts in enumerate(self.documents):
            for part, weight in parts:
                for w in part:
                    self.word_documents[w].add(i)

        for i, document in enumerate(self.documents):
            tfidf = self._tfidf_doc(document)
            self.document_words_tfidf.append(tfidf)
            self.document_vectors.append(self._tfidf_vec(tfidf))

    def query(self, sentence: str, top: int=-1) -> List[Tuple[int, Dict[str, float]]]:
        words = self._word_ids(sentence)

        query_tfidf = self._tfidf_doc([DocumentPart(words, 1.0)])
        query_vec = self._tfidf_vec(query_tfidf)

        result = []  # type: List[Tuple[int, Dict[str, float]]]

        for i, doc_tfidf in enumerate(self.document_words_tfidf):
            doc_vec = self.document_vectors[i]
            glove_score = -np.linalg.norm(query_vec - doc_vec)

            key_set = set(doc_tfidf.keys()) & set(query_tfidf.keys())
            tfidf_score = sum(query_tfidf[k] * doc_tfidf[k] for k in key_set)

            result.append((i, {
                "tfidf": tfidf_score,
                "glove": glove_score }))

        result.sort(key=lambda t: -t[1]['glove'])
        return result
