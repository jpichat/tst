from operator import itemgetter
from more_itertools import circular_shifts
import numpy as np
import typing as t


class Patterns:  # no-qa
    """circular permutation-invariant probability of occurrence of L-sequences in a series"""

    def __init__(self, size: int, disable_max_seq_len=False):
        if not disable_max_seq_len:
            assert size < 13, "sequence length should be less than 12"
        self.size = size  # size of sequences to analyse
        self.seq_ = None
        self.major_seq_ = None
        self.major_seq_prob_ = None
        self.occurences_ = None

    def find(
        self,
        x: t.Union[np.ndarray, list],
        weighted: bool = False,
    ):
        if not isinstance(x, np.ndarray):
            assert isinstance(x, list), "expects lists or numpy arrays as inputs"
            x = np.array(x)
        X = embed_np(x, self.size)
        temp = np.unique(X, axis=0)
        self.seq_ = self.c3po(temp)

        C = np.zeros(len(self.seq_))
        W = 0
        for row in X:
            w = np.sum((row - np.mean(row)) ** 2) / self.size if weighted else 1
            for k, p in enumerate(self.seq_):
                if self.is_circular(row, p):
                    C[k] += w
            W += w

        self.occurences_ = sorted(
            list(zip(self.seq_, C / W)), key=itemgetter(1), reverse=True
        )
        del X, C, temp

        if self.major_seq_is_unique:
            self.major_seq_ = self.occurences_[0][0]
            self.major_seq_prob_ = self.occurences_[0][1]

        return self

    @property
    def sequences(self):
        """up to circular permutations"""
        return self.seq_

    @property
    def num_seq(self):
        return len(self.sequences)

    @property
    def major_seq(self):  # no-qa
        return self.major_seq_

    @property
    def major_seq_prob(self):
        return self.major_seq_prob_

    @property
    def occurences(self):  # no-qa
        try:
            return self.occurences_
        except AttributeError:
            return []

    @property
    def major_seq_is_unique(self):
        """check probabilistic uniqueness of dominant pattern"""
        if len(self.occurences_) > 1 and np.isclose(
            self.occurences_[0][1], self.occurences_[1][1], 5e-2
        ):
            return False
        else:
            return True

    @property
    def is_regular(self):  # no-qa
        """checks whether there is a non-random pattern"""
        if self.major_seq_prob is None:
            return False
        if self.major_seq_prob == 1.0:
            return True
        else:
            return self.major_seq_prob > 1.0 / self.num_seq

    @staticmethod
    def c3po(list_or_arrays):
        """circular-permutation-proof patterns"""
        x = []
        P = []
        for row in list_or_arrays:
            if tuple(row) not in P:
                x.append(row)
                P.extend(circular_shifts(row))
        del P
        return x

    @staticmethod
    def is_circular(arr1, arr2):
        """whether arr1 and arr2 are circular permutations of one another"""
        if len(arr1) != len(arr2):
            return False
        str1 = " ".join(map(str, arr1))
        str2 = " ".join(map(str, arr2))
        if len(str1) != len(str2):
            return False
        return str1 in str2 + " " + str2
