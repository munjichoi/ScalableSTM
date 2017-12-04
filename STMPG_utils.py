from __future__ import absolute_import, unicode_literals  # noqa

import logging
import numbers
import sys

import numpy as np


def check_random_state(seed):
    if seed is None:
        # i.e., use existing RandomState
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("{} cannot be used as a random seed.".format(seed))

def dtm_to_lists(dtm):
    """Convert a (sparse) matrix of counts into arrays of word and doc indices

    Parameters
    ----------
    dtm : array or sparse matrix (D, V)
        document-term matrix of counts

    Returns
    -------
    (WS, DS) : tuple of two arrays
        term_lookup[k] contains the kth word in the corpus
        doc_lookup [k] contains the document index for the kth word

    """
    if np.count_nonzero(dtm.sum(axis=1)) != dtm.shape[0]:
        logger.warning("all zero row in document-term matrix found")
    if np.count_nonzero(dtm.sum(axis=0)) != dtm.shape[1]:
        logger.warning("all zero column in document-term matrix found")
    sparse = True
    try:
        # if dtm is a scipy sparse matrix
        dtm = dtm.copy().tolil()
    except AttributeError:
        sparse = False

    if sparse and not np.issubdtype(dtm.dtype, int):
        raise ValueError("expected sparse matrix with integer values, found float values")

    docs, terms = np.nonzero(dtm)
    if sparse:
        ss = tuple(dtm[i, j] for i, j in zip(docs, terms))
    else:
        ss = dtm[docs, terms]

    n_tokens = int(dtm.sum())
    #doc_lookup = np.repeat(docs, ss).astype(np.int32)
    doc_lookup = np.repeat(docs, ss).astype(np.intc)
    print doc_lookup.dtype
    term_lookup = np.empty(n_tokens, dtype=np.intc)
    startidx = 0
    for i, cnt in enumerate(ss):
        cnt = int(cnt)
        term_lookup[startidx:startidx + cnt] = terms[i]
        startidx += cnt
    return term_lookup, doc_lookup

