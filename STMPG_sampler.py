# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:35:45 2017

@author: munjichoi
"""
import numpy as np
from math import log as ln 
import time
import numbers
import scipy as sp
from scipy.special import gammaln
from sklearn import datasets
import cPickle as pickle
import pdb
#import line_profiler
from STMPG_utils import (check_random_state, dtm_to_lists)
from STMPG_cscript import gibbs_sampler_STM_PG
from pypolyagamma import PyPolyaGamma
#import pyximport; pyximport.install()
#import lda_lda.pyx

######### See the bottom lines for simulation ##########
#  You also need to specify directory path to save the results


class STM_PG:

    def __init__(self, n_topics, beta, psi, design_matrix, B0, nu, seed=None):
        """
        n_topics:       desired number of topics = K
        beta:           hyper-parameter for Dirichlet (topic-contents) 
                        a scalar (FIME: accept vector of size vocab_size)
        psi:            hyper-parameter for inverse Wishart (covariance matrix)
                        positive definite matrix of size (ntopics-1)*(n_topics-1) = (k-1)(k-1)
        design_matrix : size of (n_documents)*(n_covariates) = D*p
        B0 :            hyper-parameter for coefficients. Size (n_covariates)*(n_topics-1) = p(k-1)
        """
        self.n_topics = n_topics 
        self.beta = beta
        self.psi = psi
        self.B0 = B0
        self.X = design_matrix
        self.seed = seed
        self.nu = nu
        rng = check_random_state(self.seed)
        self._rands = rng.rand(1024**2 // 8)  # 1MiB of random variates


    # @profile
    def _initialize(self, DTM):
    	
        t0 = time.time()
        n_docs, n_voca = DTM.shape
        N = int(DTM.sum())
        n_topics = self.n_topics
        self.n_cov = n_cov = self.X.shape[1] 

        self.nzw = nzw = np.zeros((n_topics, n_voca), dtype=np.intc)
        self.ndz = ndz = np.zeros((n_docs, n_topics), dtype=np.intc)
        self.nz = nz = np.zeros(n_topics, dtype=np.intc)

        
        self.term_lookup, self.doc_lookup = term_lookup, doc_lookup = dtm_to_lists(DTM)
        self.topic_lookup = topic_lookup = np.empty_like(self.term_lookup, dtype=np.intc)

        # collection of logistic normal values
        self.eta = np.zeros((n_docs, n_topics)) # per each document
        self.lamb = np.zeros((n_docs, n_topics)) # per each document
        
        # initial values for nu and tau
        # self.nu = nu = n_topics + 5# df for wishart 
        
        # initial values for the global parameter Sigma (covariance matrix)    
        self.Sigma = Sigma = sp.stats.invwishart.rvs(df=self.nu, scale=self.psi, size=1) # covariance matrix
        self.Sigma_inv = Sigma_inv = np.linalg.inv(Sigma) #precision matrix
        self.Lambda0 = Lambda0 = np.identity(n_cov) #Coefficiant row-wise covariance matrix (p*p)
        vec_B0 = np.ravel(self.B0)
        kr_SigLamb_B = np.kron(Sigma , np.linalg.inv(Lambda0)) 
        # how can I do this without inversion?
        vec_B = np.random.multivariate_normal(vec_B0, kr_SigLamb_B)
        self.B = B = vec_B.reshape(self.B0.shape)
     
        for i in range(N):

            w, d = term_lookup[i], doc_lookup[i]
            z_new = i % n_topics
            topic_lookup[i] = z_new
            ndz[d, z_new] += 1
            nzw[z_new, w] += 1
            nz[z_new] += 1

        nd = np.sum(ndz, axis=1)
        vec_eta_mean = np.ravel(np.dot(self.X, B))
        # kr_SigLamb_e = np.kron(Sigma , np.identity(n_docs)) 
        # vec_eta = np.random.multivariate_normal(vec_eta_mean, kr_SigLamb_e)
        vec_eta = vec_eta_mean + np.random.multivariate_normal(np.zeros(vec_eta_mean.shape[0]), np.identity(vec_eta_mean.shape[0]))
        # vec_eta = vec_eta_mean
        self.eta[:,np.arange(n_topics-1)] = vec_eta.reshape((n_docs, n_topics-1))
        pg_rng = PyPolyaGamma(seed=0)
        
        for d in xrange(n_docs):
            for k in xrange(n_topics-1):
               k_indices = np.delete(np.arange(n_topics-1), k)
               eta_sum = np.sum(np.exp(self.eta[d, k_indices]))
               rho_dk = self.eta[d, k] - ln(eta_sum)
               self.lamb[d, k] = pg_rng.pgdraw(nd[d], rho_dk)

        
        # self.lamb[:,np.arange(n_topics-1)] = np.repeat(pg_rng.pgdraw(1, 0), n_docs*(n_topics-1)).reshape((n_docs, n_topics-1))
        t1 = time.time()

        print("initialization done, time consumed=", t1-t0)

        
        

    #DTM = scipy sparse matrix
    
    #@profile
    def fit(self, DTM, maxiter, burnin):
        """
        Run the Gibbs sampler.
        """
        self.n_docs, self.n_terms = DTM.shape
        rands = self._rands.copy()
        self.n_iter = maxiter
        self.burnin = burnin
        beta = np.repeat(self.beta, self.n_terms).astype(np.float64)

        #initialization
        self._initialize(DTM)
        initial = {"ini_eta":self.eta, "ini_B":self.B, "ini_Sigma":self.Sigma}

        with open("initial","wb") as f:
            pickle.dump(initial, f)
        self.loglikelihoods = np.empty(self.n_iter, dtype=np.float64, order='C')
        self.post_z, self.post_eta, self.post_lamb, self.post_B, \
        self.post_Sigma, self.loglikelihoods = \
        gibbs_sampler_STM_PG(self.n_iter, self.burnin, 10, self.n_topics, self.n_docs, self.n_terms, self.n_cov,
                         self.X, self.psi, self.Sigma, 
                         self.Sigma_inv, self.B0, self.Lambda0, 
                         self.lamb, self.eta, self.B,
                         beta, self.nu,
                         self.doc_lookup, self.term_lookup, self.topic_lookup,
                         self.ndz, self.nzw, self.nz,
                         self.seed, rands)
