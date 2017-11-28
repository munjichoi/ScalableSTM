# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 23:35:45 2017

@author: munjichoi
"""
import numpy as np
import scipy as sp
from scipy.special import gammaln
from sklearn import datasets
from pypolyagamma import PyPolyaGamma
import cPickle as pickle
import pdb
#import line_profiler
import STMPG_sampler as stm
from STMPG_gendoc import gen_documents
import sys
old_stdout = sys.stdout

log_file = open("message.txt","w")

sys.stdout = log_file

#####  Modify below for different experiment ##########
#######################################################
D = 400    #number of documents
K = 20      #number of topics
P = 4       #number of covariates
V = 4000     #number of vocabularies

Nd= 300


# {"X":X, "Sigma":Sigma, "B":B, "Phi":Phi, "DocTopic":DocTopic, "DTM":DTM}
simul = gen_documents(D,K,P,V, Nd)
Psi_try = datasets.make_spd_matrix(K-1)
B0_try = np.zeros((P,K-1))
sampler = stm.STM_PG(n_topics=K, beta=2, psi=Psi_try, design_matrix=simul["X"], B0=B0_try, seed=1234)

#pdb.set_trace()
sampler.fit(DTM=simul["DTM"], maxiter=1000, burnin=300)

# with open("Final_result","wb") as f:
#     pickle.dump(Final_result, f)

# with open("True_Values","wb") as f:
#     pickle.dump(simul, f)

with open("result","wb") as f:
    pickle.dump(Final_posterior, f)


sys.stdout = old_stdout

log_file.close()

