import numpy as np
import scipy as sp
from scipy.special import gammaln
from sklearn import datasets
from pypolyagamma import PyPolyaGamma
import cPickle
import pdb

D = 200 	#number of documents
K = 10 		#number of topics
P = 4		#number of covariates
V = 100		#number of vocabularies


def gen_documents(N_documents, N_topics, N_covariates, N_voca):
	
	Psi = datasets.make_spd_matrix(N_topics-1)
	nu  = N_topics+1
	B0  = np.zeros((N_covariates,N_topics-1))
	beta = 3
	
	X = np.random.normal(0,1, N_documents*N_covariates).reshape((N_documents, N_covariates))
	Nd = np.random.poisson(200, size=N_documents)

	#Global Parameters
	Sigma = sp.stats.invwishart.rvs(df=nu, scale=Psi, size=1) # covariance matrix
	vec_B0 = np.ravel(B0)
	kr_SigLamb = np.kron(Sigma , np.identity(N_covariates)) 
	# how can I do this without inversion?
	B = np.random.multivariate_normal(vec_B0, kr_SigLamb).reshape(B0.shape)
	XB = np.dot(X, B)

	Phi = np.random.dirichlet(alpha=np.repeat(beta, N_voca), size=N_topics)

	DTM = np.zeros((N_documents, N_voca))
	DocTopic = []

	#document level parameters
	for d in xrange(N_documents):
		eta = np.append(XB[d, :], 0)
		theta = np.exp(eta)/sum(np.exp(eta))
		WordTopic = np.zeros(Nd[d])
		for i in xrange(Nd[d]):
			z = np.random.multinomial(1, theta).nonzero()[0]   # zth topic among K
			w = np.random.multinomial(1, Phi[z].reshape(N_voca)).nonzero()[0] # vth word among V
			DTM[d, w] += 1
			WordTopic[i] = z

		DocTopic.append(WordTopic)


	result = {"X":X, "Sigma":Sigma, "B":B, "Phi":Phi, "DocTopic":DocTopic, "DTM":DTM}
	return result

#pdb.set_trace()  




