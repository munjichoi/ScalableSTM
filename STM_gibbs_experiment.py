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
import cPickle
import pdb


######### See the bottom lines for simulation ##########
#  You also need to specify directory path to save the results




def gen_documents(N_documents, N_topics, N_covariates, N_voca):
	
	Psi = datasets.make_spd_matrix(N_topics-1)
	nu  = N_topics+1
	B0  = np.zeros((N_covariates,N_topics-1))
	beta = 3
	
	X = np.random.normal(0,1, N_documents*N_covariates).reshape((N_documents, N_covariates))
	Nd = np.random.poisson(1000, size=N_documents)

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




def dtm_to_words(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. (DTM matrix) The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    (repeat the indices of the word as many times as the word happens in the given document)
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx


class LdaSampler(object):

    def __init__(self, n_topics, beta, psi, design_matrix, B0):
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

    def _initialize(self, DTM):
    	
        n_docs, vocab_size = DTM.shape

        
        
        # collection of logistic normal values
        self.eta = np.zeros((n_docs, self.n_topics)) # per each document
        self.lamb = np.zeros((n_docs, self.n_topics)) # per each document
        # topic of each word in the entire corpus
        self.topics = {}
        
        # number of times topic z occurs in document d
        self.ndk = np.zeros((n_docs, self.n_topics))
        # number of times word w is assigned to topic z
        self.nkv = np.zeros((self.n_topics, vocab_size))
        # number of total words in document d (Nd) = document length
        self.nd = np.zeros(n_docs)
        # number of total words assigned to topic Z 
        self.nk = np.zeros(self.n_topics)

        
        # initial values for nu and tau
        self.nu = self.n_topics + 5# df for wishart 
        
        # initial values for the global parameter Sigma (covariance matrix)    
        self.Sigma = sp.stats.invwishart.rvs(df=self.nu, scale=self.psi, size=1) # covariance matrix
        self.Sigma_inv = np.linalg.inv(self.Sigma) #precision matrix
        self.Lambda0 = np.identity(self.X.shape[1]) #Coefficiant row-wise covariance matrix (p*p)
        vec_B0 = np.ravel(self.B0)
        kr_SigLamb = np.kron(self.Sigma , np.linalg.inv(self.Lambda0)) 
        # how can I do this without inversion?
        vec_B = np.random.multivariate_normal(vec_B0, kr_SigLamb)
        self.B = vec_B.reshape(self.B0.shape)
        self.pg= PyPolyaGamma(seed=0)

        for d in xrange(n_docs):
            
            # document level parameters - topic-prevalence eta
            eta_d_mean = np.dot(self.X[d, :], self.B)
            eta_d = np.random.multivariate_normal(eta_d_mean, self.Sigma) 
            eta_d = np.append(eta_d, 0)
            self.eta[d, :] = eta_d

            
            self.lamb[d, np.arange(self.n_topics-1)] = np.repeat(self.pg.pgdraw(1, 0), self.n_topics-1)
            self.lamb[d, self.n_topics-1] = 0

            # i is a number between 0 and doc_length-1
            # v is a number between 0 and vocab_size-1
            for i, v in enumerate(dtm_to_words(DTM[d, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                # Return random integers from the “discrete uniform” 
                self.ndk[d,z] += 1
                self.nd[d] += 1
                self.nkv[z,v] += 1
                self.nk[z] += 1
                self.topics[(d,i)] = z

        """
        This calculates posterior p(z_dn|...) (K dimensional multinomial parameter)
        Use the result of this function to sample z. 
     	Then add 1 to corresponding frequency matrices
        """
    def _conditional_dist_topics(self, d, i, v):
        """
        d : index of documents (0~D-1)
        i : index of words in a document (0~Nd-1)
        v : which voca word_di is 
        Conditional distribution of Z (vector of size n_topics = K).
        """
        
        vocab_size = self.nkv.shape[1]
        
        z = self.topics[(d,i)] #topic of w_di
        self.ndk[d,z] -= 1
        self.nd[d] -= 1
        self.nkv[z,v] -= 1
        self.nk[z] -= 1
        
        left = (self.nkv[:,v] + self.beta) / \
               (self.nk + self.beta * vocab_size)
        right = np.exp(self.eta[d, :] )/sum(np.exp(self.eta[d, :]))
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

        """
        This calculates mean and variance of eta_dk posterior (univariate normal) 
        & polygamma variable lambda_dk
        """
    def _conditional_dist_logitnormal(self, d, k):
        """
        Conditional distribution of eta_dk (size of k-1 = kth elements is fixed to be zero)
        This part is applied to k in (0~n_topics-2)
        """
        
    	XB = np.dot(self.X, self.B)
    	eta_d_k = self.eta[d, np.arange(self.n_topics)!=k]
        #remove last column with zero
        eta_d_k = eta_d_k[np.arange(len(eta_d_k)-1)]
    	#conditional dist of multivariate normal
    	sigmasq_k = 1/self.Sigma_inv[k,k]
    	mu_dk = XB[d, k] - sigmasq_k*\
    		np.dot(self.Sigma_inv[np.arange(self.n_topics-1)!=k, np.arange(self.n_topics-1)!=k], eta_d_k - XB[d, np.arange(self.n_topics-1)!=k])
    	kappa_dk = self.ndk[d,k] - self.nd[d]/2
    	zeta_dk =np.log(sum(np.exp(eta_d_k))) 

    	# mean / var of eta_dk's posterior distribution (gamma, tau)
    	tausq_dk = 1/(self.lamb[d,k] + 1/sigmasq_k)
    	gamma_dk = tausq_dk*(kappa_dk + self.lamb[d,k]*zeta_dk + mu_dk/sigmasq_k)

    	# Polyagamma parameters
    	rho_dk = self.eta[d,k] - zeta_dk
        
        return {"tausq_dk":tausq_dk, "gamma_dk":gamma_dk, "rho_dk":rho_dk, \
        "eta_d_k":eta_d_k, "mu_dk":mu_dk, "kappa_dk":kappa_dk, "zeta_dk":zeta_dk}
        
    	# Don't forget to append kth element of eta_dk with zero


        """
        This calculates posterior mean and cov of B (matrix multivariate normal)
        """

    def _conditional_dist_topicprev(self):
        """
        <part1>
        Conditional distribution of B (a matrix size of (n_covariates)*(n_topics -1) = p(k-1)).
        Draw samples in vectorized form using multivariate normal distribution
        <part2>
        Posterior distribution of Sigma (wishart)
        """
        Mean_newB = np.dot(self.Lambda0, self.B0) + \
        	np.dot(self.X.T, self.eta[: ,np.arange(self.n_topics-1)])
        Mean_newB = np.dot(self.Lambda_n_inv, Mean_newB)

        resid_eta = self.eta[: ,np.arange(self.n_topics-1)]-np.dot(self.X, Mean_newB)
        resid_Bn = Mean_newB - self.B0 

        Scale_newSigma =  resid_eta.T.dot(resid_eta) + np.dot(np.dot(resid_Bn.T, self.Lambda0), resid_Bn) + self.psi
        
        return {"Mean_newB":Mean_newB, "Scale_newSigma":Scale_newSigma}

    def run(self, DTM, maxiter, burnin):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = DTM.shape

        self._initialize(DTM)

        # Quantities for Global Parameters
        self.XtX = np.dot(self.X.T, self.X)
        self.Lambda_n = self.Lambda0 + self.XtX
        self.Lambda_n_inv = np.linalg.inv(self.Lambda_n)
        self.df_newSigma = self.nu + n_docs

        self.post_z = []
        self.post_eta = []
        self.post_lamb = []
        self.post_B = []
        self.post_Sigma = []



        for it in xrange(maxiter):
            print it
            for d in xrange(n_docs):
                print d
                #document-word level parameters
                for i, v in enumerate(dtm_to_words(DTM[d, :])):
                    # i is a number between 0 and doc_length-1
                    # v is a number between 0 and vocab_size-1
                    
                    #update z_di
                    p_z = self._conditional_dist_topics(d, i, v)
                    z = np.random.multinomial(1, p_z).nonzero()[0]

                    self.ndk[d,z] += 1
                    self.nd[d] += 1
                    self.nkv[z,v] += 1
                    self.nk[z] += 1
                    self.topics[(d,i)] = z

                #document-topic level parameters
                for k in xrange(self.n_topics-2):                    
                #update eta_dk, lambda_dk
                    logitnormal = self._conditional_dist_logitnormal(d, k)
                    self.eta[d, k] = np.random.normal(logitnormal["gamma_dk"], logitnormal["tausq_dk"])
                    self.lamb[d, k] = self.pg.pgdraw(self.nd[d], logitnormal["rho_dk"])

                self.eta[d, self.n_topics-1] = 0
                self.lamb[d, self.n_topics-1] = 0

            #global parameter
            BSigma_post = self._conditional_dist_topicprev()
            vec_Bn = np.ravel(BSigma_post["Mean_newB"])
            kr_SigLamb_new = np.kron(self.Sigma , self.Lambda_n_inv)
            self.B  = np.random.multivariate_normal(vec_Bn, kr_SigLamb_new).reshape(self.B0.shape)
            self.Sigma = sp.stats.invwishart.rvs(df=self.df_newSigma, scale=BSigma_post["Scale_newSigma"], size=1) 
            self.Sigma_inv = np.linalg.inv(self.Sigma)

            if it <= burnin:
            	pass
            elif it > burnin:
            	self.post_z.append(self.topics)
            	self.post_eta.append(self.eta)
            	self.post_lamb.append(self.lamb)
            	self.post_B.append(self.B)
            	self.post_Sigma.append(self.Sigma)

#####  Modify below for different experiment ##########
#######################################################
D = 200     #number of documents
K = 10      #number of topics
P = 4       #number of covariates
V = 200     #number of vocabularies


# {"X":X, "Sigma":Sigma, "B":B, "Phi":Phi, "DocTopic":DocTopic, "DTM":DTM}
simul = gen_documents(D,K,P,V)
Psi_try = datasets.make_spd_matrix(K-1)
B0_try = np.zeros((P,K-1))
sampler = LdaSampler(n_topics=K, beta=2, psi=Psi_try, design_matrix=simul["X"], B0=B0_try)

#pdb.set_trace()
Final_posterior = sampler.run(DTM=simul["DTM"], maxiter=200000, burnin=10000)

Final_posterior = (sampler.post_z, sampler.post_eta, sampler.post_lamb, sampler.post_B, sampler.post_Sigma)

with open("home/munjic/ScalableSTM","wb") as f:
    pickle.dump(Final_posterior, f)

