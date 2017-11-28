#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
"""
The heavy-lifting is here in cython.

Draws from Allen Riddell's LDA library https://github.com/ariddell/lda
"""

from datetime import (datetime, timedelta)
import numpy as np

import scipy as sp
from sklearn import datasets
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from libc.stdio cimport printf
from cython cimport view
from cython.operator cimport (preincrement, predecrement)
from cython_gsl cimport (gsl_sf_lngamma as lngamma, gsl_sf_log as ln, gsl_rng, gsl_rng_type, 
                         gsl_sf_exp as exp, 
                         gsl_rng_mt19937, gsl_rng_env_setup,
                         gsl_rng_default,
                         gsl_rng_alloc, gsl_rng_set,
                         gsl_rng_uniform, gsl_rng_uniform_int,
                         gsl_ran_gaussian as gaussian)
from pypolyagamma import PyPolyaGamma



# cdef gsl_rng_type * T
# cdef gsl_rng * r
# gsl_rng_env_setup()

# T = gsl_rng_default
# r = gsl_rng_alloc (T)

cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)
# gsl_rng_set(r, 1234)


cdef print_progress(start_time, int n_report_iter, int i,
                    double likeli_now, double likeli_last):
    """
    Print progress of iterations.
    """

    if i > 0 and i % n_report_iter == 0:
        now_time = datetime.now()
        print('{} {} elapsed, iter {:>4}, LL {:.4f}, {:.2f}% change from last'
            .format(now_time,
                    now_time - start_time,
                    i,
                    likeli_now,
                    (likeli_now - likeli_last) / fabs(likeli_last) * 100))

cdef int searchsorted(double[:] arr, int length, double value) nogil:
    """Bisection search (c.f. numpy.searchsorted)

    Find the index into sorted array `arr` of length `length` such that, if
    `value` were inserted before the index, the order of `arr` would be
    preserved.
    """
    cdef int imin, imax, imid
    imin = 0
    imax = length
    while imin < imax:
        imid = imin + ((imax - imin) >> 2)
        if value > arr[imid]:
            imin = imid + 1
        else:
            imax = imid
    return imin

#make doc-lookup and term-lookup as just one dim vector

def gibbs_sampler_STM_PG(int n_iter, int burnin, int n_report_iter,
                         int n_topics, int n_docs, int n_terms, int n_cov, 
                         double[:,:] X, double[:,:] psi, double[:,:] Sigma, 
                         double[:,:] Sigma_inv, double[:,:] B0, double[:,:] Lambda0, 
                         double[:,:] lamb, double[:,:] eta, double[:,:] B,
                         double[:] beta, int nu,
                         int[:] doc_lookup, int[:] term_lookup, int[:] topic_lookup,
                         int[:,:] ndz, int[:,:] nzw, int[:] nz,
                         int seed, double[:] rands) : 
    
    # Qunatities for Global parameter

    cdef:
        double[::1] ln_likeli = np.empty(n_iter, dtype=np.float64, order='C')
        double[:,::1] Lambda_n = Lambda0 + np.dot(X.T, X)
        double[:,::1] LaB0 = np.dot(Lambda0, B0)
        double[:,::1] Lambda_n_inv = np.linalg.inv(Lambda_n)
        double[:,::1] Lninv_LaB0 = np.dot(Lambda_n_inv, LaB0)
        double[:, :] Xt = X.T
        int df_newSigma = nu + n_docs
        int[::1] nd = np.sum(ndz, axis=1, dtype=np.intc)
        int n_tokens = term_lookup.shape[0]
        double[::1] likelihood = np.empty(n_iter, dtype=np.float64, order='C')

    # cdef: 
    #     double* post_z = <double*> malloc(n_iter - burnin * sizeof(double))
    #     double* post_eta = <double*> malloc(n_iter - burnin * sizeof(double))
    #     double* post_lamb = <double*> malloc(n_iter - burnin * sizeof(double))
    #     double* post_B = <double*> malloc(n_iter - burnin * sizeof(double))
    #     double* post_Sigma = <double*> malloc(n_iter - burnin * sizeof(double))

    post_z = []
    post_eta = []
    post_lamb = []
    post_B = []
    post_Sigma = []


    cdef:
        int i, k, w, d, z, z_new, it, p, j
        double sigmasq_k, mu_dk, kappa_dk, zeta_dk, tausq_dk, gamma_dk, rho_dk, eta_sum
        double r_new, dist_cum
        int n_rand = rands.shape[0]
        double beta_sum = 0
        double lbeta_w = 0
        #double* dist_sum = <double*> malloc(n_topics * sizeof(double))
        double[::1] dist_sum = np.empty(n_topics, dtype=np.float64, order='C')
        double[:,::1] XB = np.dot(X, B)
        #eta_d_k = np.empty(n_topics-2, dtype=np.float64, order='C')
        #double[:,::1] Sigma_inv_sub= np.empty((n_topics-2,n_topics-2), dtype=np.float64, order='C')
        #double[::1] XB_sub = np.empty(n_topics-2, dtype=np.float64, order='C')
        double[::1] eXB = np.empty(n_topics-1, dtype=np.float64, order='C')
        double dot_Sig_eXB = 0
        double[:,::1] Xteta = np.empty((n_cov, n_topics-1), dtype=np.float64, order='C')
        double[:,::1] eta_sub = np.empty((n_docs, n_topics-1), dtype=np.float64, order='C')
        

    
    for w in range(n_terms):
        beta_sum += beta[w]
        lbeta_w += lngamma(beta[w])

    pg_rng = PyPolyaGamma(seed=seed)

    start_time = datetime.now()
    print('{} start iterations'.format(start_time))

    for it in range(n_iter):
        #print('{} iter'.format(it))
        printf("iter is %d \n", it)
        # sample z
        for i in range(n_tokens):
            #print('{} n_token'.format(i))
            
            w = term_lookup[i]  # ith word
            d = doc_lookup[i]   # ith word's document
            z = topic_lookup[i] # ith word's topic

            predecrement(nzw[z, w])
            predecrement(ndz[d, z])
            predecrement(nz[z])

            
            

            dist_cum = 0.
            for k in range(n_topics):
                #printf("k is %d \n", k)
                # eta is a double so cdivision yields a double
                dist_cum += (nzw[k, w] + beta[w]) / (nz[k] + beta_sum) * exp(eta[d, k])
                dist_sum[k] = dist_cum

            r_new = rands[i % n_rand] * dist_cum  # dist_cum == dist_sum[-1]
            z_new = searchsorted(dist_sum, n_topics, r_new)

            topic_lookup[i] = z_new
            preincrement(nzw[z_new, w])
            preincrement(ndz[d, z_new])        
            preincrement(nz[z_new])
        
        #print("n_token step ok \n")

        # sample logit normal
        for d in xrange(n_docs):
            # print("doc is {}".format(d))
            # kth eta always held at zero for identifiability
            for k in xrange(n_topics-1):
                # print("topic is {}".format(k))
                
                #fix k_indices
                k_indices = np.delete(np.arange(n_topics-1), k)
                # print("k_indices={}".format(k_indices))
                eta_sum = 0.
                dot_Sig_eXB = 0.

                eXB = np.empty(n_topics-1, dtype=np.float64, order='C')
                for i in k_indices:
                    # print("i={}".format(i))
                    eta_sum += exp(eta[d, i])
                    # print("eta_sum={}".format(eta_sum))
                    eXB[i] = eta[d, i] - XB[d, i] #fix initial values for eta
                    # print("eta[d, i]={}".format(eta[d, i]))
                    # print("XB[d, i]={}".format(XB[d, i]))
                    # print("eXB[i]={}".format(eXB[i]))

                for j in k_indices:
                    dot_Sig_eXB += Sigma_inv[k, j]*eXB[j]
                    # print("j={}".format(j))
                    # print("Sigma_inv[k, j]={}".format(Sigma_inv[k,j]))
                    # print("dot_Sig_eXB={}".format(dot_Sig_eXB))


                #print("eta part runs")

                sigmasq_k = 1/Sigma_inv[k,k]

                #printf("sigmasq_k is %f \n", sigmasq_k)

                mu_dk = XB[d, k] - sigmasq_k*dot_Sig_eXB
                #printf("mu_dk for doc %d, topic %d is %f \n", d, k, mu_dk)

                kappa_dk = ndz[d,k] - nd[d]/2

                zeta_dk =ln(eta_sum)
                
                #printf("zeta_dk is %f \n", zeta_dk) 

                # mean / var of eta_dk's posterior distribution (gamma, tau)
                tausq_dk = 1/(lamb[d,k] + 1/sigmasq_k)
                #printf("tausq_dk for doc %d, topic %d is %f \n", d, k, tausq_dk) 
                gamma_dk = tausq_dk*(kappa_dk + lamb[d,k]*zeta_dk + mu_dk/sigmasq_k)
                
                #print("doc is {}".format(d))
                #print("topic is {}".format(k))
                # print("eta_sum = {}".format(eta_sum))
                # print("max eta_d = {}".format(max(eta[d, :])))
                # print("sigmasq_k = {}".format(sigmasq_k))
                # print("lamb[d, k] = {}".format(lamb[d,k]))
                # print("mu_dk/sigmasq_k= = {}".format(mu_dk/sigmasq_k))
                # print("gamma_dk= = {}".format(gamma_dk))
                # print("tausq_dk= = {}".format(tausq_dk))
                # print("kappa_dk= = {}".format(kappa_dk))

                
                # if eta_sum>=1000000000000000000000000000000000000000000000000000:
               
                #     raise ValueError("eta blows")
                    

                    
                # printf("eta_sum for doc %d, topic %d is %f \n", d, k, eta_sum) 
                # printf("sigmasq_k for doc %d topic %d is %f \n", d, k, sigmasq_k)
                # printf("lamb[d,k]=%f, \n zeta_dk=%f, \n mu_dk/sigmasq_k=%f \n", lamb[d,k], zeta_dk, mu_dk/sigmasq_k)
                # printf("gamma_dk is %f \n", gamma_dk)
            

                # Polyagamma parameters
                rho_dk = eta[d,k] - zeta_dk
                #printf("rho_dk is %f \n", rho_dk)
                # print("rho_dk={}".format(rho_dk))
                #eta[d, k] = gaussian(r, tausq_dk) + gamma_dk
                eta[d, k] = gaussian(r, tausq_dk) 
                #printf("new eta is before gamma is %f \n", eta[d, k])
                eta[d, k ] +=gamma_dk
                
                #printf("gaussian ok for doc %d, topic %d \n", d, k)
                lamb[d, k] = pg_rng.pgdraw(nd[d], rho_dk)
                #printf("pgdraw ok for doc %d, topic %d \n", d, k)

            




        # Update Global Parameters
        for k in xrange(n_topics-1):
            eta_sub[:,k] = eta[:,k]

        for p in xrange(n_cov):
            for k in xrange(n_topics-1):
                Xteta[p, i] = np.dot(Xt[p, :], eta[:,i])
        Mean_newB = Lninv_LaB0 + np.dot(Lambda_n_inv, Xteta)

        resid_eta = eta_sub - np.dot(X, Mean_newB)
        resid_Bn =  Mean_newB - B0
        Scale_newSigma = np.dot(resid_eta.T, resid_eta) + \
                            np.dot(np.dot(resid_Bn.T, Lambda0), resid_Bn) + psi

        
        vec_Bn = np.ravel(Mean_newB)
        kr_SigLamb_new = np.kron(Sigma , Lambda_n_inv)
        B  = np.random.multivariate_normal(vec_Bn, kr_SigLamb_new).reshape((n_cov, n_topics-1))
        #print("vectorized normal for B ok \n")
        Sigma = sp.stats.invwishart.rvs(df=df_newSigma, scale=Scale_newSigma, size=1) 
        #print("wishart for sigma ok \n")
        Sigma_inv = np.linalg.inv(Sigma)
        #print("inversion ok \n")
        XB = np.dot(X, B)

        ##likelihood
        ln_likeli[it] = log_likelihood(beta, beta_sum, lbeta_w,
                                        eta, nzw, ndz, nz)


        if it < burnin :
            pass
        
        else :

            post_z.append(topic_lookup)
            post_eta.append(eta)
            post_lamb.append(lamb)
            post_B.append(B)
            post_Sigma.append(Sigma)

        # print progress
        print_progress(start_time, n_report_iter, it, \
            ln_likeli[it], ln_likeli[it - n_report_iter])

       



    return post_z, post_eta, post_lamb, post_B, post_Sigma, ln_likeli




cdef double log_likelihood(double[:] beta, 
                            double beta_sum, double lbeta_w,
                            double[:,:] eta, int[:, :] nzw, int[:, :] ndz, int[:] nz):

    cdef int k, d
    cdef int n_docs = ndz.shape[0]
    cdef int n_topics = ndz.shape[1]
    cdef int n_terms = nzw.shape[1]
    cdef double ll = 0
    # calculate log p(w|z)
    ll += n_topics * (lngamma(beta_sum) - lbeta_w)
    for k in range(n_topics):
        ll -= lngamma(beta_sum + nz[k])
        for w in range(n_terms):
            ll += lngamma(beta[w] + nzw[k, w])
    # calculate log p(z)
    for d in range(n_docs):
        for k in range(n_topics):
            ll += ndz[d, k]*eta[d, k]
    return ll
    

