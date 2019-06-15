import numpy as np
import math
from scipy.misc import logsumexp
from scipy.optimize import minimize_scalar

import time

from sklearn.utils.extmath import row_norms
from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _k_init as kpp_init


class RangeCl:
    def __init__(self, n_clusters=2, n_init=10, max_iter=300, tol=1e-4, shrinkage=1e-9, beta=50):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.prior_shrinkage = shrinkage
        self.beta = beta

    def fit(self, X):
        self.X = X  # Data array
        self.n_samples = len(self.X[:,0])  # Number of points
        self.dim = len(self.X[0,:])  # Dimensionality of data

        # Prior on edge lengths
        self.alpha_prior = ((np.max(self.X, axis = 0) - np.min(self.X, axis = 0)) / 2) / self.n_clusters
        # Another prior parametrization
        if self.prior_shrinkage != 0:
            self.a0 = self.n_samples / (1/self.prior_shrinkage - 1) - 1
            self.b0 = self.alpha_prior**self.beta * self.n_samples / self.beta / (1/self.prior_shrinkage - 1)

        # Some precomputed value for fast mean k-means++ seeding
        x_squared_norms = row_norms(self.X, squared=True)

        best_l = None
        for i in range(self.n_init):
            # Randomly initialize parameters
            self.mu = kpp_init(self.X, self.n_clusters, x_squared_norms, check_random_state(None))
            self.alpha = np.array([self.alpha_prior for k in range(self.n_clusters)])
            self.pi = np.array([1/self.n_clusters]*self.n_clusters)
            l = self.EM_algorithm()

            if not best_l or l > best_l:
                best_l = l
                self.best_mu = np.array(self.mu)
                self.best_alpha = np.array(self.alpha)
                self.best_pi = np.array(self.pi)
                self.best_r = np.array(self.r)

        self.centers = self.best_mu
        self.ranges = np.array([list(zip(self.best_mu[k] - self.best_alpha[k], self.best_mu[k] + self.best_alpha[k])) for k in range(self.n_clusters)])
        self.probs = self.best_r
        self.labels_ = np.argmax(self.probs, axis=1)

    def EM_algorithm(self):
        l_old = None
        for i in range(self.max_iter):
            start = time.time()
            self.E_step()
            end = time.time()
            #print(end - start)

            start = time.time()
            self.M_step()
            end = time.time()
            print(end - start)

            l_new = np.sum(logsumexp(self.log_prob_matrix(), axis=1, b=self.pi))
            if self.prior_shrinkage != 0:
                l_new += self.log_prior()

            if l_old:
                l_rel_change = abs((l_new - l_old) / l_old)
                if (l_rel_change < self.tol):
                    break
            l_old = l_new

        print(i, l_old)
        return l_old


    def E_step(self):
        '''
        Recalculate matrix r with current parameters
        '''
        self.r = self.log_prob_matrix() + np.log(self.pi)
        self.r = np.exp(self.r - logsumexp(self.r, axis=1)[:, None])

    def M_step(self):
        #print(self._m(self.mu))
        self.pi = np.mean(self.r, axis=0)
        #self.mu = self.mu - self._m_1st_derivative(self.mu) / self._m_2nd_derivative(self.mu)

        for k in range(self.n_clusters):
            for j in range(self.dim):
                self.mu[k][j] = minimize_scalar(lambda x: np.log(self._m1(x, k, j))).x

        alpha_mle = self.beta / self.n_samples * self._m(self.mu)
        self.alpha = ((1 - self.prior_shrinkage)*alpha_mle + \
                      self.prior_shrinkage*self.alpha_prior)**(1 / self.beta)

    def log_prob_matrix(self):
        '''
        Returns (n_samples, n_clusters) matrix (p_k(x_i))
        '''
        # TODO: try to remove A :)
        A = self.dim * (math.log(self.beta) - math.lgamma(1 / self.beta))
        B = -np.sum(np.log(2*self.alpha), axis=1)
        C = -np.sum((np.abs(np.stack([self.X]*self.n_clusters, axis=1) - self.mu) / self.alpha)**self.beta, axis=2)
        return A + B + C

    def log_prior(self):
        s1 = (1 + self.a0)*np.sum(np.log(self.alpha))
        s2 = np.sum(self.b0 * np.sum(1 / self.alpha**self.beta, axis=0))
        return -(s1 + s2)

    def _m1(self, x, k, j):
        # Compute sum_i r_ik |x_i - x|^beta
        # for j-th coordinate and z = k given r = (r_ik)
        return np.sum(self.r[:, k]*np.abs(x - self.X[:, j])**self.beta)

    def _m(self, t):
        '''
        Matrix function which should be minimized to obtain mu in each M-step.
        t is a matrix n_clusters x dim
        '''
        tmp = np.stack([self.X]*self.n_clusters, axis=1) - t
        return np.sum(self.r[:, :, None] * np.abs(tmp)**self.beta, axis=0)

    def _m_1st_derivative(self, t):
        tmp = np.stack([self.X]*self.n_clusters, axis=1) - t
        return self.beta * np.sum(self.r[:, :, None] * np.sign(tmp) * np.abs(tmp)**(self.beta - 1), axis=0)

    def _m_2nd_derivative(self, t):
        tmp = np.stack([self.X]*self.n_clusters, axis=1) - t
        return self.beta * (self.beta - 1) * np.sum(self.r[:, :, None] * np.abs(tmp)**(self.beta - 2), axis=0)

    def __str__(self):
        ranges_by_coord = []
        for j in range(self.dim):
            coord = []
            for k in range(self.n_clusters):
                coord.append((self.ranges[k][j], k))

            coord = list(sorted(coord, key = lambda x: x[0][0]))
            ranges_by_coord.append(coord)

        def coord_score(ranges):
            score = 0
            total_len = 0
            for i in range(len(ranges)):
                total_len += ranges[i][0][1] - ranges[i][0][0]
                for j in range(i + 1, len(ranges)):
                    score += ranges[j][0][0] - ranges[i][0][1]

            return score / total_len

        scores = [coord_score(r) for r in ranges_by_coord]
        s = ""
        for j in sorted(range(self.dim), key = lambda j: scores[j], reverse=True):
            s += "**********\n"
            s += "Axis {}: score = {}\n".format(j, scores[j])
            for r in ranges_by_coord[j]:
                s += "[{:.2f} ... {} ... {:.2f}]   ".format(r[0][0], r[1], r[0][1])
            s += "\n"

        return s.strip()
