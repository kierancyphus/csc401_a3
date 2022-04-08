import fnmatch
import os
import random

import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from tqdm import tqdm

dataDir = "/u/cs401/A3/data/"
# dataDir = "./data"


class theta:
    def __init__(self, name, M=8, d=13):
        """Class holding model parameters.
        Use the `reset_parameter` functions below to
        initialize and update model parameters during training.
        """
        self.name = name
        self._M = M
        self._d = d
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))
        self.precomputed = [None for _ in range(M)]

    def precomputedForM(self, m):
        """Put the precomputedforM for given `m` computation here
        This is a function of `self.mu` and `self.Sigma` (see slide 32)
        This should output a float or equivalent (array of size [1] etc.)
        NOTE: use this in `log_b_m_x` below
        """
        omega, mu, sigma = self.omega[m, :], self.mu[m, :], self.Sigma[m, :]
        if self.precomputed[m] is None:
            self.precomputed[m] = - mu.shape[0] / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum(np.log(sigma))
        return self.precomputed[m]

    def reset_omega(self, omega):
        """Pass in `omega` of shape [M, 1] or [M]
        """
        omega = np.asarray(omega)
        assert omega.size == self._M, "`omega` must contain M elements"
        self.omega = omega.reshape(self._M, 1)

    def reset_mu(self, mu):
        """Pass in `mu` of shape [M, d]
        """
        mu = np.asarray(mu)
        shape = mu.shape
        assert shape == (self._M, self._d), "`mu` must be of size (M,d)"
        self.mu = mu

    def reset_Sigma(self, Sigma):
        """Pass in `sigma` of shape [M, d]
        """
        Sigma = np.asarray(Sigma)
        shape = Sigma.shape
        assert shape == (self._M, self._d), "`Sigma` must be of size (M,d)"
        self.Sigma = Sigma

    def reset_precomputed(self):
        self.precomputed = [None for _ in range(self._M)]


def log_b_m_x(m, x, myTheta):
    """ Returns the log probability of d-dimensional vector x using only
        component m of model myTheta (See equation 1 of the handout)

    As you'll see in tutorial, for efficiency, you can precompute
    something for 'm' that applies to all x outside of this function.
    Use `myTheta.preComputedForM(m)` for this.

    Return shape:
        (single row) if x.shape == [d], then return value is float (or equivalent)
        (vectorized) if x.shape == [T, d], then return shape is [T]

    You should write your code such that it works for both types of inputs.
    But we encourage you to use the vectorized version in your `train`
    function for faster/efficient computation.
    """
    if len(x.shape) == 1:
        # lame version
        omega, mu, sigma = myTheta.omega[m, :], myTheta.mu[m, :], myTheta.Sigma[m, :]
        original = -1 / 2 * np.sum(((x - mu) ** 2) / sigma)
        return original + myTheta.precomputedForM(m)

    else:
        omega, mu, sigma = myTheta.omega[m, :], myTheta.mu[m, :], myTheta.Sigma[m, :]
        original = -1 / 2 * np.sum(((x - mu) ** 2) / sigma, axis=1)
        precomputed = - mu.shape[0] / 2 * np.log(2 * np.pi) - 1 / 2 * np.sum(np.log(sigma))

        # should be able to broadcast them together
        return original + precomputed


def log_p_m_x(log_Bs, myTheta):
    """ Returns the matrix of log probabilities i.e. log of p(m|X;theta)

    Specifically, each entry (m, t) in the output is the
        log probability of p(m|x_t; theta)

    For further information, See equation 2 of handout

    Return shape:
        same as log_Bs, np.ndarray of shape [M, T]

    NOTE: For a description of `log_Bs`, refer to the docstring of `logLik` below
    """

    log_omega = np.log(myTheta.omega)
    return log_omega + log_Bs - logsumexp(log_omega + log_Bs, axis=0)


def logLik(log_Bs, myTheta):
    """ Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    """
    log_omega = np.log(myTheta.omega)
    return np.sum(logsumexp(log_omega + log_Bs, axis=0))


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    """ Train a model for the given speaker. Returns the theta (omega, mu, sigma)"""
    myTheta = theta(speaker, M, X.shape[1])
    # perform initialization (Slide 32)
    # for ex.,
    myTheta.reset_omega(np.ones(myTheta.omega.shape) / myTheta.omega.shape[0])  # uniform distribution
    myTheta.reset_Sigma(np.ones(myTheta.Sigma.shape))  # all standard deviation of 1 - Can't leave as 0
    myTheta.reset_mu(X[np.random.randint(0, X.shape[0], M), :])  # samples the mus as random data points

    improvement = np.inf
    prev_l = -np.inf
    i = 0
    while i <= maxIter and improvement >= epsilon:
        # calculate intermediate values
        log_Bs = np.array([log_b_m_x(m, X, myTheta) for m in range(M)])
        log_likelihood = logLik(log_Bs, myTheta)
        log_p_ms = log_p_m_x(log_Bs, myTheta)

        # Calculate MLE params
        pms = np.exp(log_p_ms)
        reused = np.expand_dims(np.sum(pms, axis=1), -1)
        omega = reused / X.shape[0]
        mu = (pms @ X) / reused
        sigma = (pms @ (X ** 2)) / reused - mu ** 2

        # update step (and reset precomputed)
        myTheta.reset_omega(omega)
        myTheta.reset_mu(mu)
        myTheta.reset_Sigma(sigma)
        myTheta.reset_precomputed()

        # compute improvements
        likelihood = np.exp(log_likelihood)
        improvement = likelihood - prev_l
        prev_l = likelihood
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5, should_print=True):
    """ Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    """

    llhs = []
    for model in models:
        log_Bs = np.array([log_b_m_x(m, mfcc, model) for m in range(model.mu.shape[0])])
        llhs.append(logLik(log_Bs, model))

    best_model = np.argmax(llhs)

    if k > 0 and should_print:
        print(f"{models[correctID].name}")
        llhs_with_index = list(sorted(zip(llhs, range(len(llhs)))))[::-1]
        for i in range(k):
            temp_model = models[llhs_with_index[i][1]]
            print(f"{temp_model.name} {llhs_with_index[i][0]}")

        print()

    return 1 if (best_model == correctID) else 0


def train_iterations(maxIter=20, M=1, should_print=True):
    trainThetas = []
    testMFCCs = []
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = M
    maxIter = maxIter
    epsilon = 0.0
    # train a model for each speaker, and reserve data for testing

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            # if should_print:
            #     print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), "*npy")
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))

            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0

    for i in range(len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k, should_print)
    accuracy = 1.0 * numCorrect / len(testMFCCs)

    if should_print:
        print(f"Accuracy: {accuracy}")
    return accuracy


if __name__ == "__main__":
    train_iterations(20, M=8)

    # experiments
    """
    Since our epsilon is 0, it looks like out GMM always trains for the full 20 iterations, and using the likelihood
    as the termination condition isn't ideal since it is always 0, so a valid experiment might be to see what the
    effects of different max iterations are on accuracy. Since we know 20 iterations guarantees 100% accuracy.
    """
    experiment_iter = False
    experiment_M = False
    experiment_average_accuracy = False

    if experiment_average_accuracy:
        accuracies = [train_iterations(should_print=False, M=8) for _ in tqdm(range(20))]
        print("Test results below!")
        print(f"Average accuracy: {np.average(accuracies)}")
        print(f"Std deviation: {np.std(accuracies)}")

    if experiment_M:
        ms = list(range(1, 9))
        accuracies = [train_iterations(M=m, should_print=False) for m in tqdm(ms)]
        print("Test results below!")
        print(ms)
        print(accuracies)
        plt.scatter(ms, accuracies)
        plt.xlabel('Number of components')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs Number of components')
        plt.show()

    if experiment_iter:
        max_iters = list(range(0, 20, 2))
        accuracies = [train_iterations(i, False) for i in tqdm(max_iters)]
        print("Test results below!")
        print(max_iters)
        print(accuracies)
        plt.scatter(max_iters, accuracies)
        plt.xlabel('max iterations')
        plt.ylabel('Test Accuracy')
        plt.title('Test Accuracy vs iterations')
        plt.show()

