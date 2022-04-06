import fnmatch
import os
import random

import numpy as np

# dataDir = '/u/cs401/A3/data/'
dataDir = "./data/"


class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


def b_m_x(m, x, myTheta):
    omega, mu, sigma = myTheta.omega[m, :], myTheta.mu[m, :], myTheta.Sigma[m, :]

    return np.exp(-1 / 2 * np.sum(((x - mu) ** 2) / sigma)) / ((2 * np.pi) ** (mu.shape[0] / 2)) / np.sqrt(
        np.prod(sigma))


def log_b_m_x(m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    omega, mu, sigma = myTheta.omega[m, :], myTheta.mu[m, :], myTheta.Sigma[m, :]
    original = -1 / 2 * np.sum(((x - mu) ** 2) / sigma)
    precomputed = - mu.shape[0] / 2 * np.log(2 * np.pi) - 1 / 2 * np.log(np.prod(sigma))
    # print(f"original: {original}")
    # print(f"precomputed: {precomputed}")

    total = original + precomputed
    # print(f"total: {total}")
    # last term below could also be the sum of a bunch of log terms but I'm not sure how much this improves stability
    return original + precomputed


def log_p_m_x(m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    omega, mu, sigma = myTheta.omega[m], myTheta.mu[m, :], myTheta.Sigma[m, :]
    denom = np.sum([np.exp(np.log(omega) + log_b_m_x(m, x, myTheta)) for m in range(myTheta.omega.shape[0])])
    return np.log(omega) + log_b_m_x(m, x, myTheta) - np.log(denom)


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''

    inside = np.sum(myTheta.omega * np.exp(log_Bs), axis=1)
    return np.sum(np.log(inside)).squeeze(-1)


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])

    # initialize theta
    myTheta.omega = np.ones(myTheta.omega.shape) / myTheta.omega.shape[0]  # uniform distribution
    myTheta.Sigma = np.ones(myTheta.Sigma.shape)  # all standard deviation of 1 - Can't leave as 0
    myTheta.mu = X[np.random.randint(0, X.shape[0], M), :]  # samples the mus as random data points
    # myTheta.mu = np.random.choice(X[0, :])
    # mean of 0 is fine

    improvement = np.inf
    prev_l = -np.inf
    i = 0
    while i <= maxIter and improvement >= epsilon:
        print(f"i = {i}")
        # calculate intermediate values
        lpmx = [[log_p_m_x(m, X[t, :], myTheta) for t in range(X.shape[0])] for m in range(M)]
        log_Bs = [[log_b_m_x(m, X[t, :], myTheta) for t in range(X.shape[0])] for m in range(M)]
        llh = logLik(log_Bs, myTheta)
        pmx = np.exp(lpmx)

        # Calculate MLE params
        omega = np.sum(pmx, axis=1) / X.shape[0]
        print(f"pmx shape: {pmx.shape}, X shape: {X.shape}")
        mu = np.sum(pmx @ X, axis=1) / np.sum(pmx, axis=1)
        sigma = np.sum(pmx @ (X ** 2), axis=1) / np.sum(pmx, axis=1) - mu ** 2

        # udpate Theta
        myTheta.omega, myTheta.mu, myTheta.Sigma = omega, mu, sigma

        # compute improvements
        improvement = np.exp(llh) - prev_l
        prev_l = np.exp(llh)
        i += 1

    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''

    print(f"shape of mfcc = {mfcc.shape}")

    llhs = []
    for model in models:
        log_Bs = [[log_b_m_x(m, mfcc[t, :], model) for t in range(X.shape[0])] for m in range(M)]
        llhs.append(logLik(log_Bs, model))

    best_model = np.argmax(llhs)

    if k > 0:
        print(f"[{models[correctID].name}]")
        llhs_with_index = list(sorted(zip(llhs, range(len(llhs)))))[::-1]
        for i in range(k):
            temp_model = models[llhs_with_index[i][1]]
            print(f"[{temp_model.name}], [{llhs_with_index[i][0]}]")

    return 1 if (best_model == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.1
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
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
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
