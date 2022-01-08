
import numpy as np
import numpy.linalg as la

LSTDQ_eps = 0.1
def LSTDQ(D, phi, beta, gamma, w1, number_of_features):
    k = number_of_features  # Number of features
    m = len(D)  # Number of data
    w = np.zeros((k, 1), dtype=np.float64)
    A = LSTDQ_eps * np.eye(k)
    b = np.zeros(k)
    for i in range(m):
        phi1, phi2 = phi(D[i], w1)
        A = A + np.outer(phi1, (phi1 - gamma * phi2)) / m
        b = b + phi1 * D[i]['r'] / m
    # w = np.matmul(np.linalg.inv(A),b)
    # w = spla.spsolve(A,b)
    w = np.dot(la.pinv(A), b)
    return np.array(w)


def LSTDQ2(D, phi, beta, gamma, w1, number_of_features): #faster version
    k = number_of_features  # Number of features
    m = len(D)  # Number of data
    w = np.zeros((k, 1), dtype=np.float64)
    stackphi1 = np.zeros((k, m), dtype=np.float64)
    stackphi2 = np.zeros((k, m), dtype=np.float64)
    stackr = np.zeros(m, dtype=np.float64)

    for i in range(m):
        phi1, phi2 = phi(D[i], w1)
        stackphi1[:, i] = phi1
        stackphi2[:, i] = phi2
        stackr[i] = D[i]['r']
    A = stackphi1.dot((stackphi1 - gamma * stackphi2).transpose()) + LSTDQ_eps * np.eye(k)
    b = stackphi1.dot(stackr)
    w = np.dot(la.pinv(A), b)
    return np.array(w)


def FastLSTDQ(D, phi, beta, gamma, pi, number_of_features): #try to run faster but not working yet!
    k = number_of_features  # Number of features
    m = len(D)  # Number of data
    A_ = (LSTDQ_eps ** -1) * np.eye(k)
    b = np.zeros(k)
    for i in range(m):
        phi1, phi2 = phi(D[i], pi)
        v = np.dot(A_, (phi1 - gamma * phi2))
        A_ = A_ - (np.outer(np.dot(A_, phi1), v) / (1 + np.inner(v, phi1)))
        b = b + phi1 * D[i]['r']
    w = np.dot(A_, b)
    return w
