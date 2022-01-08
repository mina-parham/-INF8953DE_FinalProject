import numpy as np
import scipy.sparse.linalg as spla
def LARSTD(D, phi_function, beta, gamma, w1, number_of_features):
    k = number_of_features  # Number of features
    m = len(D)  # Number of data
    w = np.zeros(k, dtype=np.float64)
    c = np.zeros(k, dtype=np.float64)
    I = np.full(k, False, dtype=bool)
    phi1_list = []
    phi2_list = []
    for i in range(m):
        phi1, phi2 = phi_function(D[i], w1)
        phi1_list.append(np.array(phi1))
        phi2_list.append(np.array(phi2))
        # c += phi[D[i]['s'], D[i]['a'], :] * D[i]['r']
        c += phi1 * D[i]['r']
    i = np.argmax(np.abs(c))
    beta_b = np.abs(c[i])
    I[i] = True
    lenI = 1
    while (beta_b > beta):
        # 1. Find update direction dwI
        AII = np.zeros((lenI, lenI))
        for i in range(m):
            AII = AII + np.outer(phi1_list[i][I], phi1_list[i][I] - gamma * phi2_list[i][I])
            # AII = AII + np.outer(phi[D[i]['s'], D[i]['a'], I],(phi[D[i]['s'], D[i]['a'], I] - gamma * phi[D[i]['sp'], pi[D[i]['sp']], I]))

        # dw = np.matmul(np.linalg.inv(AII),np.sign(c[I]))
        dw = spla.spsolve(AII, np.sign(c[I]))
        # 2. Find step size to add element to the active set:
        d = np.zeros(k, dtype=np.float64)
        for i in range(m):
            # mul1 = np.inner(
            #    phi[D[i]['s'], D[i]['a'], I] - gamma * phi[D[i]['sp'], pi[D[i]['sp']], I],dw)
            # d = d + phi[D[i]['s'], D[i]['a'], :] * mul1

            mul1 = np.inner(phi1_list[i][I] - gamma * phi2_list[i][I], dw)
            d = d + phi1_list[i] * mul1
        d[d == 1] = 0.999
        d[d == -1] = -0.999
        f1 = (c - beta_b) / (d - 1)
        f2 = (c + beta_b) / (d + 1)
        alpha1, i1 = min_plus(np.concatenate((f1.reshape(-1, 1), f2.reshape(-1, 1)), axis=1), ~I)
        # 3. Find step size to reach n zero coefficient:
        temp = np.zeros((k, 1))
        temp[I, 0] = (-w[I] / dw)
        alpha2, i2 = min_plus(temp, I)
        if alpha2 == 0:
            alpha2 = np.Inf
        # 4. Update weights, β̄, and correlation vector:

        alpha = np.min((alpha1, alpha2, beta_b - beta))
        w[I] += alpha * dw
        w[~I] = 0
        beta_b = beta_b - alpha
        c = c - alpha * d
        # 5. Add i1 or remove i2 from active set:
        if (alpha1 < alpha2):
            I[i1] = True
            lenI += 1
        else:
            I[i2] = False
            lenI -= 1
    return np.array(w)

def min_plus(input, indices):
    x = np.array(input)
    x[~indices, :] = np.Inf
    x[x <= 0] = np.Inf
    i = x.min(1).argmin()
    a = x.min(1)[i]
    return a, i