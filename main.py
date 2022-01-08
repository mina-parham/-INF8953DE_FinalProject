import numpy as np
import matplotlib.pyplot as plt
from utils import policy_eval,generate_mountaincar_data,fig1a,fig1b,fig1c,test_mountaincar,mountaincar_policy
from LARSTD import LARSTD
from LSTDQ import LSTDQ,LSTDQ2,FastLSTDQ
import scipy.stats as st


def chainwalk_fig1():
    n_sample_vector = [100, 500, 1000, 1500, 3000, 5000]
    n_irr_features = 1000
    number_of_run = 20
    n_policy_iter = 6
    np.random.seed(9)
    rbf_eps = 0.01
    sigma = 0.3
    beta = 40
    gamma = 0.9

    V_LARSTD=fig1a(LARSTD,beta,gamma,n_sample_vector,n_irr_features,number_of_run,n_policy_iter,rbf_eps=rbf_eps,sigma=sigma)
    fig, ax = plt.subplots()
    ax.plot(n_sample_vector,np.mean(V_LARSTD,axis=0),'-b*')
    ax.legend(['L1 Regularization'],loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average discounted reward')
    plt.show()

    np.random.seed(9)
    V_LSTDQ=fig1a(LSTDQ2,beta,gamma,n_sample_vector,n_irr_features,number_of_run,n_policy_iter,rbf_eps=rbf_eps,sigma=sigma)

    fig, ax = plt.subplots()
    ax.plot(n_sample_vector,np.mean(V_LSTDQ,axis=0),'-b*')
    ax.legend(['L1 Regularization'],loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average discounted reward')
    plt.show()

    ci = 0.95
    ci_interval_LSTDQ = np.zeros((V_LSTDQ.shape[1], 2))
    ci_interval_LARSTD = np.zeros((V_LARSTD.shape[1], 2))

    for i in range(V_LSTDQ.shape[1]):
        ci_interval_LSTDQ[i] = st.norm.interval(alpha=ci, loc=np.mean(V_LSTDQ[:, i]), scale=st.sem(V_LSTDQ[:, i]))
    for i in range(V_LARSTD.shape[1]):
        ci_interval_LARSTD[i] = st.norm.interval(alpha=ci, loc=np.mean(V_LARSTD[:, i]), scale=st.sem(V_LARSTD[:, i]))

    fig, ax = plt.subplots()

    ax.plot(n_sample_vector, np.mean(V_LARSTD, axis=0), '-b*')
    ax.vlines(n_sample_vector, ci_interval_LARSTD[:, 0], ci_interval_LARSTD[:, 1], color='b')

    ax.plot(n_sample_vector, np.mean(V_LSTDQ, axis=0), '--r*')
    ax.vlines(n_sample_vector, ci_interval_LSTDQ[:, 0], ci_interval_LSTDQ[:, 1], color='r')

    ax.legend(['L1 Regularization', 'L2 Regularization'], loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of samples')
    ax.set_ylabel('Average discounted reward')
    plt.show()
    fig.savefig('fig_a.png', dpi=300)
    fig.savefig('fig_a.pdf', dpi=300)


def chainwalk_fig2():
    n_irr_features_vector = [0, 10, 200, 500, 1000, 1500, 2000, 3000, 4000]
    n_sample = 800
    number_of_run = 20
    n_policy_iter = 6
    np.random.seed(9)
    rbf_eps = 0.01
    sigma = 2
    beta = 40
    gamma = 0.9

    V_LARSTD = fig1b(LARSTD, beta, gamma, n_sample, n_irr_features_vector, number_of_run, n_policy_iter,rbf_eps=rbf_eps,sigma=sigma)
    fig, ax = plt.subplots()
    ax.plot(n_irr_features_vector, np.mean(V_LARSTD, axis=0), '-b*')
    ax.legend(['L1 Regularization'], loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of irrelevent features')
    ax.set_ylabel('Average discounted reward')
    plt.show()

    V_LSTDQ = fig1b(LSTDQ2,beta,gamma,n_sample,n_irr_features_vector,number_of_run,n_policy_iter,rbf_eps=rbf_eps,sigma=sigma)
    fig, ax = plt.subplots()
    ax.plot(n_irr_features_vector,np.mean(V_LSTDQ,axis=0),'-r*')
    ax.legend(['L2 Regularization'],loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of irrelevent features')
    ax.set_ylabel('Average discounted reward')
    plt.show()

    ci = 0.95
    ci_interval_LSTDQ = np.zeros((V_LSTDQ.shape[1], 2))
    ci_interval_LARSTD = np.zeros((V_LARSTD.shape[1], 2))

    for i in range(V_LSTDQ.shape[1]):
        ci_interval_LSTDQ[i] = st.norm.interval(alpha=ci, loc=np.mean(V_LSTDQ[:, i]), scale=st.sem(V_LSTDQ[:, i]))
    for i in range(V_LARSTD.shape[1]):
        ci_interval_LARSTD[i] = st.norm.interval(alpha=ci, loc=np.mean(V_LARSTD[:, i]), scale=st.sem(V_LARSTD[:, i]))

    fig, ax = plt.subplots()

    ax.plot(n_irr_features_vector, np.mean(V_LARSTD, axis=0), '-b*')
    ax.vlines(n_irr_features_vector, ci_interval_LARSTD[:, 0], ci_interval_LARSTD[:, 1], color='b')

    ax.plot(n_irr_features_vector, np.mean(V_LSTDQ, axis=0), '--r*')
    ax.vlines(n_irr_features_vector, ci_interval_LSTDQ[:, 0], ci_interval_LSTDQ[:, 1], color='r')

    ax.legend(['L1 Regularization', 'L2 Regularization'], loc='lower right')
    ax.grid('on')
    ax.set_xlabel('Number of irrelevent features')
    ax.set_ylabel('Average discounted reward')
    plt.show()
    fig.savefig('fig_b.png', dpi=300)
    fig.savefig('fig_b.pdf', dpi=300)

def chainwalk_fig3():
    n_irr_features_vector = [0, 10, 200, 500, 1000, 1500, 2000, 3000, 4000]
    n_sample = 800
    number_of_run = 4
    n_policy_iter = 6
    np.random.seed(50)  # 9
    rbf_eps = 0.01
    sigma = .5
    beta = 40
    gamma = 0.9

    T_LARSTD = fig1c(LARSTD, beta, gamma, n_sample, n_irr_features_vector, number_of_run, n_policy_iter, rbf_eps=rbf_eps,sigma=sigma)
    T_LSTDQ =  fig1c(LSTDQ2, beta, gamma, n_sample, n_irr_features_vector, number_of_run, n_policy_iter, rbf_eps=rbf_eps,sigma=sigma)

    fig, ax = plt.subplots()

    ax.plot(n_irr_features_vector,T_LARSTD,'-b*')
    ax.plot(n_irr_features_vector,T_LSTDQ,'--r*')

    ax.legend(['L1 Regularization','L2 Regularization'],loc='upper left')
    ax.grid('on')
    ax.set_xlabel('Number of irrelevent features')
    ax.set_ylabel('Run time per LSTD/LARS-TD iteration')
    plt.show()
    fig.savefig('fig_c.png', dpi=300)
    fig.savefig('fig_c.pdf', dpi=300)

def mountaincar_results():
    n_sample=500
    n_steps = 10
    number_of_run=20
    n_policy_iter=10
    np.random.seed(9)
    rbf_eps=.02
    beta = 2
    gamma = 0.9
    D=generate_mountaincar_data(n_sample,n_steps,rbf_eps)

    w = mountaincar_policy(D, LARSTD, beta, gamma, n_policy_iter,rbf_eps)
    success_rate = test_mountaincar(w, number_of_run, maximumsteps=1000,rbf_eps=rbf_eps)
    print('LARSTD success rate = ', success_rate)
    w = mountaincar_policy(D, LSTDQ2, beta, gamma, n_policy_iter,rbf_eps)
    success_rate = test_mountaincar(w, number_of_run, maximumsteps=1000,rbf_eps=rbf_eps)
    print('LSTDQ success rate = ', success_rate)
if __name__ == '__main__':
    chainwalk_fig1()
    chainwalk_fig2()
    chainwalk_fig3()
    mountaincar_results()


