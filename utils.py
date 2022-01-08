import time
import numpy as np
from mountaincar import MountainCar
from chainwalk import chain
from LARSTD import LARSTD
from LSTDQ import LSTDQ,LSTDQ2,FastLSTDQ
def policy_eval(env, w, gamma, theta=0.01):
    V = np.zeros(env.n_states)
    pi = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        pi[s] = env.linear_policy(w, s)

    while True:
        delta = 0
        for s in range(env.n_states):
            action = pi[s]
            TF = env.transition_function(s, action)
            v = 0
            for i in range(len(TF)):
                v += TF[i]['probability'] * (TF[i]['reward'] + gamma * V[TF[i]['next_state']])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

def generate_mountaincar_data(n_sample,n_steps,rbf_eps):
    D = []
    env = MountainCar(rbf_eps)
    policy = [0.334,0.333,0.333]
    for iter in range(n_sample):
      count=0
      x,v  = env.randomstate()
      while(True):
        action = np.random.choice(env.n_actions, p=policy)
        x2, v2, reward, done = env.step(action)
        D.append({'x1': x,'v1': v, 'a': action, 'r': reward, 'x2': x2,'v2':v2})
        x = x2
        v = v2
        count+=1
        if done:
          print('done')
          break
        if count==n_steps:
          break
    return D



def fig1a(method,beta,gamma,n_sample_vector,n_irr_features,number_of_run,n_policy_iter,rbf_eps,sigma): #Average reward versus number of samples for 1000 irrelevant features on the chain domain.
    number_of_features = 2*(n_irr_features+6)
    V=np.zeros((number_of_run,len(n_sample_vector)))
    for n in range(number_of_run):
        print('In progress: %',n*100.0/number_of_run)
        for m,n_sample in enumerate(n_sample_vector):
            print('n_sample:',n_sample)
            D = []
            state = 10
            env = chain(nstates=20, initial_state=state, n_rbf=5, n_irrel=n_irr_features,rbf_eps=rbf_eps,sigma=sigma)
            policy = [0.5, 0.5]
            for iter in range(n_sample):
                action = np.random.choice(env.n_actions, p=policy)
                next_state, reward = env.step(action)
                D.append({'s': state, 'a': action, 'r': reward, 'sp': next_state})
                state = next_state
            w=np.random.randn(number_of_features)
            for x in range(n_policy_iter):
                w=method(D, env.phi_f, beta, gamma, w,number_of_features)
            v0= policy_eval(env,w,gamma)
            V[n,m] = np.mean(v0)
    return V


def fig1b(method, beta, gamma, n_sample, n_irr_features_vector, number_of_run,
          n_policy_iter,rbf_eps,sigma):  # Average reward versus number of samples for 1000 irrelevant features on the chain domain.
    V = np.zeros((number_of_run, len(n_irr_features_vector)))
    for n in range(number_of_run):
        print('In progress: %', n * 100.0 / number_of_run)
        D = []
        state = 10
        env = chain(nstates=20, initial_state=state, n_rbf=5, n_irrel=0,rbf_eps=rbf_eps,sigma=sigma)
        policy = [0.5, 0.5]
        for iter in range(n_sample):
            action = np.random.choice(env.n_actions, p=policy)
            next_state, reward = env.step(action)
            D.append({'s': state, 'a': action, 'r': reward, 'sp': next_state})
            state = next_state

        for m, n_irr_features in enumerate(n_irr_features_vector):
            number_of_features = 2 * (n_irr_features + 6)
            state = 10
            env = chain(nstates=20, initial_state=state, n_rbf=5, n_irrel=n_irr_features,rbf_eps=rbf_eps,sigma=sigma)
            w=np.random.randn(number_of_features)
            for x in range(n_policy_iter):
                w=method(D, env.phi_f, beta, gamma, w,number_of_features)
            v0 = policy_eval(env, w, gamma)
            print(np.mean(v0))
            V[n, m] = np.mean(v0)
    return V

def fig1c(method, beta, gamma, n_sample, n_irr_features_vector, number_of_run,
          n_policy_iter,rbf_eps,sigma):  # Average reward versus number of samples for 1000 irrelevant features on the chain domain.
    t = np.zeros(len(n_irr_features_vector), dtype=np.float64)
    pi = np.random.randint(2, size=20, dtype=int)

    D = []
    state = 10
    env = chain(nstates=20, initial_state=state, n_rbf=5, n_irrel=0,rbf_eps=rbf_eps,sigma=sigma)
    policy = [0.5, 0.5]
    for iter in range(n_sample):
        action = np.random.choice(env.n_actions, p=policy)
        next_state, reward = env.step(action)
        D.append({'s': state, 'a': action, 'r': reward, 'sp': next_state})
        state = next_state

    for m, n_irr_features in enumerate(n_irr_features_vector):
        print('In progress: %', m * 100.0 / len(n_irr_features_vector))

        number_of_features = 2 * (n_irr_features + 6)
        state = 10
        env = chain(nstates=20, initial_state=state, n_rbf=5, n_irrel=n_irr_features,rbf_eps=rbf_eps,sigma=sigma)
        w = np.random.randn(number_of_features)

        start_time = time.process_time()
        w = method(D, env.phi_f, beta, gamma, w, number_of_features)
        t[m] = time.process_time() - start_time
    return t

def test_mountaincar(w,number_of_run,maximumsteps,rbf_eps):
    env = MountainCar(rbf_eps)
    success=0
    for i in range(number_of_run):
        x=-0.5
        v=0
        env.x=x
        env.v=v
        count=0
        while(True):
            action = env.linear_policy(w,x,v)
            x2, v2, reward, done = env.step(action)
            x = x2
            v = v2
            count+=1
            if done:
              success+=1
              break
            if count==maximumsteps:
              break
    success_rate = 100.0*success/number_of_run
    return success_rate

def mountaincar_policy(D,method,beta,gamma,n_policy_iter,rbf_eps):
    env = MountainCar(rbf_eps)
    w=np.random.randn(3*env.number_of_features)
    for x in range(n_policy_iter):
        w=method(D, env.phi_f, beta, gamma, w,3*env.number_of_features)
    return w