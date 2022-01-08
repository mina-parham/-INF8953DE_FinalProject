import numpy as np

class chain():
    def __init__(self, nstates, initial_state, n_rbf, n_irrel,rbf_eps,sigma):
        self.n_states = nstates

        self.n_actions = 2
        self.action_space = [0, 1]
        self.current_state = initial_state
        self.n_rbf = n_rbf
        self.n_irrel = n_irrel
        self.sigma = sigma
        self.rbf_mean = np.linspace(0, nstates - 1, num=self.n_rbf)
        self.rbf_eps = rbf_eps
        self.p_slip = 0.05
        self.number_of_features = 1 + n_irrel + n_rbf
        self.phitable = self.generate_phi()

    def transition_function(self, s, a):
        l = []
        if a == 0:
            na = -1
        else:
            na = +1
        next_state = s + na

        if next_state < 0:
            next_state = 0
        if next_state >= self.n_states:
            next_state = self.n_states - 1

        if (next_state == self.n_states - 1) or (next_state == 0):
            reward = 1
        else:
            reward = 0
        l.append({'next_state': next_state, 'reward': reward, 'probability': 1 - self.p_slip})

        next_state2 = s - na
        if next_state2 < 0:
            next_state2 = 0
        if next_state2 >= self.n_states:
            next_state2 = self.n_states - 1

        if (next_state2 == self.n_states - 1) or (next_state2 == 0):
            reward2 = 1
        else:
            reward2 = 0
        l.append({'next_state': next_state2, 'reward': reward2, 'probability': self.p_slip})
        return l

    def step(self, a):
        # Left 0
        # Right 1
        if a == 0:
            na = -1
        else:
            na = +1
        p = np.random.rand()
        if p < self.p_slip:
            next_state = self.current_state - na
        else:
            next_state = self.current_state + na

        if next_state < 0:
            next_state = 0
        if next_state >= self.n_states:
            next_state = self.n_states - 1

        if (next_state == self.n_states - 1) or (next_state == 0):
            reward = 1
        else:
            reward = 0
        self.current_state = next_state
        return next_state, reward

    def RBF(self, x):
        return np.exp(-self.rbf_eps * ((x - self.rbf_mean) ** 2))

    def linear_policy(self, w, s):
        return np.argmax([np.dot(w, self.phi(s, a)) for a in range(self.n_actions)])

    def generate_phi(self):
        phi = np.zeros((self.n_states, self.n_actions, 2 * self.number_of_features))
        for state in range(self.n_states):
            randomfeature = self.sigma * np.random.randn(self.n_irrel)
            phi[state, 0, :] = np.concatenate(
                [[1], self.RBF(state), randomfeature, np.zeros((self.number_of_features))], axis=0)
            phi[state, 1, :] = np.concatenate(
                [np.zeros((self.number_of_features)), [1], self.RBF(state), randomfeature], axis=0)
        return np.array(phi)

    def phi(self, state, a):
        randomfeature = self.sigma * np.random.randn(self.n_irrel)
        p = self.phitable[state, a, :]
        if a == 0:
            p[self.n_rbf + 1:self.n_rbf + 1 + self.n_irrel] = randomfeature
        #  return np.concatenate([[1], self.RBF(state), randomfeature, np.zeros((self.number_of_features))], axis=0)
        else:
            p[
            self.number_of_features + self.n_rbf + 1:self.number_of_features + self.n_rbf + 1 + self.n_irrel] = randomfeature
        # return np.concatenate([np.zeros((self.number_of_features)), [1], self.RBF(state), randomfeature], axis=0)
        return p

    def phi_f(self, D, w):
        state = D['s']
        a = D['a']
        phi1 = self.phi(D['s'], D['a'])

        state = D['sp']
        a = self.linear_policy(w, D['sp'])
        phi2 = self.phi(D['sp'], a)
        return np.array(phi1), np.array(phi2)