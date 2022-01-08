import numpy as np

class MountainCar:
    def __init__(self,rbf_eps):

        self.g = 9.8
        self.k = 0.3
        self.dt = 0.1
        self.m = 0.2
        self.x = -0.5
        self.v = 0.0
        self.n_actions = 3
        self.action_list = [-2, 0, 2]

        self.x_space = [-1.2, 0.5]
        self.v_space = [-1.5, 1.5]
        nrbf = [2, 4, 8, 16, 32]
        rbf_mean_x = []
        rbf_mean_v = []

        for n_r in nrbf:
            xm = np.linspace(self.x_space[0], self.x_space[1], num=n_r)
            vm = np.linspace(self.v_space[0], self.v_space[1], num=n_r)
            for k1 in range(n_r):
                for k2 in range(n_r):
                    rbf_mean_x.append(xm[k1])
                    rbf_mean_v.append(vm[k2])
        self.rbf_mean_x = np.array(rbf_mean_x)
        self.rbf_mean_v = np.array(rbf_mean_v)
        self.rbf_eps = rbf_eps
        self.number_of_features = 1365

    def randomstate(self):
        self.x = np.random.uniform(self.x_space[0], self.x_space[1])
        self.v = np.random.uniform(self.v_space[0], self.v_space[1])
        return self.x, self.v

    def RBF(self, x, v):
        return np.multiply(np.exp(-self.rbf_eps * ((x - self.rbf_mean_x) ** 2)),
                           np.exp(-self.rbf_eps * ((v - self.rbf_mean_v) ** 2)))

    def phi(self, x, v, a):
        if a == 0:
            return np.concatenate([[1], self.RBF(x, v), np.zeros(2 * self.number_of_features)], axis=0)
        elif a == 1:
            return np.concatenate(
                [np.zeros(self.number_of_features), [1], self.RBF(x, v), np.zeros(self.number_of_features)], axis=0)
        else:
            return np.concatenate([np.zeros(2 * self.number_of_features), [1], self.RBF(x, v)], axis=0)

    def step(self, action):
        action_t = self.action_list[action]
        v2 = self.v + (-self.g * self.m * np.cos(3 * self.x) + (action_t / self.m) - (self.k * self.v)) * self.dt
        if v2 < self.v_space[0]:
            v2 = self.v_space[0]
        if v2 > self.v_space[1]:
            v2 = self.v_space[1]
        x2 = self.x + (v2 * self.dt)

        if x2 < self.x_space[0]:
            x2 = self.x_space[0]
            v2 = 0

        self.x = x2
        self.v = v2

        if x2 >= self.x_space[1]:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False

        return x2, v2, reward, done

    def phi_f(self, D, w):
        x = D['x1']
        v = D['v1']
        a = D['a']
        phi1 = self.phi(x, v, a)
        x2 = D['x2']
        v2 = D['v2']
        a2 = self.linear_policy(w, x2, v2)
        phi2 = self.phi(x2, v2, a2)
        return np.array(phi1), np.array(phi2)

    def linear_policy(self, w, x, v):
        return np.argmax([np.dot(w, self.phi(x, v, a)) for a in range(self.n_actions)])