import agent
import gradient_TD
import MC
import algo
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class myAgent(agent.Agent):
    def __init__(self, step_size_value):
        super().__init__(discount=1.0)#, lambd_policy=0.92,lambd_value=0.92)

        self.step_size_value = step_size_value
        self.step_size_policy = step_size_value
        # self.step_size_value = 2**(-6)
        # self.step_size_policy = 2**(-9)

        self.w = np.zeros((2,1),dtype=float)
        self.z_value = np.zeros((2,1),dtype=float)
        self.theta = np.array([-1.47, 1.47]).reshape(2,1)
        self.z_policy = np.zeros((2,1),dtype=float)
        
        self.x_data = []
        self.y_data = []
    
    def feature(self, state, action):
        return np.array([[0, 1],[1, 0]],dtype=float)[:,int(action)].reshape(2,1)

    def new_episode(self):
        return 0

    def policy(self, state, action):
        h = np.array([np.matmul(self.theta.T, self.feature(state,0)), np.matmul(self.theta.T, self.feature(state,1))], dtype=float)
        t = np.exp(h - np.max(h)).reshape(2) # avoid too large to overflow
        pmf = t / np.sum(t)
        imin = np.argmin(pmf)
        e = 0.05
        if pmf[imin] < e:
            pmf[:] = 1-e
            pmf[imin] = e
        return pmf[action]
    
    def policy_select(self, state):
        return int(np.random.rand() < self.policy(state, 1))

    def action_value(self, state, action):
        if self.stop_state(state):
            return 0.0
        x = self.feature(state, action)
        return np.matmul(self.w.T, x)

    def update(self, t, state, action, target):
        x = self.feature(state, action)

        # grad_value = x
        # diff = target - self.action_value(state, action)
        # z_ex = np.zeros_like(self.z_value)
        # w_ex = np.zeros_like(self.w)
        # # dutch trace
        # # z_ex = - self.step_size*self.discount*self.lambd*np.matmul(self.z.T, delta_value) * delta_value
        # # w_ex = self.step_size * (self.action_value(state, action)-np.matmul(self.w_last.T, x)) * (self.z - grad_value)
        # # self.w_last = self.w
        # self.z_value = self.discount * self.lambd_value * self.z_value + grad_value + z_ex
        # self.w += self.step_size_value * diff * self.z_value + w_ex
        
        diff = target
        # diff = target - np.sum([self.policy(state, i) * self.action_value(state, i) for i in range(2)])
        grad_ln_p = x - np.sum([self.feature(state, i)*self.policy(state, i) for i in range(2)])
        self.z_policy = self.discount * self.lambd_policy * self.z_policy + (self.discount**t) * grad_ln_p
        self.theta += self.step_size_policy * diff * self.z_policy

    def act(self, state, action):
        if state == 3:
            return state, 0.0

        if state == 1:
            if action == 1:
                x = state - 1
            else:
                x = state + 1
        else:
            if action == 1:
                x = state + 1
            else:
                x = state - 1
        x = max(0, x)
        r = -1.0
        return x, r
    
    def stop_state(self, state):
        return state == 3

    def print_t(self,t,St,At,Rtn,Stn,Atn,Gt):
        pass

    def print_e(self,e,S,A,R):
        G = 0.0
        for i in reversed(range(1,len(R))):
            G = R[i] + self.discount*G
        self.x_data.append(len(self.x_data) + 1)
        self.y_data.append(G)

if __name__ == "__main__":
    epoch = 1000
    trials = 100
    x_data = [i for i in range(epoch)]
    step_sizes = [ 2e-3, 2e-4, 2e-5]
    plt.ylim([-90, -10])
    for k, s in enumerate(step_sizes):
        y_datas = np.zeros((trials,epoch))
        for i in range(trials):
            print(k, i, ' ')
            ag = myAgent(s)
            method = algo.Method(ag)
            method.learn(epoch, step=float('inf'))
            y_datas[i] = ag.y_data
        plt.plot(x_data, np.mean(y_datas, axis=0))
    plt.show()