import agent
import gradient_TD
import tiles3
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__(action_size=3, discount=1.0)

        self.step_size = 0.3/8
        self.w = np.zeros((2048,1),dtype=float)
        self.w_last = np.copy(self.w)


        # difference
        self.R_mean = 0.0
        self.beta = 0.0
        # trace
        self.z = np.zeros_like(self.w)
        self.lambd = 0.92
        
        self.iht = tiles3.IHT(2048)
        self.fig = plt.figure()
        self.ax = Axes3D(self.fig)
        plt.ion()
    
    def feature(self, state, action):
        x, xdot = state
        a = action - 1
        idx = tiles3.tiles(self.iht,8,[8.0*x/(0.5+1.2), 8.0*xdot/(0.07+0.07)],[a])
        fea = np.zeros((2048,1),dtype=float)
        fea[idx] = 1.0
        return fea

    def new_episode(self):
        s1 = np.random.uniform(-0.6, -0.4)
        s2 = 0.0
        return [s1,s2]
    
    def get_actions(self, state):
        return [a for a in range(self.action_size)]

    def policy(self, state, action):
        e = 0.0
        greedy_a = np.argmax([self.action_value(state, a) for a in self.get_actions(state)])
        if action == greedy_a:
            return 1 - e + e / self.action_size
        else:
            return e / self.action_size
    
    def policy_select(self, state):
        e = 0.0
        greedy_a = np.argmax([self.action_value(state, a) for a in self.get_actions(state)])
        if np.random.rand() < e:
            return np.random.randint(0,self.action_size)
        else:
            return greedy_a

    def action_value(self, state, action):
        if self.stop_state(state):
            return 0.0
        x = self.feature(state, action)
        return np.matmul(self.w.T, x)

    def update(self, state, action, target):
        if not self.stop_state(state):
            delta_value = self.feature(state, action)
            diff = target - self.action_value(state, action)
            self.R_mean += self.beta * diff

            # z_ex = np.zeros_like(self.z)
            # w_ex = np.zeros_like(self.w)
            # dutch trace
            z_ex = - self.step_size*self.discount*self.lambd*np.matmul(self.z.T, delta_value) * delta_value
            w_ex = self.step_size * (self.action_value(state, action)-np.matmul(self.w_last.T, delta_value)) * (self.z - delta_value)
            self.w_last = self.w

            self.z = self.discount * self.lambd * self.z + delta_value + z_ex
            self.w += self.step_size * diff * self.z + w_ex



    def act(self, state, action):
        x,xp = state
        move = action - 1
        xp = xp + 0.001 * move - 0.0025 * np.cos(3 * x)
        xp = min(0.07 , max(-0.07, xp))
        x = x + xp
        x = min(0.5 , max(-1.2, x))
        if x == -1.2:
            xp = 0.0
        new_state = [x,xp]
        r = -1.0
        return new_state, r
    
    def stop_state(self, state):
        if state[0] >= 0.5:
            return True
        else:
            return False

    def print(self):
        x = np.linspace(-1.2, 0.5, 50)
        y = np.linspace(-0.07, 0.07, 50)
        X,Y = np.meshgrid(x,y)
        Z = np.zeros_like(X)
        for i in range(50):
            for j in range(50):
                s = [X[i][j], Y[i][j]]
                values = [self.action_value(s, a) for a in self.get_actions(s)]
                greedy_a = np.argmax(values)
                Z[i][j] = -values[greedy_a]
        self.ax.clear()
        self.ax.plot_surface(X,Y,Z)
        plt.pause(0.01)

if __name__ == "__main__":
    test = myAgent()
    method = gradient_TD.algo(test)
    method.TD_control(9000,step=1)
    plt.waitforbuttonpress()