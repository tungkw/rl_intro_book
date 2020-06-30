import MC
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


state2axis = []
axis2state = {}
for i in range(-10, 10+1):
    for j in range(-10, 10+1):
        axis2state[(i,j)] = len(state2axis)
        state2axis.append([i,j])

edges = []
edge_values = {}
for k in range(len(state2axis)):
    i,j = state2axis[k]
    if [i-1,j] not in state2axis \
    or [i+1,j] not in state2axis \
    or [i,j-1] not in state2axis \
    or [i,j+1] not in state2axis :
        edges.append(k)
        edge_values[k] = 5*np.sin(i/5+j/5)


class Agent:
    def __init__(self, state_size, action_size, tao=0.9):
        self.tao = tao
        self.state_size = state_size
        self.action_size = action_size
        self.v = [0.0 for i in range(state_size)]
        self.q = np.zeros((self.state_size, self.action_size))
        self.p = [0 for i in range(state_size)]
        self.off_policy = False

    def state_value(self, state):
        return self.v[state.state_idx]

    def action_value(self, state, action):
        return self.q[state][action]

    def policy(self, state, action):
        if state in edges:
            return 0
        return 1/self.action_size

        # # e-greedy
        # e = 0.1
        # if self.p[state] == action:
        #     return 1 - e + e/self.action_size
        # else:
        #     return e/self.action_size
    
    def policy_select(self, state):
        e = 0.1
        if np.random.rand() < e:
            return np.random.randint(0,2)
        else:
            return self.p[state]

    def policy_off(self, state, action):
        # e-greedy
        e = 0.1
        if self.p[state] == action:
            return 1 - e + e/self.action_size
        else:
            return e/self.action_size

    def policy_off_select(self, state):
        e = 0.1
        if np.random.rand() < e:
            return np.random.randint(0,2)
        else:
            return self.p[state]
        
    def get_actions(self, state):
        return [i for i in range(self.action_size)]
    
    def new_episode(self):
        s = np.random.randint(0,self.state_size)
        # a = self.policy_select(s)
        a = np.random.randint(0, self.action_size)
        
        traj = []
        while True:
            if s in edges:
                r = 0
                traj.append([s,a,r])
                break

            x,y = state2axis[s]
            if a == 0:
                x += 1
            elif a == 1:
                x -= 1
            elif a == 2:
                y += 1
            else:
                y -= 1
            s_ = axis2state[(x,y)]
            if s_ in edges:
                r = edge_values[s_]
            else:
                r = 0
            traj.append([s,a,r])
            s = s_
            a = np.random.randint(0, self.action_size)
            # a = self.policy_select(s)
                
        return traj

    def print_evaluation(self):
        print("value matrix")
        
        x = np.arange(-10,10+1,1)
        y = np.arange(-10,10+1,1)
        X,Y = np.meshgrid(x,y)
        z = np.zeros_like(X).astype(np.float)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                x = X[i][j]
                y = Y[i][j]
                if (x,y) in axis2state.keys():
                    s = axis2state[(x,y)]
                    if s in edges:
                        z[i,j] = edge_values[s]
                    else:
                        # z[i,j] = np.mean([self.action_value(s, a) for a in self.get_actions(s)])
                        z[i,j] = self.action_value(s, self.p[s])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.set_xlim(-11,11)
        ax.set_ylim(-11,11)
        ax.set_zlim(-11,11)
        ax.plot_surface(X,Y,z)
        plt.show()
             
    def print_improvement(self):
        pass

if __name__ == "__main__":
    agent = Agent(len(state2axis), 4, 1)

    # for i in range(10):
    #     print("=================", "episode", i, "=================")
    #     traj = agent.new_episode()
    #     points = np.array([state2axis[s] for s,_,_ in traj])
    #     plt.xlim([-5,5])
    #     plt.ylim([-5,5])
    #     plt.scatter(points[:,0], points[:,1])
    #     plt.show()
    #     print(traj)

    method = MC.algo(agent, 0.0001)
    method.MC_control(10000)#, show=True)
    agent.print_evaluation()
    agent.print_improvement()