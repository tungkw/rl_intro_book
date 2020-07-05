import numpy as np
from matplotlib import pyplot

class algo:
    def __init__(self, agent):
        self.agent = agent

    def TD_control(self, epoch=10, step=1, step_size=0.5, threshold=0.1, show=False):
        self.step = step
        self.step_size = step_size
        self.update_threshold = threshold
    
        G_ = 0
        x = []
        y = []
        y_ = []
        pyplot.ion()
        for episode in range(epoch):
            t = 0
            tau = 0
            T = np.Infinity
            S = [self.agent.new_episode()]
            A = [self.agent.policy_select(S[0])]
            R = [0]
            while True:
                if t < T:
                    sn, r = self.agent.act(S[t], A[t])
                    an = self.agent.policy_select(sn)
                    S.append(sn)
                    A.append(an)
                    R.append(r)
                    if sn == self.agent.stop_state():
                        T = t+1
                tau = t - self.step + 1
                if tau >= 0:
                    for k in reversed(range(tau+1, min(tau+self.step, T)+1)):
                        if k == T:
                            G = R[k]
                        elif k == tau + self.step:
                            # action_value = self.agent.action_value(S[k], A[k])
                            # action_value = self.agent.action_value(S[k], self.agent.p[S[k]])
                            action_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])
                            G = R[k] + self.agent.discount * action_value
                        else:
                            # mean_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])                  
                            # diff = G - self.agent.action_value(S[k],A[k])
                            # G = R[k] + self.agent.discount * self.agent.policy(S[k],A[k]) * diff + self.agent.discount * mean_value
                            G = R[k] + self.agent.discount * G
                    self.agent.q[S[tau]][A[tau]] += self.step_size * (G - self.agent.action_value(S[tau], A[tau]))
                    self.agent.p[S[tau]] = np.argmax(self.agent.q[S[tau]])
                if tau == T-1:
                    break
                t += 1
            
            G_ = np.sum([R[i] for i in range(1,T)])
            x.append(episode+1)
            y.append(G_)
            y_.append(np.mean(y))
            pyplot.clf()
            pyplot.ylim([-100,0])
            pyplot.plot(x,y)#_)
            pyplot.pause(0.001)