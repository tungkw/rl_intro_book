import numpy as np

class algo:
    def __init__(self, agent, threshold=0.1):
        self.agent = agent
        self.update_threshold = threshold
        
        self.off_policy = False
        self.weighted_importance_sampling = False

        if self.off_policy:
            if self.weighted_importance_sampling:
                self.C = np.zeros((self.agent.state_size, self.agent.action_size))
        else:
            self.return_cnt = np.zeros((self.agent.state_size, self.agent.action_size))

    def MC_update(self, show=False):
        new_episode = self.agent.new_episode()
        action_values = {}
        G = 0
        for i in reversed(range(len(new_episode))):
            st,at,rtp1 = new_episode[i]
            G = rtp1 + self.agent.tao * G
            action_values[(st,at)] = G

        W = 1
        for (s,a) in action_values.keys():
            G = action_values[(s,a)]
            
            if self.off_policy:
                if self.weighted_importance_sampling:
                    self.C[s][a] += W
                    diff = W/self.C[s][a] * (G - self.agent.q[s][a])
                else:
                    self.return_cnt[s][a] += 1
                    diff = W/self.return_cnt[s][a] * (G - self.agent.q[s][a])
            else:
                self.return_cnt[s][a] += 1
                diff = 1/self.return_cnt[s][a] * (G - self.agent.q[s][a])
            self.agent.q[s][a] += diff

            values = [[a, self.agent.action_value(s,a)] for a in self.agent.get_actions(s)]
            values.sort(key=lambda x: x[1], reverse=True)
            if self.agent.p[s] != values[0][0]:
                self.agent.p[s] = values[0][0]
                if show:
                    self.agent.print_evaluation()
                    self.agent.print_improvement()

            if self.off_policy:
                W *= self.agent.policy(s,a) / (self.agent.policy_off(s,a) + 1e-5)
                if W == 0:
                    break
    
    def MC_control(self, epoch, show=False):
        for i in range(epoch):
            print("episode", i)
            self.MC_update(show)