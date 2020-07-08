import numpy as np

class algo:
    def __init__(self, agent, threshold=0.1):
        self.agent = agent
        self.update_threshold = threshold
        
        # self.off_policy = False
        # self.weighted_importance_sampling = False

        # if self.off_policy:
        #     if self.weighted_importance_sampling:
        #         self.C = np.zeros((self.agent.state_size, self.agent.action_size))
        # else:
        #     self.return_cnt = np.zeros((self.agent.state_size, self.agent.action_size))

    
    def MC_control(self, epoch=10, step=1, threshold=0.1, show=False):
        for episode in range(epoch):
            t = 0
            tau = 0
            T = float('inf')
            S = [self.agent.new_episode()]
            A = [self.agent.policy_select(S[0])]
            R = [0.0]
            while True:
                if t < T:
                    sn, r = self.agent.act(S[t], A[t])
                    an = self.agent.policy_select(sn)
                    S.append(sn)
                    A.append(an)
                    R.append(r)
                    if self.agent.stop_state(sn):
                        T = t+1
                        step = min(step, T)
                tau = t - step + 1
                if tau >= 0:
                    for k in reversed(range(tau+1, min(tau+step, T)+1)):
                        if k == T:
                            G = R[k]-self.agent.R_mean
                        elif k == tau + step:
                            action_value = self.agent.action_value(S[k], A[k])
                            # action_value = self.agent.action_value(S[k], self.agent.p[S[k]])
                            # action_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])
                            G = R[k]-self.agent.R_mean + self.agent.discount * action_value
                        else:
                            # mean_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])                  
                            # diff = G - self.agent.action_value(S[k],A[k])
                            # G = R[k] + self.agent.discount * self.agent.policy(S[k],A[k]) * diff + self.agent.discount * mean_value
                            G = R[k]-self.agent.R_mean + self.agent.discount * G
                    self.agent.update(tau, S[tau], A[tau], G)
                    self.agent.print_t(tau, S[tau], A[tau], R[tau+1], S[tau+1], A[tau+1])
                if tau == T-1:
                    break
                t += 1
            self.agent.print_e(episode, S, A, R)



        #     # new_episode = self.agent.new_episode()
        #     # action_values = {}
        #     # G = 0
        #     # for i in reversed(range(len(new_episode))):
        #     #     st,at,rtp1 = new_episode[i]
        #     #     G = rtp1 + self.agent.tao * G
        #     #     action_values[(st,at)] = G

        #     # W = 1
        #     for (s,a) in action_values.keys():
        #         G = action_values[(s,a)]
                
        #         if self.off_policy:
        #             if self.weighted_importance_sampling:
        #                 self.C[s][a] += W
        #                 diff = W/self.C[s][a] * (G - self.agent.q[s][a])
        #             else:
        #                 self.return_cnt[s][a] += 1
        #                 diff = W/self.return_cnt[s][a] * (G - self.agent.q[s][a])
        #         else:
        #             self.return_cnt[s][a] += 1
        #             diff = 1/self.return_cnt[s][a] * (G - self.agent.q[s][a])
        #         self.agent.q[s][a] += diff

        #         values = [[a, self.agent.action_value(s,a)] for a in self.agent.get_actions(s)]
        #         values.sort(key=lambda x: x[1], reverse=True)
        #         if self.agent.p[s] != values[0][0]:
        #             self.agent.p[s] = values[0][0]
        #             if show:
        #                 self.agent.print_evaluation()
        #                 self.agent.print_improvement()

        #         if self.off_policy:
        #             W *= self.agent.policy(s,a) / (self.agent.policy_off(s,a) + 1e-5)
        #             if W == 0:
        #                 break
        
        # for episode in range(epoch):
        #     t = 0
        #     tau = 0
        #     T = float('inf')
        #     S = [self.agent.new_episode()]
        #     A = [self.agent.policy_select(S[0])]
        #     R = [0.0]
        #     while True:
        #         if t < T:
        #             sn, r = self.agent.act(S[t], A[t])
        #             an = self.agent.policy_select(sn)
        #             S.append(sn)
        #             A.append(an)
        #             R.append(r)
        #             if self.agent.stop_state(sn):
        #                 T = t+1
        #         tau = t - self.step + 1
        #         if tau >= 0:
        #             for k in reversed(range(tau+1, min(tau+self.step, T)+1)):
        #                 if k == T:
        #                     G = R[k]-self.agent.R_mean
        #                 elif k == tau + self.step:
        #                     action_value = self.agent.action_value(S[k], A[k])
        #                     # action_value = self.agent.action_value(S[k], self.agent.p[S[k]])
        #                     # action_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])
        #                     G = R[k]-self.agent.R_mean + self.agent.discount * action_value
        #                 else:
        #                     # mean_value = np.sum([self.agent.policy(S[k], a) * self.agent.action_value(S[k], a) for a in self.agent.get_actions(S[k])])                  
        #                     # diff = G - self.agent.action_value(S[k],A[k])
        #                     # G = R[k] + self.agent.discount * self.agent.policy(S[k],A[k]) * diff + self.agent.discount * mean_value
        #                     G = R[k]-self.agent.R_mean + self.agent.discount * G
        #             # self.agent.q[S[tau]][A[tau]] += self.step_size * (G - self.agent.action_value(S[tau], A[tau]))
        #             # self.agent.p[S[tau]] = np.argmax(self.agent.q[S[tau]])
        #             self.agent.update(tau, S[tau], A[tau], G)
        #         # print(" ",t,S[-1],A[-1],R[-1])    

        #         if tau == T-1:
        #             break
        #         t += 1
        #     print(" ",t)
        #     if episode % 10 == 0:
        #         print(episode)
        #         self.agent.print(S,A,R)