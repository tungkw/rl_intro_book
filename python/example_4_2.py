import DP
import numpy as np
from scipy.stats import poisson

poisson_cache = dict()

def poisson_probability(n, lam):
    global poisson_cache
    key = n * 10 + lam
    if key not in poisson_cache:
        poisson_cache[key] = poisson.pmf(n, lam)
    return poisson_cache[key]


class Agent:
    def __init__(self, state_size, action_size, tao=0.9):
        self.tao = tao
        self.state_size = state_size
        self.action_size = action_size
        self.v = [0.0 for i in range(state_size)]
        self.p = [0 for i in range(state_size)]

    def state_value(self, state):
        return self.v[state.state_idx]

    def action_value(self, state, action):
        moved = action - 5
        num1 = min(20, state // 21 + moved)
        num2 = min(20, state  % 21 - moved)
        
        expected_reward = - np.abs(moved) * 2
        for rent1 in range(11):
            for rent2 in range(11):
                p_rent = poisson_probability(rent1, 3) * poisson_probability(rent2, 4)
                valid_rent1 = min(num1, rent1)
                valid_rent2 = min(num2, rent2)
                r_rent = (valid_rent1 + valid_rent2) * 10 
                # for ret1 in range(11):
                #     for ret2 in range(11):
                #         p_ret = poisson_probability(ret1, 3) * poisson_probability(ret2, 2)
                #         remain1 = min(20, num1 - valid_rent1 + ret1)
                #         remain2 = min(20, num2 - valid_rent2 + ret2)
                #         expected_reward += p_rent * p_ret * (r_rent + self.tao * self.v[remain1 * 21 + remain2])
                ret1 = 3
                ret2 = 2
                remain1 = min(20, num1 - valid_rent1 + ret1)
                remain2 = min(20, num2 - valid_rent2 + ret2)
                expected_reward += p_rent * (r_rent + self.tao * self.v[remain1 * 21 + remain2])

        return expected_reward

    def policy(self, state, action):
        return int(self.p[state] == action)

    def get_actions(self, state):
        num1 = state // 21
        num2 = state  % 21
        actions = []
        for a in range(self.action_size):
            moved = a - 5
            if num1 + moved < 0 or num2 - moved < 0 :continue
            actions.append(a)
        return actions

    def print_evaluation(self):
        print("value matrix")
        for i in range(21):
            for j in range(21):
                state_idx = (i*21+j)
                print("{:.3f}".format(self.v[state_idx]), end=' ')
            print()
    
    def print_improvement(self):
        print("policy matrix")
        for i in range(21):
            for j in range(21):
                state_idx = (i*21+j) 
                print(self.p[state_idx], end=' ')
            print()

if __name__ == "__main__":
    agent = Agent(21*21, 11, 0.9)
    method = DP.algo(agent, 0.0001)
    method.policy_iteration()#show=True)
