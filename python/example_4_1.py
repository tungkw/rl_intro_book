import DP
import agent

class myAgent(agent.Agent):
    def __init__(self, state_size, action_size, tao=0.9):
        super().__init__(state_size, action_size, tao)

    def state_value(self, state):
        return self.v[state.state_idx]

    def action_value(self, state, action):
        if state == 0:
            next_state = 0
        
        elif action == 0 and state not in [1,2,3]:
            next_state = (state - 4) % 15
        elif action == 1 and state not in [12,13,14]:
            next_state = (state + 4) % 15
        elif action == 2 and state not in [4,8,12]:
            next_state = (state - 1) % 15
        elif action == 3 and state not in [3,7,11]:
            next_state = (state + 1) % 15
        else:
            next_state = state

        r = self.reward(state, action)
        
        return r + self.tao * self.v[next_state]

    def policy(self, state, action):
        return 1/self.action_size
    
    def reward(self, state, action):
        if state == 0:
            return 0.0
        else:
            return -1.0

    def get_actions(self, state):
        return [i for i in range(self.action_size)]

    def print_evaluation(self):
        print("value matrix")
        for i in range(4):
            for j in range(4):
                state_idx = (i*4+j) % 15
                print("{:.3f}".format(self.v[state_idx]), end=' ')
            print()
    
    def print_improvement(self):
        print("policy matrix")
        for i in range(4):
            for j in range(4):
                state_idx = (i*4+j) % 15
                print(self.p[state_idx], end=' ')
            print()

if __name__ == "__main__":
    test = myAgent(15, 4, 1.0)
    method = DP.algo(test, threshold=0.0001)
    method.policy_evaluation(show=True)
    # method.policy_iteration(show=True)
    # method.value_iteration(show=True)