import TD
import agent
import numpy as np

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__(8, 2, 1.0)
    
    def get_actions(self, state):
        return np.arange(0,self.action_size,1,dtype=int)
    
    def act(self, state, action):
        if state in [0,6,7]:
            new_state = 7
            r = 0
        else:
            if action == 0:
                new_state = state - 1
            else :
                new_state = state + 1
            new_state = min(self.state_size-1, max(0, new_state))
            r = float(new_state == 6)
        return new_state, r

    def stop_state(self):
        return 7

    def new_episode(self):
        return 3

    def policy(self, state, action):
        return 1/self.action_size

    def policy_select(self, state):
        if np.random.rand() < 1/self.action_size:
            return 0
        else:
            return 1

    def action_value(self, state, action):
        return self.q[state][action]

    def print_improvement(self):
        for i in range(7):
            value = np.sum([self.policy(i,a) * self.action_value(i, a) for a in self.get_actions(i)])
            print(value, end=' ')
        print()

if __name__ == "__main__":
    quest = myAgent()
    method = TD.algo(quest)
    method.TD_control(epoch=100, step=1, step_size=0.5)
    quest.print_improvement()