import TD
import agent
import numpy as np

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__(4*12, 4, 1)

    
    def get_actions(self, state):
        return np.arange(0,self.action_size,1,dtype=int)
    
    def act(self, state, action):
        x = state // 12
        y = state % 12

        # move
        if action == 0 : x -= 1
        elif action == 1 : x += 1
        elif action == 2 : y -= 1
        else: y += 1

        x = min(4-1, max(0, x))
        y = min(12-1, max(0, y))

        # cliff
        if x == 3 and y > 0 and y < 11:
            r = -100.0
            x = 3
            y = 0
        else:
            r = -1.0

        new_state = x*12 + y
        return new_state, r

    def stop_state(self):
        return 3 * 12 + 11

    def new_episode(self):
        return 3 * 12 + 0

    def policy(self, state, action):
        # if action == self.p[state]:
        #     return 1.0
        # else:
        #     return 0.0
        e = 0.1
        if action == self.p[state]:
            return 1.0 - e + e/self.action_size
        else:
            return e/self.action_size

    def policy_off(self, state, action):
        e = 0.1
        if action == self.p[state]:
            return 1.0 - e + e/self.action_size
        else:
            return e/self.action_size

    def policy_select(self, state):
        # e-greedy
        e = 0.1
        if np.random.rand() < e:
            return np.random.randint(0,self.action_size)
        else:
            return self.p[state]

    def action_value(self, state, action):
        return self.q[state][action]

    def print_improvement(self):
        for i in range(4):
            for j in range(12):
                print(self.p[i*12+j], end=' ')
            print()

if __name__ == "__main__":
    quest = myAgent()
    method = TD.algo(quest)
    method.TD_control(500, 1)
    quest.print_improvement()