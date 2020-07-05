import TD
import agent
import numpy as np

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__(7*10, 4, 1)
        self.step = 1
    
    def get_actions(self, state):
        return np.arange(0,self.action_size,1,dtype=int)
    
    def act(self, state, action):
        x = state // 10
        y = state % 10

        # win
        if y in [3,4,5,6,8]: x -= 1
        elif y in [6,7]: x -= 2

        # move
        if action == 0 : x -= 1
        elif action == 1 : x += 1
        elif action == 2 : y -= 1
        else: y += 1

        x = min(7-1, max(0, x))
        y = min(10-1, max(0, y))
        new_state = x*10 + y
        return new_state, -1.0
    
    def is_terminal(self, state):
        if state == 3 * 10 + 7:
            return True
        else:
            return False

    def policy_select(self, state):
        # e-greedy
        e = 0.1
        if np.random.rand() < e:
            return np.random.randint(0,self.action_size)
        else:
            return self.p[state]

    def action_value(self, state, action):
        return self.q[state][action]

    def new_episode(self):
        return 3*10+0

    def print_improvement(self):
        print("policy")
        for i in range(7):
            for j in range(10):
                print(self.p[i*10+j], end=' ')
            print()
        print("policy")
        for i in range(7):
            for j in range(10):
                print("{:.3f}".format(self.q[i*10+j][self.p[i*10+j]]), end=' ')
            print()

if __name__ == "__main__":
    quest = myAgent()
    method = TD.algo(quest)
    method.TD_control(8000, show=True)
    quest.print_improvement()