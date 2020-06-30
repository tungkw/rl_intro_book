import numpy as np

class algo:
    def __init__(self, agent, threshold=0.1):
        self.agent = agent
        self.update_threshold = threshold
    
    def policy_evaluation(self,show=False):
        iter = 0
        while True:
            v_new = np.copy(self.agent.v)
            for s in range(self.agent.state_size):
                print("policy evaluation", "state ({}/{})".format(s, self.agent.state_size), end='\r' + ' '*100 + '\r')
                update = 0
                for a in self.agent.get_actions(s):
                    p = self.agent.policy(s, a)
                    if p > 0:
                        update += p * self.agent.action_value(s, a)
                self.agent.v[s] = update
            delta = np.max(np.abs(np.subtract(self.agent.v, v_new)))

            print("policy evaluation", 'step :', iter, "delta :", delta)
            if show:
                self.agent.print_evaluation()
            if delta < self.update_threshold:
                break
            iter+=1

    def policy_improvement(self, show=False):
        stable = True
        for s in range(self.agent.state_size):
            print("policy improvement", "state ({}/{})".format(s, self.agent.state_size), end='\r' + ' '*100 + '\r')
            action_values = [[a, self.agent.action_value(s,a)] for a in self.agent.get_actions(s)]
            action_values.sort(key=lambda x: x[1], reverse=True)
            if self.agent.p[s] != action_values[0][0]:
                stable = False
                self.agent.p[s] = action_values[0][0]
        return stable

    def policy_iteration(self, show=False):
        stable = False
        iter = 0
        while not stable:
            print("--------------------------------")
            print("iteration :", iter)
            self.policy_evaluation(show)
            stable = self.policy_improvement(show)
            print("policy improvement", 'stable :', stable)
            if show:
                self.agent.print_improvement()
            iter+=1
    
    def value_iteration(self, states = None, show=False):
        iter = 0
        while True:
            print("--------------------------------")
            print("iteration :", iter)

            v_old = np.copy(self.agent.v)
            if states is None:
                states = [i for i in range(self.agent.state_size)]
            for state in states:
                print("value iteration", "state ({}/{})".format(state, len(states)), end='\r' + ' '*100 + '\r')
                action_values = [[a,self.agent.action_value(state,a)] for a in self.agent.get_actions(state)]
                action_values.sort(key=lambda x: x[1], reverse=True)
                self.agent.v[state] = action_values[0][1]

            delta = np.max(np.abs(np.subtract(self.agent.v, v_old)))
            print("value iteration", 'step :', iter, "delta :", delta)
            if show:
                self.agent.print_evaluation()
            if delta < self.update_threshold:
                break
            iter+=1
        
        self.policy_improvement(show)
        print("policy improvement")
        if show:
            self.agent.print_improvement()