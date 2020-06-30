import MC
import agent
import numpy as np

class myAgent(agent.Agent):
    def __init__(self, state_size, action_size, tao=0.9):
        super().__init__(state_size, action_size, tao)

    def action_value(self, state, action):
        return self.q[state][action]

    def policy(self, state, action):
        # e-greedy
        e = 0.1
        if self.p[state] == action:
            return 1 - e + e/self.action_size
        else:
            return e/self.action_size
    
    def policy_select(self, state):
        e = 0.1
        if np.random.rand() < e:
            return np.random.randint(0,2)
        else:
            return self.p[state]

    def get_actions(self, state):
        return [0,1]
    
    def new_episode(self):
        
        def dealer_policy(showed_card):
            sum = showed_card
            vA = False
            if showed_card == 1:
                sum += 10
                vA = True
            while sum < 17:
                get_card = min(10, np.random.randint(0,13) + 1)
                sum += get_card
                if sum > 21:
                    if vA:
                        sum -= 10
                        vA = False
                    else:
                        return -1
                else:
                    if get_card == 1 and sum+10 <=21:
                        sum += 10
                        vA = True
            return sum
        
        s = np.random.randint(0,self.state_size)
        a = np.random.randint(0,self.action_size)
        
        finished = False
        traj = []
        while True:
            valuable_A = s % 2
            dealer_card = s // 2 % 10 + 1
            player_sum = s // 2 // 10 + 12
            if a == 0:
                get_card = min(10, np.random.randint(0,13) + 1)
                # print("get", get_card)
                player_sum += get_card
                if player_sum > 21:
                    if valuable_A == 1:
                        player_sum -= 10
                        valuable_A = 0
                        r = 0
                    else:
                        r = -1
                        finished = True
                else:
                    r = 0
            else:
                dealer_sum = dealer_policy(dealer_card)
                if dealer_sum == -1 or dealer_sum < player_sum:
                    r = 1
                elif dealer_sum > player_sum:
                    r = -1
                else:
                    r = 0
                finished = True
            traj.append([s,a,r])
            if finished:
                break
            s = ((player_sum-12)*10+dealer_card-1)*2 + valuable_A
            a = self.p[s]
        return traj

    def print_evaluation(self):
        print("value matrix")
        for valuable_A in range(2):
            print("valuable A", valuable_A)
            for dealer_card in range(10):
                for player_sum in range(10):
                    state_idx = (player_sum*10+dealer_card)*2 + valuable_A
                    print("{:.3f}".format(self.q[state_idx][self.p[state_idx]]), end=' ')
                print()
    
    def print_improvement(self):
        print("policy matrix")
        for valuable_A in range(2):
            print("valuable A", valuable_A)
            for dealer_card in range(10):
                for player_sum in range(10):
                    state_idx = (player_sum*10+dealer_card)*2 + valuable_A
                    print(self.p[state_idx], end=' ')
                print()

if __name__ == "__main__":
    test = myAgent((21-12+1)*10*2, 2, 1.0)

    # for i in range(10):
    #     print("=================", "episode", i, "=================")
    #     test.new_episode()

    method = MC.algo(test, 0.0001)
    method.MC_control(100)#show=True)
    test.print_evaluation()
    test.print_improvement()