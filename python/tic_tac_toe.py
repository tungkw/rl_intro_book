import numpy as np

board_width = 3
board_height = 3

def check_win(state):
    # row
    for i in range(board_height):
        if state.data[i][0] == 0:
            continue
        finish = True
        winner = state.data[i][0]
        for j in range(1, board_width):
            if state.data[i][j] != winner:
                finish = False
                break
        if finish:
            return winner
            
    # column
    for j in range(board_width):
        if state.data[0][j] == 0:
            continue
        finish = True
        winner = state.data[0][j]
        for i in range(1, board_height):
            if state.data[i][j] != winner:
                finish = False
                break
        if finish:
            return winner

    # diag
    if (state.data[0][0] == state.data[1][1] and state.data[2][2] == state.data[1][1]) \
    or (state.data[0][2] == state.data[1][1] and state.data[2][0] == state.data[1][1]):
        if state.data[1][1] != 0:
            return state.data[1][1]
    
    # tied
    full = True
    for i in range(board_height):
        for j in range(board_width):
            if state.data[i][j] == 0:
                full = False
    if full:
        return 0

    return -1


class State:
    def __init__(self, data=[[0,0,0],[0,0,0],[0,0,0]]):
        self.data = data
        self.state_idx = 0
        for i in range(board_height):
            for j in range(board_width):
                self.state_idx += 3**(i*3+j) * self.data[i][j]

    def next(self, action):
        i,j,sym = action
        next_state = State()
        if self.data[i][j] == 0:
            next_state.data = np.copy(self.data)
            next_state.data[i][j] = sym
            next_state.state_idx = self.state_idx + 3**(i*3+j) * sym
        return next_state
    
    def print(self):
        for i in range(board_height):
            print(self.data[i][0], self.data[i][1], self.data[i][2])

class agent:
    def __init__(self, sym):
        self.values = [-1 for i in range(3**(board_width*board_height))]
        # self.policy = []
        self.sym = sym

    def act(self, state, eps):
        winner = check_win(state)  
        if self.values[state.state_idx] == -1:
            if winner == self.sym:
                self.values[state.state_idx] = 1.0
            elif winner != -1:
                self.values[state.state_idx] = 0.0
            else:
                self.values[state.state_idx] = 0.5

        candidates = []
        for i in range(board_height):
            for j in range(board_width):
                if state.data[i][j] == 0:
                    next_state = state.next([i, j, self.sym])
                    winner = check_win(next_state)
                    # values initialization
                    if self.values[next_state.state_idx] == -1:
                        if winner == self.sym:
                            self.values[next_state.state_idx] = 1.0
                        elif winner != -1:
                            self.values[next_state.state_idx] = 0.0
                        else:
                            self.values[next_state.state_idx] = 0.5
                    candidates.append([i,j,self.sym, self.values[next_state.state_idx]])
        if len(candidates) == 0:
            return []
        candidates.sort(key=lambda x: x[3], reverse=True)
        action = candidates[0][:3]
        if np.random.rand() < eps:
            idx = np.random.randint(0,len(candidates))
            action = candidates[idx][:3]
        return action

    def train(self, episode, step_size):
        for i in reversed(range(len(episode)-1)):
            diff = self.values[episode[i+1].state_idx] - self.values[episode[i].state_idx]
            self.values[episode[i].state_idx] += step_size * diff

class Environment:
    def __init__(self):
        self.state = State()
        self.winner = -1

    def transition(self, action):
        if self.winner != -1:
            return
        self.state = self.state.next(action)
        self.winner = check_win(self.state)

    def reset(self):
        self.state = State()
        self.winner = -1

    def end(self):
        return self.winner != -1
                        


if __name__ == "__main__":

    # state = State(
    #     [
    #         [1,2,1],
    #         [2,1,1],
    #         [1,2,2],
    #     ]
    # )
    # print(check_win(state))
    # print(state.state_idx)

    # 19695

    agent1 = agent(1)
    agent2 = agent(2)
    env = Environment()
    step_size = 0.1
    eps = 0.1

    player1 = agent1
    player2 = agent2
    for iter in range(100000):
        print(iter)
        env.reset()

        episode1 = []
        episode2 = []
        while not env.end():
            episode1.append(env.state)
            env.transition(player1.act(env.state, eps))
            episode2.append(env.state)
            env.transition(player2.act(env.state, eps))
        episode1.append(env.state)
        episode2.append(env.state)

        player1.train(episode1, step_size)
        player2.train(episode2, step_size)

        # tmp = player1
        # player1 = player2
        # player2 = tmp



    player1 = agent1
    player2 = agent2
    for iter in range(10):
        print(iter)
        env.reset()
        episode1 = []
        episode2 = []

        episode1 = []
        episode2 = []
        while not env.end():
            episode1.append(env.state)
            env.transition(player1.act(env.state, 0))
            print("player 1 :", player1.sym)
            env.state.print()
            episode2.append(env.state)
            env.transition(player2.act(env.state, 0))
            print("player 2 :", player2.sym)
            env.state.print()
        episode1.append(env.state)
        episode2.append(env.state)
        
        if env.winner == player1.sym:
            episode1.append(env.state)
        elif env.winner == player2.sym:
            episode2.append(env.state)

        for state in episode1:
            print(player1.values[state.state_idx], end=' ')
        print()

        for state in episode2:
            print(player2.values[state.state_idx], end=' ')
        print()

        tmp = player1
        player1 = player2
        player2 = tmp