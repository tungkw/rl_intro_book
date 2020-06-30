import MC
import agent

class myAgent(agent.Agent):
    def __init__(self):
        super().__init__((21-12+1)*10*2, 2, 1.0)


if __name__ == "__main__":
    test = myAgent()

    method = MC.algo(test, 0.0001)
    method.MC_control(10000)#show=True)
    test.print_evaluation()
    test.print_improvement()