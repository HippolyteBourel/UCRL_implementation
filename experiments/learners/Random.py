class Random:
    def __init__(self,env):
        self.env=env

    def name(self):
        return "Random"

    def reset(self,inistate):
        ()

    def play(self,state):
        return self.env.action_space.sample()

    def update(self, state, action, reward, observation):
        ()