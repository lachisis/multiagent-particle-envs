import diplomacy_basic
from multiagent.core import World, Agent, Landmark

class Scenario(diplomacy_basic.Scenario):

    def make_world(self):
        world = World()
        world.teams = diplomacy_basic.Teams()

        #set world properties
        world.dim_c = 0
        world.dim_p = 2
        world.dim_color = 2

        #agents
        num_agents = 1
        num_landmarks = 6

        #ad the agents
        world.agents = [Agent() for i in range(num_agents)]
        for i,agent in enumerate(world.agents):
            self.setup_new_agent(i, agent, world)

        #add the langmarks
        self.setup_landmarks(world)

        #make initial conditions
        self.reset_world(world)

        return world

    def benchmark_data(self, agent, world):
        #should be all 6 territories
        return len(agent.territory)