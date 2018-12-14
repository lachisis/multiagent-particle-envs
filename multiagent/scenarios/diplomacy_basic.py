import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
import random

class Teams:
  def __init__(self):
    self.teams = []

  def add_agent_to_new_team(self, agent):
    self.teams.append(set([agent]))

  def combine_teams(self, agent1, agent2):
    #combine the two teams of these agents, without permission from teammates
    team1 = self.__find_agent_team(agent1)
    team2 = self.__find_agent_team(agent2)

    team2 = team2.union(team1)
    team1 = team1.clear()

  def make_new_team(self, agents):
    #put all these agents on a new team, without permission from teammates
    #note: you can call this with one agent + call combine_teams to make only
    #a single agent defect
    team_new = set()
    for agent in agents:
      team = self.__find_agent_team(agents)
      team.pop(agent)
      team_new.add(agent)

    self.teams.append(team_new)

  def __clear_empty_teams(self):
    self.teams = [team for team in self.teams if len(team) > 0]

  def __find_agent_team(self,agent):
    for team in self.teams:
      if agent in team:
        return team
    raise(ValueError('agent isnt on any team'))

  def are_adversaries(self, agent1,agent2):
    team1 = self.__find_agent_team(agent1)
    team2 = self.__find_agent_team(agent2)
    return team1 != team2

class Territories:
  def __init__(self, landmarks):
    self.landmarks = {ld:None for ld in landmarks}

  def takeover(self,agent,ld):
    self.landmarks[ld] = agent

  def is_owner(self,agent,ld):
    return self.landmarks[ld] == agent

  def get_owner(self,ld):
      return self.landmarks[ld]

class Scenario(BaseScenario):
  COLOURS = [[0.85,0.35,0.35],
            [0.35,0.35,0.85],
            [0.35,0.85,0.35],
            [0.15,0.65,0.15],
            [0.15,0.15,0.65],
            [0.65,0.15,0.15],
            [0.15,0.15,0.15]]

  def setup_new_agent(self, agent, world):
      agent.name = 'agent %d' %i
      agent.collide = True
      agent.silent = True
      #agent.adversary = True
      agent.size = 0.15
      agent.original_size = 0.15
      agent.n_landmarks = 0
      agent.territory = set()
      agent.collisions = set()
      agent.size_zero_flag = False

      #size, accel, max_speed = defaults
      world.teams.add_agent_to_new_team(agent)

  def setup_landmarks(self, world):
    world.landmarks = [Landmark() for i in range(num_landmarks)]
    for i, landmark in enumerate(world.landmarks):
      landmark.name = 'landmark %d' %i
      landmark.collide = True
      landmark.movable = False
      landmark.original_size = landmark.size
      #landmark.size = 0
      landmark.boundary = False
    world.territories = Territories(world.landmarks)

  def make_world(self):

    world = World()
    world.teams = Teams()

    #set any world properties
    world.dim_c = 0 #no communication channel
    world.dim_p = 2 #position dimenstionality
    world.dim_color = 2 #number of teams
    #world.collaborative = False #????? since they don't share rewards

    #agents
    num_agents = 4#7
    num_landmarks = 6#34

    #add the agents
    world.agents = [Agent() for i in range(num_agents)]
    for i, agent in enumerate(world.agents):
        self.setup_new_agent(i,agent, world)


    #add the landmarks
    self.setup_landmarks(world)

    #make intial conditions
    self.reset_world(world)
    
    return world

  def reset_world(self, world):

    # set agent teams
    for i, agent in enumerate(world.agents):
      agent.color = np.array(Scenario.COLOURS[i]) #make them all different
      agent.size = 0.15


    #landmarks
    #for i, landmark in enumerate(world.landmarks):

    #initial states
    for agent in world.agents:
      agent.state.p_pos = np.random.uniform(-1,+1,world.dim_p)
      agent.state.p_vel = np.zeros(world.dim_p)
      agent.state.c = np.zeros(world.dim_c)

    for i, landmark in enumerate(world.landmarks):
      landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
      landmark.state.p_vel = np.zeros(world.dim_p)
      landmark.color = np.array(Scenario.COLOURS[-1])


  def benchmark_data(self,agent, world):
    pass


  def teams(self,world):
    return self.world.teams

  def is_collision(self, agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
    return True if dist < dist_min else False


  def reward(self,agent,world):
    rew = 0
    #agents are rewarded for gaining new territory and for losing territory
    #add shape clause to affect reward basedon distance from homebase

    #agent.collisions.clear()

    if agent.collide:

        for ag in world.agents:
            if self.is_collision(agent,ag):
              if world.teams.are_adversaries(agent,ag):
                #combat - randomly make one of them smaller
                #
                #if already calculated
                if(not ag in agent.collisions):
                    agent.collisions.add(ag)
                    #agent.collisions.add(ag)
                    neg_rew_agent, neg_rew_ag = prob_shrink(agent, ag, shrink_size = 0.02)

                    agent.size_zero_flag |= neg_rew_agent
                    ag.size_zero_flag |= neg_rew_ag
                else:
                    agent.collisions.remove(ag)
            #if(agent in ag.collisions):
            #    #don't collide3
            #    ag.collisions.remove(agent)
            #    agent.collisions.remove(ag)
            if(ag in agent.collisions):
                #undo it
                agent.collisions.remove(ag)
                # if p < 0.5:
                #   agent.size -= 0.01
                # else:
                #   ag.size -= 0.01
                #
        if agent.size_zero_flag:
            #agent.size_zero_flag
            agent.size_zero_flag = False
            rew -= 50

        had_landmark_collision = False
        for ld in world.landmarks:
            if self.is_collision(agent,ld):
                old_owner = world.territories.get_owner(ld)
                world.territories.takeover(agent,ld)
                ld.color = agent.color
                if(ld not in agent.territory):
                    agent.territory.add(ld)
                    rew += 5
                    had_landmark_collision = True
        if(not had_landmark_collision):
            rew -= 0.5


    #for negative reward, see if anybody collided with our territory
    for ag in world.agents:
        if ag == agent:
            continue
        if ag.collide:
            for ld in world.landmarks:
                if(self.is_collision(ag,ld)):
                    if(ld in agent.territory):
                        agent.territory.remove(ld)
                        rew -=5

    # agents are penalized for exiting the screen, so that they can be caught by the adversaries
    def bound(x):
        if x < 0.9:
            return 0
        if x < 1.0:
            return (x - 0.9) * 10
        return min(np.exp(2 * x - 2), 10)
    for p in range(world.dim_p):
        x = abs(agent.state.p_pos[p])
        rew -= bound(x)

    return rew


  def observation(self,agent,world):
    # get positions of all entities in this agent's reference frame
    entity_pos = []
    for entity in world.landmarks:
        if not entity.boundary:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)

    #landmark ownership
    entity_own = []
    for entity in world.landmarks:
        if not entity.boundary:
            entity_own.append([1,0,0] if entity in agent.territory else [0,1,0]) #mine, enemy, ally
    # communication of all other agents
    comm = []
    other_pos = []
    other_vel = []
    other_size = []
    for other in world.agents:
        if other is agent: continue
        comm.append(other.state.c)
        other_pos.append(other.state.p_pos - agent.state.p_pos)
        other_size.append([other.size,0,0])
        #if not other.adversary:
        if not world.teams.are_adversaries(agent,other):
          other_vel.append(other.state.p_vel)

    return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + [[agent.size,0,0]] + entity_pos + entity_own + other_size + other_pos + other_vel)

def get_threshold_for_prob_shrink(size_agent, size_ag):
    val = min(size_ag,size_agent)/float(max(size_ag,size_agent))
    val = 0.5 + (1-val)
    val = max([val,0.9])
    return val

def prob_shrink(agent, ag, shrink_size = 0.01):
    p = random.random()
    val = get_threshold_for_prob_shrink(agent.size, ag.size)

    #above the thresdhold - shrink the smaller one
    if p < val:
        if(agent.size <= ag.size):
            #agent is smaller, shrink him
            agent.size -= shrink_size
        else:
            ag.size -= shrink_size
    #below the threshold - shrink the bigger one (rarer)
    else:
        if(agent.size <= ag.size):
            ag.size -= shrink_size
        else:
            agent.size -= shrink_size

    if agent.size <= 0:
        neg_agent_rew = True
        agent.size = 0.02
    else:
        neg_agent_rew = False

    if ag.size <= 0:
        neg_ag_rew = True
        ag.size = 0.02
    else:
        neg_ag_rew = False

    return neg_agent_rew, neg_ag_rew

if __name__ == "__main__":
    assert(get_threshold_for_prob_shrink(14,15) == 0.5+1/float(15)), get_threshold_for_prob_shrink(14,15)
