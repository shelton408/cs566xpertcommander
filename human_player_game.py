from the_game.agents.human_agent import HumanAgent
from the_game.env.env import Env


NUM_OF_HUMAN_PLAYERS = 1

# Create Environment
config = {
    'num_players': NUM_OF_HUMAN_PLAYERS,
    'log_filename': './logs/human_agent.log'
}
env = Env(config)

agents = []
for i in range(NUM_OF_HUMAN_PLAYERS):
    agents.append(HumanAgent(i))

env.set_agents(agents)
env.run()
print('GAME OVER!')
