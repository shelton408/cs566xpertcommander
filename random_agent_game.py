from the_game import RandomAgent
from the_game import Env

NUM_OF_RANDOM_PLAYERS = 2

# Create Environment
config = {
    'num_players': NUM_OF_RANDOM_PLAYERS,
    'log_filename': './logs/random_agent.log'
}
env = Env(config)

agents = []
for i in range(NUM_OF_RANDOM_PLAYERS):
    agents.append(RandomAgent(i))

env.set_agents(agents)
env.run()
print('GAME OVER!')
