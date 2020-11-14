from the_game import RandomAgent
from the_game import Env
from utils import plot_learning_curve

NUM_OF_RANDOM_PLAYERS = 1
NUM_OF_CARDS = 50
# Create Environment
config = {
    'num_players': NUM_OF_RANDOM_PLAYERS,
    'log_filename': './logs/random_agent.log',
    # 'total_num_cards': NUM_OF_CARDS,
    'static_drawpile': True
}
env = Env(config)

agents = []
for i in range(NUM_OF_RANDOM_PLAYERS):
    agents.append(RandomAgent(i))

env.set_agents(agents)
evaluations = []
num_iter = 50
for i in range(num_iter):  # lets play 50 games
    env.run()
    evaluations.append(env.get_num_cards_in_drawpile())
print('GAME OVER!')
plot_learning_curve(evaluations, num_iter)
