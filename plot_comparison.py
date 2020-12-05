import pickle
import glob
from matplotlib import pyplot as plt

cnt = 0
files = glob.glob('pickle_files/P*.pickle')
colors = ['red', 'blue', 'green', 'black', 'darkviolet', 'darkturquoise']

for filename in files:
    pickle_in = open(filename,'rb')
    data = pickle.load(pickle_in)
    if filename == 'pickle_files\PPO_single_agent.pickle':
        plt.plot(data['single_agent'], label='PPO single agent', color=(colors[cnt]))
    elif filename == 'pickle_files\PPO_multi_agent1.pickle':
        plt.plot(data['multi_agent1'], label='PPO multi player w/o hints', color=colors[cnt])
    elif filename == 'pickle_files\PPO_multi_agent2.pickle':
        plt.plot(data['multi_agent2'], label='PPO multi player with hints', color=colors[cnt])
    elif filename == 'pickle_files\PG_single_agent.pickle':
        plt.plot(data['single_agent'], label='PG single player', color=colors[cnt])
    elif filename == 'pickle_files\PG_multi_agent1.pickle':
        plt.plot(data['multi_agent1'], label='PG multi player w/o hints', color=colors[cnt])
    elif filename == 'pickle_files\PG_multi_agent2.pickle':
        plt.plot(data['multi_agent2'], label='PG multi player with hints', color=colors[cnt])
    cnt+=1

plt.ylim([0, 98])
plt.xlim([0, 199])
plt.title('Training curve')
plt.xlabel('training iteration')
plt.ylabel('average drawpile size')
plt.grid('on')
plt.legend()
plt.show()
