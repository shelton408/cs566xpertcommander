import pickle
import glob
from matplotlib import pyplot as plt

cnt = 0
files = glob.glob('pickle_files/*.pickle')
colors = ['red', 'blue', 'green']

for filename in files:
    pickle_in = open(filename,'rb')
    data = pickle.load(pickle_in)
    if filename == 'pickle_files\single_agent.pickle':
        plt.plot(data['single_agent'], label='single agent', color=colors[cnt])
    elif filename == 'pickle_files\multi_agent1.pickle':
        plt.plot(data['multi_agent1'], label='multi player w/o hints', color=colors[cnt])
    elif filename == 'pickle_files\multi_agent2.pickle':
        plt.plot(data['multi_agent2'], label='multi player with hints', color=colors[cnt])
    cnt+=1

plt.ylim([0, 98])
plt.xlim([0, 200])
plt.title('Training curve')
plt.xlabel('training iteration')
plt.ylabel('average drawpile size')
plt.grid('on')
plt.legend()
plt.show()
