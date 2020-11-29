import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def plot_learning_curve(evals, num_it, num_cards=98):
    plt.figure()
    plt.plot(evals, label='num cards left')
    plt.ylim([0, num_cards])
    plt.xlim([0, num_it - 1])
    plt.xlabel('train iter')
    plt.grid('on')
    plt.show()

def plot_testing(evals, num_games, num_cards=98):
    mean = np.mean(evals)
    plt.figure()
    plt.plot(evals, label='num cards left')
    plt.plot(np.ones(len(evals)) * mean, label='average num cards left')
    plt.ylim([0, num_cards])
    plt.xlim([0, num_games - 1])
    plt.xlabel('Game')
    plt.grid('on')
    plt.show()
