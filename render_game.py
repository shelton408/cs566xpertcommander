import tkinter as tk
import time
from tkinter import messagebox

import torch
from training.instantiate import instantiate
from training.PPO import PPO
from training.training import Trainer
from training.utils import ParamDict
from utils import plot_learning_curve
from cs566xpertcommander.the_game import Env
from training.policy import Policy
import logging

import warnings
warnings.filterwarnings("ignore")

class Render:
    def __init__(self, master):
        self.master = master
        self.lbl1 = tk.Label(master, text='Increasing piles', font=('Helvetica', 10))
        self.lbl2 = tk.Label(master, text='Decreasing piles', font=('Helvetica', 10))
        self.deck1 = tk.Label(master, text='1', height=3, width=3, background='blue', relief='groove', font=('Courier', 60))
        self.deck2 = tk.Label(master, text='1', height=3, width=3, background='blue', relief='groove', font=('Courier', 60))
        self.deck3 = tk.Label(master, text='100', height=3, width=3, background='yellow', relief='groove', font=('Courier', 60))
        self.deck4 = tk.Label(master, text='100', height=3, width=3, background='yellow', relief='groove', font=('Courier', 60))
        self.drawpile = tk.Label(master, height=10, width=15, text='drawpile', wraplength=100, relief='groove', font=(200))
        self.hand = tk.Label(master, height=10, width=15, text='player hand', relief='groove', font=(200))
        self.lbl1.grid(row=0, column=0, columnspan=2, pady=(10, 10))
        self.lbl2.grid(row=0, column=2, columnspan=2, pady=(10, 10))
        self.deck1.grid(row=2, column=0, padx=(10, 10), pady=(10, 10))
        self.deck2.grid(row=2, column=1, padx=(10, 10), pady=(10, 10))
        self.deck3.grid(row=2, column=2, padx=(10, 10), pady=(10, 10))
        self.deck4.grid(row=2, column=3, padx=(10, 10), pady=(10, 10))
        self.drawpile.grid(row=3, column=0, padx=(10, 10), pady=(20, 20))
        self.hand.grid(row=3, column=1, padx=(10, 10), pady=(20, 20))
        self.idx = 0
        self.update_label()
    
    def update_label(self):
        if self.idx < l:
            action = actions[self.idx]
            card_id, deck_id = (action//4, action % 4)
            if deck_id == 0 and action != 32:
                self.deck1.config(text=int(hands[self.idx][card_id]))
            elif deck_id == 1:
                self.deck2.config(text=int(hands[self.idx][card_id]))
            elif deck_id == 2:
                self.deck3.config(text=int(hands[self.idx][card_id]))
            elif deck_id == 3:
                self.deck4.config(text=int(hands[self.idx][card_id]))
            self.drawpile.config(text='current drawpile:' + str(drawpile[self.idx]))
            self.idx+=1
            i = self.idx % 4
            if i == 0:
                self.deck1.after(100, self.update_label)
            elif i == 1:
                self.deck2.after(100, self.update_label)
            elif i == 2:
                self.deck3.after(100, self.update_label)
            elif i == 3:
                self.deck4.after(100, self.update_label)
        else:
            messagebox.showinfo("RESULT", "Final drawpile size is " + str(drawpile[self.idx - 1]))
            self.master.destroy()

# hyperparameters
policy_params = ParamDict(
    policy_class=Policy,   # Policy class to use (replaced later)
    hidden_dim=128,         # dimension of the hidden state in actor network
    learning_rate=1e-3,    # learning rate of policy update
    batch_size=1024,       # batch size for policy update
    policy_epochs=25,       # number of epochs per policy update
    entropy_coef=0.002    # hyperparameter to vary the contribution of entropy loss
)
params = ParamDict(
    policy_params=policy_params,
    rollout_size=5000,     # number of collected rollout steps per policy update
    num_updates=200,       # number of training policy iterations
    discount=0.99,        # discount factor
    plotting_iters=100,    # interval for logging graphs and policy rollouts
    # env_name=Env(),  # we are using a tiny environment here for testing
)
rollouts, policy = instantiate(params)
policy.actor.load_state_dict(torch.load('./models/policy1.pt'))

NUM_OF_PLAYERS = 1

config = {
    'num_players': NUM_OF_PLAYERS,
    'log_filename': './logs/policy_agent.log',
    'static_drawpile': False
}
logging.basicConfig(filename=config['log_filename'], filemode='w', level=logging.INFO)
env = Env(config)

actions, hands, drawpile = env.run_PG(policy, render=True)
l = len(actions)

root = tk.Tk()
root.title('THE GAME')
Render(root)
root.mainloop()

print('GAME OVER')