import torch
import numpy as np

class StateParser():
    def __init__(self, env):
        self.env = env