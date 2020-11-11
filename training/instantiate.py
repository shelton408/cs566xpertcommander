import copy
from training.training import RolloutStorage


def instantiate(params_in, seed=123):
    '''
    SETTING SEED: it is good practice to set seeds when running experiments to keep results comparable
    '''
    import numpy as np
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    params = copy.deepcopy(params_in)

    '''
    1. Instantiate the environment
    # If an environment is given as input to this function, we directly use that.
    # Else, we use the environment specified in `params`.
    '''
    # if nonwrapped_env is None:
    #     nonwrapped_env = gym.make(params.env_name)
    # env = FlatObsWrapper(nonwrapped_env)
    obs_size = 4
    num_actions = 8
    # env.seed(seed)   # Required for reproducibility in stochastic environments.

    '''
    2. Instantiate Rollout Buffer and Policy
    '''
    rollouts = RolloutStorage(params.rollout_size, 4)  # obs_size = 8
    policy_class = params.policy_params.pop('policy_class')

    policy = policy_class(4, num_actions, **params.policy_params)  # obs_size = 8
    return rollouts, policy
