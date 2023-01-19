import rsoccer_gym
from sympy import false
from src.pos_latent import PosLatent
import gym 
import torch
import ipdb
import numpy as np

    
WINDOW_SIZE = 10
BATCH_SIZE = 128 # testar 64
HIDDEN_SIZE = 256
WEIGHT_DECAY = 0
MAX_EPOCHS = 128
VAL_INTERVAL = 4
HIDDEN_IS_WITH_NOISE = True
NUM_OF_DATA_TO_TEST = 1000
LEARNING_RATE = 1e-3
FEATURES = 40
AUTOENCODER = True
percentage_to_train = 0.9
WEIGHTS = (0.5, 0.5)
percentage_to_val = 0.05
total_of_data = 50000

model = PosLatent(
    window= WINDOW_SIZE, 
    input_size= FEATURES , 
    hidden_size= HIDDEN_SIZE,
    batch_size=BATCH_SIZE,
    should_test_the_new_data_set=True,
    data_root= '../all_data/data-3v3-v2',
    # num_of_data_to_train = 256, 
    output_size=40,
    act_path = './autoencoder/rnn-autoencoder-v3/21z8udnb/checkpoints/epoch=127-step=60927.ckpt',
    pos_path='./autoencoder/rnn-autoencoder-v3/q9uv6d3w/checkpoints/epoch=127-step=30463.ckpt',
    num_of_data_to_train = 30450,
    # act_path_autoencoder='./autoencoder/rnn-autoencoder-v3/3r58fxvn/checkpoints/epoch=255-step=60927.ckpt',
    # num_of_data_to_train = 256, 
    weights = WEIGHTS,
    lr=LEARNING_RATE,
    # num_of_data_to_val = 256,
    num_of_data_to_val = 1000
    )

# model = model.load_from_checkpoint('./next_positions-old/rnn-predict-next-positions-with-v3/2zxd4xe1/checkpoints/epoch=127-step=60927.ckpt')
model = model.load_from_checkpoint('./next_positions/rnn-predict-next-positions-with-v3/37j6va2n/checkpoints/epoch=127-step=30463.ckpt')

env = gym.make('VSS-v0')

env.reset()
# Run for 1 episode and print reward at the end
for i in range(100):
    done = False
    obs = env.reset()
    steps = []
    should_create = True
    while not done:
        if should_create:
            for i in range(10):
                action = env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                steps.append(next_state)
                should_create = False
        # Step using random actions
        # ipdb.set_trace()
        steps_occured = torch.FloatTensor(steps)
        steps_occured = steps_occured.view((1, 10, 40))
        # steps_occured = torch.stack(torch.FloatTensor(steps))
        obs = torch.FloatTensor(steps_occured).to('cuda')
        _, acts = model(obs)
        action = acts.detach().cpu().numpy()
        actions = v_to_action(action[0][0], action[0][1])
        print(actions)
        next_state, reward, done, _ = env.step(actions)
        steps.append(next_state)
        steps.pop(0)

        env.render()
    # print(reward)
