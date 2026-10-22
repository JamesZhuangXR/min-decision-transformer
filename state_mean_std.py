import torch, d4rl, gym
import numpy as np

if __name__ == '__main__':
    env_name = 'maze2d-medium-v1'
    env = gym.make(env_name)
    dataset = env.get_dataset()
    mean_data = np.mean(dataset['observations'], axis = 0)
    std_data = np.std(dataset['observations'], axis = 0)
    file_name = f'{env_name}.txt'
    with open(file_name, 'w') as f:
        f.write('mean:\n')
        count = 0
        for i in mean_data:
            count += 1
            f.write('{:10}'.format(i))
            if count % 3 == 0:
                f.write(',\n')
            else:
                f.write(',\t')
        f.write('\nstd:\n')
        count = 0
        for i in std_data:
            count += 1
            f.write('{:10}'.format(i))
            if count % 3 == 0:
                f.write(',\n')
            else:
                f.write(',\t')
