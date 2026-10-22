import matplotlib.pyplot as plt
import gym, d4rl, torch
import collections
import numpy as np
import time

def get_traj(dataset):
    N = dataset['rewards'].shape[0]
    data_ = collections.defaultdict(list)

    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    paths = []
    for i in range(N):
        done_bool = bool(dataset['terminals'][i])
        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == 1000 - 1)
        for k in ['observations', 'actions', 'rewards', 'terminals', 'infos/goal', 'infos/qpos', 'infos/qvel']:
            data_[k].append(dataset[k][i])
        if done_bool or final_timestep:
            for i in range(len(data_['observations'])):
                data_['rewards'][i] = -(np.linalg.norm(data_['infos/qpos'][i] - data_['infos/goal'][i], axis = -1) > 0.5).astype(np.float32)
            episode_step = 0
            episode_data = {}
            for k in data_:
                episode_data[k] = np.array(data_[k])
            paths.append(episode_data)
            data_ = collections.defaultdict(list)
        episode_step += 1

    return paths


def view(traj, ax):

    N = traj['rewards'].shape[0]
    point, = ax.plot([], [], marker='o', color = 'blue', markersize=2)
    goal_point, = ax.plot([], [], marker='o', color='green', markersize=2)
    for i in range(N):
        rew = traj['rewards'][i]
        color = 'red' if rew > 0.5 else 'blue'
        loc = traj['observations'][i][:2]

        point.set_data(loc)
        point.set_color(color)
        # print(loc)
        # plt.draw()
        goal_point.set_data(traj['infos/goal'][i])
        goal_point.set_color('green')
        plt.pause(0.002)



if __name__ == '__main__':

    env_name = 'maze2d-medium-v1'
    env = gym.make(env_name)
    dataset = env.get_dataset()
    trajs = get_traj(dataset)
    print(trajs[0]['rewards'])
    print(len(trajs))
    # print(trajs[0].keys())
    canvas, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 7)
    plt.show(block=False)
    plt.ion()
    for i in range(len(trajs)):
        view(trajs[i], ax)
        rew = trajs[i]['rewards']
        print(f"Traj {i+1} terminated\t length = {rew.shape[0]}\t rew = {np.sum(rew)}")
        plt.pause(1)

    plt.ioff()


