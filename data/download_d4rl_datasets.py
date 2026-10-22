import os
import gym
import numpy as np

import collections
import pickle

import d4rl

maze_traj_len = {
	'umaze' : 300,
	'medium' : 600,
	'large' : 800
}

def download_d4rl_data():
	datasets = []

	data_dir = 'data'

	print(data_dir)

	if not os.path.exists(data_dir):
		os.makedirs(data_dir)

	for env_name in ['walker2d', 'halfcheetah', 'hopper']:
		for dataset_type in ['medium', 'medium-expert', 'medium-replay']:

			name = f'{env_name}-{dataset_type}-v2'
			pkl_file_path = os.path.join(data_dir, name)

			print("processing: ", name)

			env = gym.make(name)
			dataset = env.get_dataset()

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
					final_timestep = (episode_step == 1000-1)
				for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
					data_[k].append(dataset[k][i])
				if done_bool or final_timestep:
					episode_step = 0
					episode_data = {}
					for k in data_:
						episode_data[k] = np.array(data_[k])
					paths.append(episode_data)
					data_ = collections.defaultdict(list)
				episode_step += 1

			returns = np.array([np.sum(p['rewards']) for p in paths])
			num_samples = np.sum([p['rewards'].shape[0] for p in paths])
			print(f'Number of samples collected: {num_samples}')
			print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

			with open(f'{pkl_file_path}.pkl', 'wb') as f:
				pickle.dump(paths, f)

	for env_size in ['umaze', 'medium', 'large']:
		name = f'maze2d-{env_size}-v1'
		pkl_file_path = os.path.join(data_dir, name)

		print("processing: ", name)

		env = gym.make(name)
		dataset = env.get_dataset()

		N = dataset['rewards'].shape[0]
		data_ = collections.defaultdict(list)

		use_timeouts = False
		if 'timeouts' in dataset:
			use_timeouts = True

		'''
		episode_step = 0
		paths = []
		
		for i in range(N):
			done_bool = bool(dataset['terminals'][i])
			if use_timeouts:
				final_timestep = dataset['timeouts'][i]
			else:
				final_timestep = (episode_step == 1000 - 1)
			for k in ['observations', 'actions', 'rewards', 'terminals', 'infos/goal']:
				data_[k].append(dataset[k][i])
			if done_bool or final_timestep:
				episode_step = 0
				episode_data = {}
				traj_len = len(data_['rewards'])
				# max_traj_len = maze_traj_len[env_size]
				# if data_['rewards'][-1] > 1e-5:
				# 	 data_['rewards'][-1] = data_['rewards'][-1] + np.float32(max_traj_len - traj_len)
				for k in data_:
					episode_data[k] = np.array(data_[k])
				episode_data['observations'] = np.concatenate([episode_data['observations'],
															   episode_data['infos/goal']],
															  axis = 1)
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1
		'''

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
					data_['rewards'][i] = -(
								np.linalg.norm(data_['infos/qpos'][i] - data_['infos/goal'][i], axis=-1) > 0.5).astype(
						np.float32)
				episode_step = 0
				episode_data = {}
				for k in data_:
					episode_data[k] = np.array(data_[k])
				episode_data['observations'] = np.concatenate([episode_data['observations'],
															   episode_data['infos/goal']],
															  axis=1)
				paths.append(episode_data)
				data_ = collections.defaultdict(list)
			episode_step += 1

		returns = np.array([np.sum(p['rewards']) for p in paths])
		num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		print(f'Number of samples collected: {num_samples}')
		print(
			f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')

		with open(f'{pkl_file_path}.pkl', 'wb') as f:
			pickle.dump(paths, f)


if __name__ == "__main__":
	download_d4rl_data()
