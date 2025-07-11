from src.d4rl_utils import kitchen_render
from typing import Dict
import jax
import gym
import numpy as np
from collections import defaultdict
import time
from tqdm import trange


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """
    Wrapper that supplies a jax random key to a function (using keyword `seed`).
    Useful for stochastic policies that require randomness.

    Similar to functools.partial(f, seed=seed), but makes sure to use a different
    key for each new call (to avoid stale rng keys).

    """

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """
    Helper function that flattens a dictionary of dictionaries into a single dictionary.
    E.g: flatten({'a': {'b': 1}}) -> {'a.b': 1}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def env_reset(env_name, env, goal_info, base_observation):
    observation, done = env.reset(), False
    # if policy_type == 'random_skill' and 'antmaze' in env_name:
    #     observation[:2] = [20, 8]
    #     env.set_state(observation[:15], observation[15:])

    if 'antmaze' in env_name:
        goal = env.wrapped_env.target_goal
        obs_goal = np.concatenate([goal, base_observation[-27:]])
    elif 'kitchen' in env_name:
        if 'visual' in env_name:
            observation = kitchen_render(env)
            obs_goal = goal_info['ob']
        else:
            observation, obs_goal = observation[:30], observation[30:]
            obs_goal[:9] = base_observation[:9]
    elif 'calvin' in env_name:
            observation = observation['ob']
            goal = np.array([0.25, 0.15, 0, 0.088, 1, 1])
            obs_goal = base_observation.copy()
            obs_goal[15:21] = goal
    else:
        raise NotImplementedError

    return observation, obs_goal


def env_step(env_name, env, action):
    if 'antmaze' in env_name:
        next_observation, reward, done, info = env.step(action)
    elif 'kitchen' in env_name:
        next_observation, reward, done, info = env.step(action)
        if 'visual' in env_name:
            next_observation = kitchen_render(env)
        else:
            next_observation = next_observation[:30]
    elif 'calvin' in env_name:
        next_observation, reward, done, info = env.step({'ac': np.array(action)})
        next_observation = next_observation['ob']
        del info['robot_info']
        del info['scene_info']
    elif 'procgen' in env_name:
        if np.random.random() < 0.05:
            action = np.random.choice([2, 3, 5, 6])

        next_observation, r, done, info = env.step(np.array([action]))
        next_observation = next_observation[0]
        reward = 0.
        done = done[0]
        info = dict()
    else:
        raise NotImplementedError

    return next_observation, reward, done, info


def get_frame(env_name, env):
    if 'antmaze' in env_name:
        size = 200
        cur_frame = env.render(mode='rgb_array', width=size, height=size).transpose(2, 0, 1).copy()
    elif 'kitchen' in env_name:
        cur_frame = kitchen_render(env, wh=100).transpose(2, 0, 1)
    elif 'calvin' in env_name:
        cur_frame = env.render(mode='rgb_array').transpose(2, 0, 1)
    else:
        raise NotImplementedError
    return cur_frame


def add_episode_info(env_name, env, info, trajectory):
    if 'antmaze' in env_name:
        info['final_dist'] = np.linalg.norm(trajectory['next_observation'][-1][:2] - env.wrapped_env.target_goal)
    elif 'kitchen' in env_name:
        info['success'] = float(info['episode']['return'] == 4.0)
    elif 'calvin' in env_name:
        info['return'] = sum(trajectory['reward'])
    elif 'procgen' in env_name:
        info['return'] = sum(trajectory['reward'])
    else:
        raise NotImplementedError


def evaluate_with_trajectories(
        agent, env: gym.Env, goal_info, env_name, num_episodes, base_observation=None, num_video_episodes=0,
) -> Dict[str, float]:
    policy_fn = supply_rng(agent.sample_skill_actions)

    trajectories = []
    stats = defaultdict(list)

    renders = []
    for i in trange(num_episodes + num_video_episodes):
        trajectory = defaultdict(list)

        if 'procgen' in env_name:
            from src.envs.procgen_env import ProcgenWrappedEnv
            from src.envs.procgen_viz import ProcgenLevel
            eval_level = goal_info['eval_level']
            cur_level = eval_level[np.random.choice(len(eval_level))]

            level_details = ProcgenLevel.create(cur_level)
            border_states = [i for i in range(len(level_details.locs)) if
                             len([1 for j in range(len(level_details.locs)) if
                                  abs(level_details.locs[i][0] - level_details.locs[j][0]) + abs(
                                      level_details.locs[i][1] - level_details.locs[j][1]) < 7]) <= 2]
            target_state = border_states[np.random.choice(len(border_states))]
            goal_img = level_details.imgs[target_state]
            goal_loc = level_details.locs[target_state]
            env = ProcgenWrappedEnv(1, 'maze', cur_level, 1)

            from src.envs.procgen_viz import get_xy_single
            observation, done = env.reset(), False
            observation = observation[0]
            obs_goal = goal_img
        else:
            observation, obs_goal = env_reset(env_name, env, goal_info, base_observation)

        done = False

        render = []
        step = 0
        skill = None
        while not done:
            policy_obs = observation
            policy_goal = obs_goal

            phi_obs, phi_goal = agent.get_phi(np.array([policy_obs, policy_goal]))
            skill = (phi_goal - phi_obs) / np.linalg.norm(phi_goal - phi_obs)
            if 'procgen' in env_name:
                policy_obs = policy_obs.reshape(1, 64, 64, 3)
                skill = skill.reshape(1, skill.shape[0])
            action = policy_fn(observations=policy_obs, skills=skill, temperature=0.)

            action = np.array(action)
            if 'procgen' in env_name:
                action = action.squeeze(-1)

            next_observation, reward, done, info = env_step(env_name, env, action)
            step += 1

            # Render
            if 'procgen' in env_name:
                loc = get_xy_single(next_observation)
                if np.linalg.norm(loc - goal_loc) < 4:
                    r = 1.
                    done = True

                cur_render = next_observation
                cur_frame = cur_render.transpose(2, 0, 1).copy()
                cur_frame[2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 255
                cur_frame[:2, goal_loc[1]-1:goal_loc[1]+2, goal_loc[0]-1:goal_loc[0]+2] = 0
                render.append(cur_frame)
            else:
                if i >= num_episodes and step % 3 == 0:
                    cur_frame = get_frame(env_name, env)
                    render.append(cur_frame)
                transition = dict(
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    done=done,
                    skill=skill,
                    info=info,
                )
                if i < num_episodes:
                    add_to(trajectory, transition)
                    add_to(stats, flatten(info))
                observation = next_observation

        if i < num_episodes:
            add_episode_info(env_name, env, info, trajectory)
            add_to(stats, flatten(info, parent_key="final"))
            trajectories.append(trajectory)
        else:
            renders.append(np.array(render))

    scalar_stats = {}
    for k, v in stats.items():
        scalar_stats[k] = np.mean(v)
    return scalar_stats, trajectories, renders


class EpisodeMonitor(gym.ActionWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action: np.ndarray):
        observation, reward, done, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}

        if done:
            info["episode"] = {}
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time

            if hasattr(self, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.get_normalized_score(info["episode"]["return"]) * 100.0
                )

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        self._reset_stats()
        return self.env.reset()
