# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_ataripy
import argparse
import os
import random
import time
from distutils.util import strtobool

# import gymnasium as gym
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import sys
import pathlib
directory = pathlib.Path(__file__).resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name
from interaction_dreamerv3.client_interface import ClientInterface

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="I-sim",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=510000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=int(2e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=2500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=4e4, # 2e4
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    # ENV specific arguments
    parser.add_argument("map_name", type=str, default="DR_USA_Intersection_EP0", help="Name of the scenario (to identify map and folder for track files)", nargs='?')
    parser.add_argument("load_mode", type=str, default="vehicle", help="Dataset to load (vehicle, pedestrian, or both)", nargs='?')
    parser.add_argument("loader_type", type=str, default='prediction', help="prediction or dataset", nargs='?')
    parser.add_argument("state_frame", type=str, default="global", help="Vector state's frame, in ego frame or global frame", nargs='?')

    parser.add_argument("continous_action", type=bool, default=False, help="Is the action type continous or discrete", nargs='?')
    parser.add_argument("control_steering", type=bool, default=False, help="Control both lon and lat motions", nargs='?')
    parser.add_argument("max_steps", type=int, default=None, help="max steps of one episode, None means orifinal max steps of the vehicle in xlm files", nargs='?')
    
    parser.add_argument("npc_type", type=str, default='react', help="Default npc type (react or record)", nargs='?')
    parser.add_argument("npc_num", type=int, default=5, help="Default considered npc num", nargs='?')
    parser.add_argument("other_num", type=int, default=5, help="Default considered far away npc num", nargs='?')
    parser.add_argument("route_type", type=str, default='ground_truth', help="Default route type (predict, ground_truth or centerline)", nargs='?')

    parser.add_argument("visualization", type=bool, default=True, help="Visulize or not", nargs='?')
    parser.add_argument("ghost_visualization", type=bool, default=True, help="Render ghost ego or not", nargs='?')
    parser.add_argument("route_visualization", type=bool, default=True, help="Render ego's route or not", nargs='?')
    parser.add_argument("route_bound_visualization", type=bool, default=False, help="Render ego's route bound or not", nargs='?')
    
    parser.add_argument("--port", type=int, default=8888, help="Number of the port (int)")
    parser.add_argument('--only_trouble', action="store_true", default=False, help='only select troubled vehicles in predictions as ego for testing')
    parser.add_argument('--eval', action="store_true", default=False, help='all possible ego vehicles are selected equal times')

    args = parser.parse_args()
    # args.total_timesteps = int(10.1e5)
    args.loader_type = 'prediction' # 'dataset'
    args.state_frame = 'ego'
    args.npc_type = 'record'
    args.seed = random.randint(0, 1000000)
    args.visualization = False

    # fmt: on
    return args

# COMPARE EXP: isim env
class Isim():
    def __init__(self, args):
        self.args = args
        self.env = ClientInterface(args)
        self.single_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1045, ), dtype=np.float32)
        self.single_action_space = gym.spaces.Discrete(4)
        print(self.args)
    
    def reset(self):
        state_dict = self.env.reset()
        # TODO: we only consider one ego vehicle for now
        for ego_id in state_dict.keys():
            state = state_dict[ego_id]

        order = state['index_npc'] + state['index_other']
        # for zero-padding npc and other
        for i in range(self.args.npc_num):
          if f'npc_{i+1}' not in order:
            order.append(f'npc_{i+1}')
        for i in range(self.args.other_num):
          if f'other_{i+1}' not in order:
            order.append(f'other_{i+1}')
        # concat ego, npcs and others features, npcs and others features are ordered by distance
        value = state['ego'].reshape((-1))
        for key_order in order:
          value = np.concatenate([value, state[key_order].reshape(-1)], axis=0, dtype=np.float32)
            
        return value
    
    def step(self, action):
        # TODO: we only consider one ego vehicle for now
        action_dict = {self.env.ego_id_list[0]: [action.tolist()]}
        state_dict, reward_dict, done_dict, aux_info_dict = self.env.step(action_dict, prediction=None)
        for ego_id in state_dict.keys():
            state = state_dict[ego_id]
            reward = reward_dict[ego_id]
            done = done_dict[ego_id]
            aux_info = aux_info_dict[ego_id]
        
        order = state['index_npc'] + state['index_other']
        # for zero-padding npc and other
        for i in range(self.args.npc_num):
            if f'npc_{i+1}' not in order:
                order.append(f'npc_{i+1}')
        for i in range(self.args.other_num):
            if f'other_{i+1}' not in order:
                order.append(f'other_{i+1}')
        # concat ego and npcs features, npc feature is ordered by distance
        value = state['ego'].reshape((-1))
        for key_order in order:
            value = np.concatenate([value, state[key_order].reshape(-1)], axis=0, dtype=np.float32)

        return value, reward, done, aux_info

class MultiEnv():
    def __init__(self, args, envs_list):
        self.args = args
        self.envs_list = envs_list
        self.num_envs = len(envs_list)
        self.single_observation_space = envs_list[0].single_observation_space
        self.single_action_space = envs_list[0].single_action_space

    def preprocess(self, state):
        return np.sign(state) * np.log(1 + np.abs(state))
    
    def reset(self):
        state = np.zeros((self.args.num_envs, ) + self.single_observation_space.shape)
        for i in range(len(self.envs_list)):
            env = self.envs_list[i]
            single_state = env.reset()
            # state[i] = self.preprocess(single_state)
            state[i] = single_state
        return state
    
    def step(self, action_array):
        state = np.zeros((self.args.num_envs, ) + self.single_observation_space.shape)
        reward = np.zeros((self.args.num_envs,))
        done = np.zeros((self.args.num_envs,))
        info = dict()
        for i in range(len(self.envs_list)):
            # select corresponding env and action
            env = self.envs_list[i]
            action = action_array[i]
            single_state, single_reward, single_done, aux_info = env.step(action)
            # state[i] = self.preprocess(single_state)
            state[i] = single_state
            reward[i] = single_reward
            done[i] = single_done
        info = aux_info
        return state, reward, done, info

def make_env(args):
    envs_num = args.num_envs
    envs_list = [Isim(args) for _ in range(envs_num)]
    envs = MultiEnv(args, envs_list)
    return envs


# ALGO LOGIC: initialize agent here:

# def layer_init(layer, bias_const=0.0):
#     nn.init.kaiming_normal_(layer.weight)
#     torch.nn.init.constant_(layer.bias, bias_const)
#     return layer

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
        )

        self.fc_q = nn.Sequential(
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, int(envs.single_action_space.n))),
        )

    def forward(self, x):
        x = self.encoder(x)
        q_vals = self.fc_q(x)
        return q_vals
    

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    # import stable_baselines3 as sb3

#     if sb3.__version__ < "2.0":
#         raise ValueError(
#             """Ongoing migration: run the following command to install the new dependencies:

# poetry run pip install "stable_baselines3==2.0.0a1" "gymnasium[atari,accept-rom-license]==0.28.1"  "ale-py==0.8.1" 
# """
#         )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = make_env(args)

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        # optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    score_list = []
    collision_ticks_list = []
    speed_list = []
    completion_list = []

    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        score_list.append(rewards.sum())
        collision_ticks_list.append(infos['result'] == 'collision')
        speed_list.append(infos['speed'])
        completion_list.append(infos['completion_rate'])
        if terminated:
            # calculate some statistic terms for i-sim driving scenes
            score = float(np.array(score_list).sum())
            avg_speed = float(np.array(speed_list).mean())
            collision_ticks = float(np.array(collision_ticks_list).sum())
            completion = float(np.array(completion_list[-1]).mean())
            # record them in writer
            writer.add_scalar("episode/score", score, global_step)
            writer.add_scalar("episode/avg_speed", avg_speed, global_step)
            writer.add_scalar("episode/collision_ticks", collision_ticks, global_step)
            writer.add_scalar("episode/completion", completion, global_step)
            print(f'Episode has {len(score_list)} steps and return {score:.1f}. Complete {completion:.2f} of the route and average speed is {avg_speed:.1f} m/s. Collision ticks is {collision_ticks}.')
            # reset infos
            score_list = []
            collision_ticks_list = []
            speed_list = []
            completion_list = []
            obs = envs.reset()
        else:
            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards.flatten() + args.gamma * target_max * (1 - data.dones.flatten())
                old_val = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
        if not global_step % 100e3:
            if args.save_model:
                # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
                model_path = f'runs/{run_name}/checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
                torch.save(q_network.state_dict(), model_path)
                print(f"model saved to {model_path}")

                # from cleanrl_utils.evals.dqn_eval import evaluate

                # episodic_returns = evaluate(
                #     model_path,
                #     make_env,
                #     args.env_id,
                #     eval_episodes=10,
                #     run_name=f"{run_name}-eval",
                #     Model=QNetwork,
                #     device=device,
                #     epsilon=0.05,
                # )
                # for idx, episodic_return in enumerate(episodic_returns):
                #     writer.add_scalar("eval/episodic_return", episodic_return, idx)

                # if args.upload_model:
                #     from cleanrl_utils.huggingface import push_to_hub

                #     repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
                #     repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
                #     push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")
                #     print('model uploaded to HF')

    envs.close()
    writer.close()
