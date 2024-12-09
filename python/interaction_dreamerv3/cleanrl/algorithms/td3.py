# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/td3/#td3_continuous_actionpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="I-sim",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=int(2e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-noise", type=float, default=0.2,
        help="the scale of policy noise")
    parser.add_argument("--exploration-noise", type=float, default=0.1,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=2,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    
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
    args.loader_type = 'prediction'
    args.state_frame = 'ego'
    args.npc_type = 'record'
    args.visualization = False

    # fmt: on
    return args

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
    def __init__(self, envs_list):
        self.envs_list = envs_list
        self.num_envs = len(envs_list)
        self.single_observation_space = envs_list[0].single_observation_space
        self.single_action_space = envs_list[0].single_action_space

    def reset(self):
        state = np.zeros((args.num_envs, ) + self.single_observation_space.shape)
        for i in range(len(self.envs_list)):
            env = self.envs_list[i]
            single_state = env.reset()
            state[i] = single_state
        return state
    
    def step(self, action_array):
        state = np.zeros((args.num_envs, ) + self.single_observation_space.shape)
        reward = np.zeros((args.num_envs,))
        done = np.zeros((args.num_envs,))
        info = dict()
        for i in range(len(self.envs_list)):
            # select corresponding env and action
            env = self.envs_list[i]
            action = action_array[i]
            single_state, single_reward, single_done, aux_info = env.step(action)
            state[i] = single_state
            reward[i] = single_reward
            done[i] = single_done
        info = aux_info
        return state, reward, done, info

def make_env(args):
    envs_num = args.num_envs
    envs_list = [Isim(args) for _ in range(envs_num)]
    envs = MultiEnv(envs_list)
    return envs


# ALGO LOGIC: initialize agent here:

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod() + int(envs.single_action_space.n), 400)),
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
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 1), std=1.0),
        )


    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.encoder(x)
        q_vals = self.fc_q(x)
        return q_vals


class Actor(nn.Module):
    def __init__(self, env):
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

        self.fc_logits = nn.Sequential(
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, int(envs.single_action_space.n)), std=0.01),
        )




        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


if __name__ == "__main__":
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
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                    -args.noise_clip, args.noise_clip
                ) * target_actor.action_scale

                next_state_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                    envs.single_action_space.low[0], envs.single_action_space.high[0]
                )
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()
