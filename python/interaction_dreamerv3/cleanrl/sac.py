# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_ataripy
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
# from stable_baselines3.common.atari_wrappers import (
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )
from stable_baselines3.common.buffers import ReplayBuffer
from torch.distributions.categorical import Categorical
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
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="I-sim",
        help="the id of the environment")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--total-timesteps", type=int, default=510000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(2e6),
        help="the replay memory buffer size") # smaller than in original paper but evaluation is done only for 100k steps anyway
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.0,
        help="target smoothing coefficient (default: 1)") # Default is 1 to perform replacement update
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--learning-starts", type=int, default=2e4,
        help="timestep to start learning")
    parser.add_argument("--policy-lr", type=float, default=4e-5,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=1e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--update-frequency", type=int, default=8,
        help="the frequency of training updates")
    parser.add_argument("--target-network-frequency", type=int, default=1000,
        help="the frequency of updates for the target networks")
    parser.add_argument("--alpha", type=float, default=0.2,
        help="Entropy regularization coefficient.")
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=True,
        help="automatic tuning of the entropy coefficient")
    parser.add_argument("--target-entropy-scale", type=float, default=0.89,
        help="coefficient for scaling the autotune entropy target")

    # ENV specific arguments
    parser.add_argument("map_name", type=str, default="DR_USA_Intersection_EP0", help="Name of the scenario (to identify map and folder for track files)", nargs='?')
    parser.add_argument("load_mode", type=str, default="vehicle", help="Dataset to load (vehicle, pedestrian, or both)", nargs='?')
    parser.add_argument("loader_type", type=str, default='prediction', help="prediction or dataset", nargs='?')
    parser.add_argument("state_frame", type=str, default="global", help="Vector state's frame, in ego frame or global frame", nargs='?')

    parser.add_argument("drive_as_record", type=bool, default=False, nargs='?')
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
    args.seed = random.randint(0, 1000000)
    args.loader_type = 'prediction'
    args.state_frame = 'ego'
    args.npc_type = 'record'
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
            state[i] = self.preprocess(single_state)
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
            state[i] = self.preprocess(single_state)
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
# NOTE: Sharing a CNN encoder between Actor and Critics is not recommended for SAC without stopping actor gradients
# See the SAC+AE paper https://arxiv.org/abs/1910.01741 for more info
# TL;DR The actor's gradients mess up the representation when using a joint encoder

def layer_init(layer, bias_const=0.0):
    nn.init.kaiming_normal_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Encoder(nn.Module):
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
    
    def forward(self, x):
        x = self.encoder(x)
        return x


class SoftQNetwork(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        # )
        self.encoder = encoder

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


class Actor(nn.Module):
    def __init__(self, envs, encoder):
        super().__init__()
        # self.encoder = nn.Sequential(
        #     layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        #     layer_init(nn.Linear(400, 400)),
        #     nn.ReLU(),
        # )
        self.encoder = encoder

        self.fc_logits = nn.Sequential(
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, int(envs.single_action_space.n))),
        )

    def forward(self, x):
        x = self.encoder(x).detach()
        logits = self.fc_logits(x)
        return logits

    def get_action(self, x):
        logits = self(x)
        policy_dist = Categorical(logits=logits)
        action = policy_dist.sample()
        # Action probabilities for calculating the adapted soft-Q loss
        action_probs = policy_dist.probs
        log_prob = F.log_softmax(logits, dim=1)
        return action, log_prob, action_probs


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
    # envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = make_env(args)

    encoder = Encoder(envs).to(device)
    actor = Actor(envs, encoder).to(device)
    qf1 = SoftQNetwork(envs, encoder).to(device)
    qf2 = SoftQNetwork(envs, encoder).to(device)
    qf1_target = SoftQNetwork(envs, encoder).to(device)
    qf2_target = SoftQNetwork(envs, encoder).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    # TRY NOT TO MODIFY: eps=1e-4 increases numerical stability
    # q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr, eps=1e-4)
    # actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, eps=1e-4)
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -args.target_entropy_scale * torch.log(1 / torch.tensor(envs.single_action_space.n))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        # a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, eps=1e-4)
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
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
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        score_list.append(rewards.sum())
        collision_ticks_list.append(infos['result'] == 'collision')
        speed_list.append(infos['speed'])
        completion_list.append(infos['completion_rate'])
        if dones:
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
            if global_step % args.update_frequency == 0:
                data = rb.sample(args.batch_size)
                # CRITIC training
                with torch.no_grad():
                    _, next_state_log_pi, next_state_action_probs = actor.get_action(data.next_observations)
                    qf1_next_target = qf1_target(data.next_observations)
                    qf2_next_target = qf2_target(data.next_observations)
                    # we can use the action probabilities instead of MC sampling to estimate the expectation
                    min_qf_next_target = next_state_action_probs * (
                        torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                    )
                    # adapt Q-target for discrete Q-function
                    min_qf_next_target = min_qf_next_target.sum(dim=1)
                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target)

                # use Q-values only for the taken actions
                qf1_values = qf1(data.observations)
                qf2_values = qf2(data.observations)
                qf1_a_values = qf1_values.gather(1, data.actions.long()).view(-1)
                qf2_a_values = qf2_values.gather(1, data.actions.long()).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # ACTOR training
                _, log_pi, action_probs = actor.get_action(data.observations)
                with torch.no_grad():
                    qf1_values = qf1(data.observations)
                    qf2_values = qf2(data.observations)
                    min_qf_values = torch.min(qf1_values, qf2_values)
                # no need for reparameterization, the expectation can be calculated for discrete actions
                actor_loss = (action_probs * ((alpha * log_pi) - min_qf_values)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                if args.autotune:
                    # re-use action probabilities for temperature loss
                    alpha_loss = (action_probs.detach() * (-log_alpha * (log_pi + target_entropy).detach())).mean()

                    a_optimizer.zero_grad()
                    alpha_loss.backward()
                    a_optimizer.step()
                    alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
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
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        if not global_step % 100e3:
            # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
            encoder_model_path = f'runs/{run_name}/encoder_checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
            torch.save(encoder.state_dict(), encoder_model_path)
            actor_model_path = f'runs/{run_name}/actor_checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
            torch.save(actor.state_dict(), actor_model_path)
            qf1_model_path = f'runs/{run_name}/qf1_checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
            torch.save(qf1.state_dict(), qf1_model_path)
            print(f"model saved to {qf1_model_path}")
            qf2_model_path = f'runs/{run_name}/qf2_checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
            torch.save(qf2.state_dict(), qf2_model_path)
            print(f"model saved to {qf2_model_path}")

    envs.close()
    writer.close()
