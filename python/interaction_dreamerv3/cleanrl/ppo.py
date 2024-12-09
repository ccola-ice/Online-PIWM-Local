# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

# COMPARE EXP: i-sim env import
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
    parser.add_argument("--total-timesteps", type=int, default=510000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=1024,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    
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
    args.loader_type = 'dataset'
    args.state_frame = 'ego'
    args.npc_type = 'record'
    args.seed = random.randint(0, 1000000)
    args.visualization = False

    # args.batch_size = int(args.num_envs * args.num_steps)
    args.batch_size = args.num_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


# COMPARE EXP: isim env
class Isim():
    def __init__(self, args):
        self.args = args
        self.env = ClientInterface(args)
        self.single_observation_space_shape = (1045,)
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
        self.single_observation_space_shape = envs_list[0].single_observation_space_shape
        self.single_action_space = envs_list[0].single_action_space

    def reset(self):
        state = np.zeros((self.args.num_envs, ) + self.single_observation_space_shape)
        for i in range(len(self.envs_list)):
            env = self.envs_list[i]
            single_state = env.reset()
            state[i] = single_state
        return state
    
    def step(self, action_array):
        state = np.zeros((self.args.num_envs, ) + self.single_observation_space_shape)
        reward = np.zeros((self.args.num_envs,))
        done = np.zeros((self.args.num_envs,))
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
    envs = MultiEnv(args, envs_list)
    return envs

# COMPARE EXP: 
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.encoder = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space_shape).prod(), 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 1), std=1.0),
        )

        self.actor = nn.Sequential(
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, 400)),
            nn.ReLU(),
            layer_init(nn.Linear(400, int(envs.single_action_space.n)), std=0.01),
        )

    def get_value(self, x):
        embedding = self.encoder(x)
        value = self.critic(embedding)
        return value

    def get_action_and_value(self, x, action=None):
        embedding = self.encoder(x)
        logits = self.actor(embedding)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(embedding)


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
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    # )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    envs = make_env(args)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    # COMPARE EXP: store episode infos
    score_list = []
    collision_ticks_list = []
    speed_list = []
    completion_list = []

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        # Collect interaction data
        for step in range(0, args.num_steps): # args.num_steps is the on-policy batch size
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            # COMPARE EXP: record episode info
            score_list.append(reward)
            collision_ticks_list.append(info['result'] == 'collision')
            speed_list.append(info['speed'])
            completion_list.append(info['completion_rate'])
            if done:
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
                next_obs = torch.Tensor(envs.reset()).to(device)
                
            if not global_step % 100e3:
                # model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
                model_path = f'runs/{run_name}/checkpoint_' + str(int(global_step / 100e3)) + '00k.h5'
                torch.save(agent.state_dict(), model_path)
                print(f"model saved to {model_path}")

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # env.close()
    # writer.close()
