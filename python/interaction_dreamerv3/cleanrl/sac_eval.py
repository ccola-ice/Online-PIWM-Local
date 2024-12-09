import random
from typing import Callable

import numpy as np
import torch


def evaluate(
    envs,
    eval_episodes: int,
    Model: torch.nn.Module,
    device,
):
    
    model = Model
    overall_score_list = []
    overall_collision_ticks_list = []
    overall_speed_list = []
    overall_completion_list = []
    overall_rmse_list = []

    episodes = 0
    ep_score_list = [0.]
    ep_collision_ticks_list = [0.]
    ep_speed_list = [0.]
    ep_completion_list = [0.]
    ep_rmse_list = [0.]

    obs = envs.reset()
    while episodes < eval_episodes:
        actions, _, _ = model.get_action(torch.Tensor(obs).to(device))
        actions = actions.detach().cpu().numpy()

        next_obs, rewards, terminated, infos = envs.step(actions)

        ep_score_list.append(rewards.sum())
        ep_collision_ticks_list.append(infos['result'] == 'collision')
        ep_speed_list.append(infos['speed'])
        ep_completion_list.append(infos['completion_rate'])
        ep_rmse_list.append(infos['distance_to_gt'])

        if terminated:
            # calculate some statistic terms for i-sim driving scenes
            score = float(np.array(ep_score_list).sum())
            avg_speed = float(np.array(ep_speed_list).mean())
            collision_ticks = float(np.array(ep_collision_ticks_list).sum())
            completion = float(np.array(ep_completion_list[-1]).mean())
            rmse = float(np.sqrt(ep_rmse_list).mean())
            print(f'Episode has {len(ep_score_list)} steps and return {score:.1f}. Complete {completion:.2f} of the route and average speed is {avg_speed:.1f} m/s. Collision ticks is {collision_ticks}.')
            # reset infos
            episodes += 1
            ep_score_list = [0.]
            ep_collision_ticks_list = [0.]
            ep_speed_list = [0.]
            ep_completion_list = [0.]
            ep_rmse_list = [0.]
            obs = envs.reset()
            # print overall results
            overall_score_list.append(score)
            overall_collision_ticks_list.append(collision_ticks)
            overall_completion_list.append(completion)
            overall_speed_list.append(avg_speed)
            overall_rmse_list.append(rmse)
            collision_rate = np.mean([i> 0 for i in overall_collision_ticks_list])
            success_rate = np.mean([overall_completion_list[i] > 0.9 and overall_collision_ticks_list[i] < 1 for i in range(len(overall_completion_list))])
            print(f'Episode {len(overall_completion_list)} has finished:')
            print(f'Test Average: score is {np.mean(overall_score_list):.2f}, collision ticks is {np.mean(overall_collision_ticks_list):.2f}, compeltion is {np.mean(overall_completion_list):.4f}, average speed is {np.mean(overall_speed_list):.2f}, rmse is {np.mean(overall_rmse_list):.2f}.')
            print(f'Overall result: collision_rate is {collision_rate:.4f}, success_rate is {success_rate:.4f}.')
            print('**********************' * 3)
        else:
            obs = next_obs


if __name__ == "__main__":
    # from huggingface_hub import hf_hub_download
    # model_path = hf_hub_download(repo_id="cleanrl/CartPole-v1-dqn-seed1", filename="q_network.pth")
    
    import os
    import argparse
    from distutils.util import strtobool
    from sac import Encoder, Actor, make_env

    def parse_args():
        # fmt: off
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
        parser.add_argument('--eval', action="store_true", default=True, help='all possible ego vehicles are selected equal times')

        args = parser.parse_args()
        args.seed = random.randint(0, 1000000)
        args.loader_type = 'prediction'
        args.npc_type = 'record'
        args.visualization = False

        # fmt: on
        return args

    args = parse_args()
    encoder_model_path = "/home/gdg/InteractionRL/Dreamer_Inter/python/interaction_dreamerv3/cleanrl/runs/sac_small_11/encoder_checkpoint_500k.h5"
    actor_model_path = "/home/gdg/InteractionRL/Dreamer_Inter/python/interaction_dreamerv3/cleanrl/runs/sac_small_11/actor_checkpoint_500k.h5"
    envs = make_env(args)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    encoder = Encoder(envs).to(device)
    encoder.load_state_dict(torch.load(encoder_model_path, map_location=device))
    encoder.eval()
    actor = Actor(envs, encoder).to(device)
    actor.load_state_dict(torch.load(actor_model_path, map_location=device))
    actor.eval()

    evaluate(
        envs,
        eval_episodes=200,
        Model=actor,
        device=device,
    )
