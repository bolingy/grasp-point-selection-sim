import isaacgym
import isaacgymenvs
from utils import evaluate_policy, str2bool
from datetime import datetime
import gymnasium as gym
import os, shutil
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
from PPO import PPO_discrete
import argparse
import torch
from gym.spaces import Discrete
from rl.sac import *
from rl.rl_utils import *
from rl.sac_replay_memory import *
import wandb
wandb.login()


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument("--set_visible_cuda_devices", type=str, default="0", help = "Select GPU DEVICES (CUDA) for training")
parser.add_argument("--sim_device", type=str, default="cuda:0", help = "Select GPU DEVICE ID (CUDA) for simulation")
parser.add_argument("--sim_device_id", type=int, default=0, help = "Select GPU DEVICE ID (CUDA) for simulation")
parser.add_argument('--train_device', type=int, default=0, help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='Full_Nocam or Full')
parser.add_argument('--num_envs', type=int, default=1, help='number of envs')
parser.add_argument('--write', type=str2bool, default=False, help='Use wandb to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--res_net', type=str2bool, default=False, help='Use resnet or not')
parser.add_argument('--LoadModel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelName', type=str, default='DDQN_2023-12-06_03-56-03', help='Model name to load')
parser.add_argument('--ModelIdx', type=int, default=250*1000, help='which model to load')

parser.add_argument('--seed', type=int, default=209, help='random seed')
parser.add_argument('--data_augmentation_prob', type=float, default=0., help='data augmentation')
parser.add_argument('--T_horizon', type=int, default=2048, help='length of long trajectory')
parser.add_argument('--Max_train_steps', type=int, default=5e7, help='Max training steps')
parser.add_argument('--save_interval', type=int, default=1e5, help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=5e3, help='Model evaluating interval, in steps.')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lambd', type=float, default=0.95, help='GAE Factor')
parser.add_argument('--clip_rate', type=float, default=0.2, help='PPO Clip rate')
parser.add_argument('--K_epochs', type=int, default=10, help='PPO update times')
parser.add_argument('--net_width', type=int, default=64, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--l2_reg', type=float, default=0, help='L2 regulization coefficient for Critic')
parser.add_argument('--batch_size', type=int, default=64, help='length of sliced trajectory')
parser.add_argument('--buffer_size', type=float, default=int(3e3), help='size of replay buffer')
parser.add_argument('--entropy_coef', type=float, default=0, help='Entropy coefficient of Actor')
parser.add_argument('--entropy_coef_decay', type=float, default=0.99, help='Decay rate of entropy_coef')
parser.add_argument('--adv_normalization', type=str2bool, default=False, help='Advantage normalization')
parser.add_argument('--num_classes', type=int, default=6, help='Number of Actor output classes')

opt = parser.parse_args()
if opt.set_visible_cuda_devices:
    print("GPU DEVICES (CUDA) SELECTED: % s" % opt.set_visible_cuda_devices)
os.environ["CUDA_VISIBLE_DEVICES"] = opt.set_visible_cuda_devices

opt.train_device = torch.device(opt.train_device) # from str to torch.device
print(opt)

def main():
    if opt.EnvIdex == 0 and opt.res_net:
        print("RL_UR16eManipulation_Full_Nocam (EnvIdex 0) does not support resnet")
        return
    elif opt.EnvIdex == 1 and not opt.res_net:
        print("RL_UR16eManipulation_Full (EnvIdex 1) only supports resnet")
        return
    env_name_lst = ['RL_UR16eManipulation_Full_Nocam', 'RL_UR16eManipulation_Full']
    env_name = env_name_lst[opt.EnvIdex]
    ne = opt.num_envs
    head_less = not opt.render
    DEVICE = opt.sim_device
    DEVICE_ID = opt.sim_device_id
    if opt.set_visible_cuda_devices[0] == '0':
        graphics_id = 0
    else:
        graphics_id = 3
    env = isaacgymenvs.make(
        seed=0,
        task=env_name,
        num_envs=ne,
        sim_device=DEVICE, # cpu cuda:0
        rl_device=DEVICE, # cpu cuda:0
        multi_gpu=False,
        graphics_device_id=graphics_id,
        headless=head_less
    )
    action_space = Discrete(6)
    opt.state_dim = 10
    opt.action_dim = 6

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Env:',env_name,'  state_dim:',opt.state_dim,'  action_dim:',opt.action_dim,'   Random Seed:',opt.seed)
    print('\n')

    # Use tensorboard to record training curves
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}'.format(env_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        run = wandb.init(project="ppo", name=env_name, config=opt)
        env.run = run

    if not os.path.exists('model'): os.mkdir('model')
    agent = PPO_discrete(**vars(opt))
    if opt.LoadModel: agent.load(opt.ModelIdex)
    buf_envs = [RolloutBuffer() for _ in range(ne)]
    
    # env.reset()
    actions = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to(DEVICE)
    state, scrap, reward, done, true_indicies = step_primitives(actions, env) #env.reset() by kickstarting w random action
    state, _, r, dw, true_indicies = returns_to_device(state, scrap, reward, done, true_indicies, DEVICE)
    if opt.res_net:
        real_ys, real_dxs = get_real_ys_dxs(state)
        s = rearrange_state_timestep(state)
    else:
        s = state[:, -10:]
        state = state[:, :-10]
        real_ys, real_dxs = get_real_ys_dxs(state)
        state = rearrange_state(state)
        s = normalize_state(s).to(DEVICE)

    if False:
        while True:
            ep_r = evaluate_policy(env, agent, turns=1)
            print(f'Env:{env_name}, Episode Reward:{ep_r}')
    else:
        traj_length, total_steps = 0, 0
        print_running_reward, print_running_ep = 0, 0
        action_envs_log = torch.zeros(ne, 1).to(DEVICE)
        while total_steps < opt.Max_train_steps:
            # s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
            # env_seed += 1
            # done = False

            # '''Interact & trian'''
            # while not done:
                '''Interact with Env'''
                a, logprob_a = agent.select_action(s.to(opt.train_device), deterministic=False) # use stochastic when training
                assert a.shape == (true_indicies.shape[0], 1), "Expected shape {}, got {}".format((true_indicies.shape[0], 1), a.shape)
                a = torch.concat((a.to(DEVICE), torch.ones(true_indicies.shape[0], 1).to(DEVICE)), dim = 1).to(DEVICE)
                action_envs_log[true_indicies] = a[:, 0].unsqueeze(1).clone().detach()
                for i, true_i in enumerate(true_indicies):
                    buf_envs[true_i].states.append(s[i].unsqueeze(0).clone().detach().cpu())
                    buf_envs[true_i].actions.append(a[i].clone().detach().cpu())
                    buf_envs[true_i].logprobs.append(logprob_a[i].clone().detach().cpu())
                    if true_i == 0:
                        print("action of env 0 updated", a[i])
                env_action = convert_actions(a, real_ys, real_dxs, sim_device=DEVICE)
                # s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                assert env_action.shape == (true_indicies.shape[0], 4)
                assert actions.shape == (ne, 4)
                create_env_action_via_true_indicies(true_indicies, env_action, actions, ne, DEVICE)
                assert actions.shape == (ne, 4)
                next_state, _, r, dw, true_indicies = step_primitives(actions, env)
                for i in range(true_indicies.shape[0]):
                    print_running_reward += r[i]
                if opt.res_net:
                    real_ys, real_dxs = get_real_ys_dxs(next_state)
                    next_state = rearrange_state_timestep(next_state)
                    s_next = next_state
                else:
                    s_next = next_state[:, -10:]
                    next_state = next_state[:, :-10]
                    real_ys, real_dxs = get_real_ys_dxs(next_state)
                    next_state = rearrange_state(next_state)
                    s_next = normalize_state(s_next)  

                
                # done = (dw or tr)
                done = dw

                '''Store the current transition'''
                for i, true_i in enumerate(true_indicies):
                    if len(buf_envs[true_i].rewards) != len(buf_envs[true_i].states):
                        buf_envs[true_i].rewards.append(r[i].clone().detach().unsqueeze(0).cpu())
                        buf_envs[true_i].is_terminals.append(done[i].clone().detach().unsqueeze(0).cpu())
                        buf_envs[true_i].new_states.append(s_next[i].unsqueeze(0).clone().detach().cpu())
                        buf_envs[true_i].traj_lengths.append(traj_length)
                    if(len(buf_envs[true_i].is_terminals) != 0 and buf_envs[true_i].is_terminals[-1] == True):
                        # agent.put_data(s, a, r, s_next, logprob_a, done, dw, idx = traj_length)
                        # agent.put_data(buf_envs[true_i].states, buf_envs[true_i].actions, buf_envs[true_i].rewards, buf_envs[true_i].new_states, 
                        #                buf_envs[true_i].logprobs, buf_envs[true_i].is_terminals, buf_envs[true_i].is_terminals)
                        for i in range(len(buf_envs[true_i].states)):
                            agent.replay_buffer.add(buf_envs[true_i].states[i], buf_envs[true_i].actions[i], buf_envs[true_i].rewards[i], buf_envs[true_i].new_states[i], buf_envs[true_i].logprobs[i], buf_envs[true_i].is_terminals[i])
                        traj_length += 1
                        total_steps += 1
                        print_running_ep += 1
                s = s_next
                state = next_state

                

                '''Update if its time'''
                if traj_length % opt.T_horizon == 0 and traj_length != 0:
                    agent.train()
                    traj_length = 0

                '''Record & log'''
                if total_steps % opt.eval_interval == 0 and total_steps != 0:
                    if opt.write:
                        print("total_steps: ", total_steps, " print_running_reward: ", print_running_reward, " avg_score: ", print_running_reward/print_running_ep)
                        wandb.log({'score': print_running_reward, 'steps': total_steps, 'avg_score': print_running_reward/print_running_ep})
                        wandb.log({'action_envs_log': wandb.Histogram(action_envs_log.cpu())})
                        print_running_reward = 0
                        print_running_ep = 0

                '''Save model'''
                if total_steps % opt.save_interval==0:
                    agent.save(total_steps)

        env.close()

if __name__ == '__main__':
    main()
