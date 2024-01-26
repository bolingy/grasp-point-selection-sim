from utils import evaluate_policy, str2bool
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import isaacgym
import isaacgymenvs
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
import os, shutil
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
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

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--data_augmentation_prob', type=float, default=0., help='data augmentation')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(10), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(10), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(40), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=10, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-7, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='length of sliced trajectory')
parser.add_argument('--buffer_size', type=float, default=int(1e3), help='size of replay buffer')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')
parser.add_argument('--eval', type=str2bool, default=False, help='True:deterministic; False:non-deterministic')
parser.add_argument('--num_classes', type=int, default=6, help='Number of QNet output classes')

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

    #Use DDQN or DQN
    if opt.DDQN: algo_name = 'DDQN'
    else: algo_name = 'DQN'

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:',algo_name,'  Env:', env_name,'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '\n')
    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo_name, env_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        run = wandb.init(project="dqn", name=env_name, config=opt)
        env.run = run

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.LoadModel: 
        agent.load(opt.ModelName, "L", opt.ModelIdx)
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


    # if opt.render:
    if False:
        while True:
            score = evaluate_policy(env, agent, 1)
            print('EnvName:', env_name, 'seed:', opt.seed, 'score:', score)
    else:
        print_running_reward = 0
        print_running_ep = 0
        total_steps = 0
        update_steps = 0
        eval_steps = 0 
        save_steps = 0
        action_envs_log = torch.zeros(ne, 1).to(DEVICE)
        while total_steps < opt.Max_train_steps:
            # s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            # env_seed += 1
            
            # done = False

            # '''Interact & trian'''
            # while not done:
                #e-greedy exploration
                if total_steps < opt.random_steps and not opt.eval: 
                    # a = action_space.sample()
                    a = torch.randint(0, 6, (true_indicies.shape[0], 1)).to(DEVICE)
                    # a = torch.ones(true_indicies.shape[0], 1).to(DEVICE) * 1.0
                    assert a.shape == (true_indicies.shape[0], 1)
                    a = torch.concat((a, torch.ones(true_indicies.shape[0], 1).to(DEVICE)), dim = 1).to(DEVICE)
                    assert a.shape == (true_indicies.shape[0], 2)
                else: 
                    if opt.res_net:
                        assert s.shape == (true_indicies.shape[0], 5, 180, 260), "Expected shape {}, got {}".format((true_indicies.shape[0], 3, 180, 260), s.shape)
                    else:
                        assert s.shape == (true_indicies.shape[0], 10)

                    if opt.eval:
                        a = agent.select_action(s.to(opt.train_device), deterministic=True)
                    else:
                        a = agent.select_action(s.to(opt.train_device), deterministic=False)
                    assert a.shape == (true_indicies.shape[0], 1), "Expected shape {}, got {}".format((true_indicies.shape[0], 1), a.shape)
                    a = torch.concat((a.to(DEVICE), torch.ones(true_indicies.shape[0], 1).to(DEVICE)), dim = 1).to(DEVICE)
                    assert a.shape == (true_indicies.shape[0], 2)
                action_envs_log[true_indicies] = a[:, 0].unsqueeze(1).clone().detach()
                for i, true_i in enumerate(true_indicies):
                    buf_envs[true_i].states.append(s[i].unsqueeze(0).clone().detach().cpu())
                    buf_envs[true_i].actions.append(a[i].clone().detach().cpu())
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

                for i, true_i in enumerate(true_indicies):
                    if len(buf_envs[true_i].rewards) != len(buf_envs[true_i].states):
                        buf_envs[true_i].rewards.append(r[i].clone().detach().unsqueeze(0).cpu())
                        buf_envs[true_i].is_terminals.append(done[i].clone().detach().unsqueeze(0).cpu())
                        buf_envs[true_i].new_states.append(s_next[i].unsqueeze(0).clone().detach().cpu())
                    if(len(buf_envs[true_i].is_terminals) != 0 and buf_envs[true_i].is_terminals[-1] == True):
                        for i in range(len(buf_envs[true_i].states)):
                            agent.replay_buffer.add(buf_envs[true_i].states[i], buf_envs[true_i].actions[i], buf_envs[true_i].rewards[i], buf_envs[true_i].new_states[i], buf_envs[true_i].is_terminals[i])
                        print_running_ep += 1
                        total_steps += 1
                        update_steps += 1
                        eval_steps += 1
                        save_steps += 1
                        buf_envs[true_i].clear()

                s = s_next
                state = next_state

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and update_steps >= opt.update_every and not opt.eval:
                    for j in range(opt.update_every): agent.train()
                    update_steps -= opt.update_every

                '''record & log'''
                if eval_steps >= opt.eval_interval:
                    if opt.write:
                        print("total_steps: ", total_steps, " print_running_reward: ", print_running_reward, " avg_score: ", print_running_reward/print_running_ep)
                        wandb.log({'score': print_running_reward, 'steps': total_steps, 'avg_score': print_running_reward/print_running_ep})
                        wandb.log({'action_envs_log': wandb.Histogram(action_envs_log.cpu())})
                        print_running_reward = 0
                        print_running_ep = 0
                    eval_steps -= opt.eval_interval 


                # if total_steps % opt.eval_interval == 0:
                #     agent.exp_noise *= opt.noise_decay
                #     score = evaluate_policy(eval_env, agent, turns = 3)
                #     if opt.write:
                #         writer.add_scalar('ep_r', score, global_step=total_steps)
                #         writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                #     print('EnvName:',env_name[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))
                #     wandb.log({'score': score, 'noise': agent.exp_noise, 'steps': total_steps})


                '''save model'''
                if save_steps >= opt.save_interval and not opt.eval:
                    name = algo_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    print('Saving model {} at {} steps'.format(name, total_steps)) 
                    agent.save(name ,env_name[opt.EnvIdex],total_steps)
                    # agent.replay_buffer.save_buffer(env_name[opt.EnvIdex], suffix=name)
                    save_steps -= opt.save_interval
    env.close()
    # eval_env.close()

if __name__ == '__main__':
    main()








