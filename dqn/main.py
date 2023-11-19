from utils import evaluate_policy, str2bool
import isaacgym
import isaacgymenvs
from datetime import datetime
from DQN import DQN_agent
import gymnasium as gym
import os, shutil
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
parser.add_argument('--dvc', type=str, default='cuda', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='CP-v1, LLd-v2')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--res_net', type=str2bool, default=False, help='Use resnet or not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=250*1000, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(1e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(3e3), help='steps for random policy to explore')
parser.add_argument('--update_every', type=int, default=50, help='training frequency')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=200, help='Hidden net width')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=512, help='length of sliced trajectory')
parser.add_argument('--exp_noise', type=float, default=0.2, help='explore noise')
parser.add_argument('--noise_decay', type=float, default=0.99, help='decay rate of explore noise')
parser.add_argument('--DDQN', type=str2bool, default=True, help='True:DDQN; False:DQN')
parser.add_argument('--eval', type=str2bool, default=False, help='True:deterministic; False:non-deterministic')

opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
print(opt)


def main():
    # EnvName = ['CartPole-v1','LunarLander-v2']
    # BriefEnvName = ['CPV1', 'LLdV2']
    # env = gym.make(EnvName[opt.EnvIdex], render_mode = "human" if opt.render else None)
    # eval_env = gym.make(EnvName[opt.EnvIdex])
    # opt.state_dim = env.observation_space.shape[0]
    # opt.action_dim = env.action_space.n
    # opt.max_e_steps = env._max_episode_steps

    env_name = "RL_UR16eManipulation_Full_Nocam"
    ne = 1
    head_less = not opt.render
    DEVICE = "cuda:0"
    DEVICE_ID = 0
    env = isaacgymenvs.make(
        seed=0,
        task=env_name,
        num_envs=ne,
        sim_device=DEVICE, # cpu cuda:0
        rl_device=DEVICE, # cpu cuda:0
        multi_gpu=False,
        graphics_device_id=DEVICE_ID,
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    print('Algorithm:',algo_name,'  Env:', env_name,'  state_dim:',opt.state_dim,
          '  action_dim:',opt.action_dim,'  Random Seed:',opt.seed, '\n')
    if opt.write:
        from torch.utils.tensorboard import SummaryWriter
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(algo_name, env_name) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)
        wandb.init(project="dqn", name=env_name, config=opt)

    #Build model and replay buffer
    if not os.path.exists('model'): os.mkdir('model')
    agent = DQN_agent(**vars(opt))
    if opt.Loadmodel: agent.load(algo_name, "L", opt.ModelIdex)
    
    # env.reset()
    actions = torch.tensor(ne * [[0.11, 0., 0.28, 0.22]]).to(DEVICE)
    state, scrap, reward, done, true_indicies = step_primitives(actions, env) #env.reset() by kickstarting w random action
    state, _, r, dw, true_indicies = returns_to_device(state, scrap, reward, done, true_indicies, DEVICE)
    if opt.res_net:
        real_ys, real_dxs = get_real_ys_dxs(state)
        state = rearrange_state_timestep(state)
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
        while total_steps < opt.Max_train_steps:
            # s, info = env.reset(seed=env_seed) # Do not use opt.seed directly, or it can overfit to opt.seed
            # env_seed += 1
            
            done = False

            '''Interact & trian'''
            while not done:
                #e-greedy exploration
                if total_steps < opt.random_steps and not opt.eval: 
                    # a = action_space.sample()
                    a = torch.randint(0, 6, (1,)).to(DEVICE)
                    a = torch.cat((a, torch.tensor([1.0]).to(DEVICE)), dim=0).unsqueeze(0)
                else: 
                    if opt.eval:
                        a = agent.select_action(s, deterministic=True)
                    else:
                        a = agent.select_action(s, deterministic=False)
                    a = torch.tensor([a, 1.0]).to(DEVICE).unsqueeze(0)
                print("action: ", a)
                env_action = convert_actions(a, real_ys, real_dxs, sim_device=DEVICE)
                # s_next, r, dw, tr, info = env.step(a) # dw: dead&win; tr: truncated
                next_state, _, r, dw, true_indicies = step_primitives(env_action, env)
                print_running_reward += r
                if opt.res_net:
                    real_ys, real_dxs = get_real_ys_dxs(next_state)
                    state = rearrange_state_timestep(next_state)
                else:
                    s_next = next_state[:, -10:]
                    next_state = next_state[:, :-10]
                    real_ys, real_dxs = get_real_ys_dxs(next_state)
                    next_state = rearrange_state(next_state)
                s_next = normalize_state(s_next)  

                
                # done = (dw or tr)
                done = dw

                if opt.EnvIdex == 1:
                    if r <= -100: r = -10  # good for LunarLander
                agent.replay_buffer.add(s, a.squeeze(0)[0], r, s_next, dw)
                s = s_next
                state = next_state

                '''update if its time'''
                # train 50 times every 50 steps rather than 1 training per step. Better!
                if total_steps >= opt.random_steps and total_steps % opt.update_every == 0:
                    for j in range(opt.update_every): agent.train()

                '''record & log'''
                '''TOOD: add wandb log to record the training'''
                if total_steps % opt.eval_interval == 0:
                    if opt.write:
                        wandb.log({'score': print_running_reward, 'steps': total_steps, 'avg_score': print_running_reward/print_running_ep})
                        print_running_reward = 0
                        print_running_ep = 0


                # if total_steps % opt.eval_interval == 0:
                #     agent.exp_noise *= opt.noise_decay
                #     score = evaluate_policy(eval_env, agent, turns = 3)
                #     if opt.write:
                #         writer.add_scalar('ep_r', score, global_step=total_steps)
                #         writer.add_scalar('noise', agent.exp_noise, global_step=total_steps)
                #     print('EnvName:',env_name[opt.EnvIdex],'seed:',opt.seed,'steps: {}k'.format(int(total_steps/1000)),'score:', int(score))
                #     wandb.log({'score': score, 'noise': agent.exp_noise, 'steps': total_steps})
                total_steps += 1
                print_running_ep += 1

                '''save model'''
                if total_steps % opt.save_interval == 0:
                    name = algo_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    agent.save(name ,env_name[opt.EnvIdex],total_steps)
    env.close()
    # eval_env.close()

if __name__ == '__main__':
    main()








