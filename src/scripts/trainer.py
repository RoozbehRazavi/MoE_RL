import numpy as np
from ..algorithm.q_learning import QLearning
import torch.nn as nn
from itertools import count
import torch.optim as optim
import torch
from ..models.utils import ReplayBuffer
import gym
import time
import torch.utils.tensorboard as tb
from ..utils.plotter import plot_wei, plot_tsne, plot_wei_lin, plot_durations
from ..utils.utils import save_frames_as_video, find_latest_checkpoint, recreate_directory, create_directory


class Trainer(object):
    def __init__(self, args, device, seed, writer):
        super(Trainer).__init__()

        self.writer = writer
        self.env = gym.make("MsPacman-v4", render_mode='rgb_array')
        action_size = self.env.action_space.n
        state, info = self.env.reset()
        obs_size = state.shape
        print(obs_size)
        if args.model.state_encoder.type == 'cnn':
            q_network_input_size = None
        elif args.model.state_encoder.type == 'cnn_moe':
            q_network_input_size = args.model.state_encoder.state_embedding.layers.channel * args.model.MOE.experts_output_dim
        else:
            raise Exception()
        if args.training.algorithm == 'DQN':
            self.agent = QLearning(args, state_size=q_network_input_size, action_size=action_size).to(device)

        self.optimizer = optim.AdamW(self.agent.learnable_params(), lr=args.training.learning_rate, amsgrad=True)

        self.memory = ReplayBuffer()

        self.writer = tb.SummaryWriter(args.path.logs_path)

        self.start_step = 0
        self.total_frames = 0
        self.steps_done = 0
        self.action_size = action_size
        self.eps_threshold = None
        self.args = args
        self.device = device
        self.end_time = 0
        self.seed = seed
        self.start_time = time.time()

        self.load_model()

    def train(self):
        eval_returns = []
        for i_episode in range(self.start_step, self.args.training.num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            for t in count():
                self.total_frames += 1
                # TODO action, _, _, _ = self.agent.select_action(state)
                action = self.agent.select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())

                terminated = [terminated]
                truncated = [truncated]

                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # if any(terminated):
                #     true_indices = [index for index, value in enumerate(terminated) if value]
                #     # TODO
                #     next_state[true_indices] = torch.zeros((len(true_indices), *next_state.shape[1:])).to('cuda') * torch.nan
                
                done = torch.tensor([done], dtype=torch.int).unsqueeze(-1)
                self.memory.push(state, action, next_state, reward, done)
                state = next_state
                if all(done):
                    break

                training_loss = self.training_step(i_episode)
                self.agent.after_update(i_episode)
                eval_return = self.evaluation_step(i_episode, training_loss)
                self.save_model(iteration=i_episode)
                eval_returns.append(eval_return)
        return np.array(eval_returns)

    def training_step(self, iteration):
        if len(self.memory) < self.args.training.batch_size:
            return 0
        if iteration % self.args.training.update_every == 0:
            transitions = self.memory.sample(self.args.training.batch_size)
            loss = self.agent.calculate_loss(batch=transitions)
            loss = self.optimize(loss)
            return loss
        else:
            return 0

    def optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.agent.learnable_params(), 100)
        self.optimizer.step()
        return loss.item()

    def evaluation_step(self, iteration, training_loss):
        if iteration % self.args.evaluation.eval_every == 0:
            done = False
            state, _ = self.env.reset()
            return_ = 0
            weights = []
            states = []
            extras = []
            out_gates = []
            renders = []
            counter = 0
            while not done:
                if self.args.evaluation.render:
                    rendered = self.env.render()
                    renders.append(rendered)
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                states.append(state)
                # action, gating_weight, extra, out_gat = self.agent.select_action(state, random_=False)
                action = self.agent.select_action(state, random_=False)
                # extras.append(extra.permute(-1, 0))
                # weights.append(gating_weight)
                # out_gates.append(out_gat)
                state, reward, done, _, _ = self.env.step(action.item())
                return_ += (reward * (self.args.training.gamma ** counter))
                counter += 1

            # weights = torch.cat(weights, dim=0).cpu().transpose(-1, -2).numpy()
            # extras = torch.stack(extras, dim=0).squeeze(1).cpu().permute(1, 0).numpy()
            # out_gates = torch.cat(out_gates, dim=0).unsqueeze(0).cpu().numpy()
            states = torch.cat(states).cpu().numpy()

            self.log(training_loss, return_, extras, weights, out_gates, states, renders, iteration)

            return return_,

    def log(self, training_loss, eval_return, experts_activation_counter, experts_activation_weights,
            outer_gates_counter, states, renders, i_episode):
        if self.args.evaluation.log_weights_norm:
            for name, param in self.agent.state_encoding.named_parameters():
                if param.requires_grad:
                    param_norm = torch.norm(param).item()
                    self.writer.add_scalar(f'Param_state_encoding_{name}', param_norm, i_episode)
                    if param.grad is not None:
                        grad_param_norm = torch.norm(param.grad).item()
                        self.writer.add_scalar(f'Grad_state_encoding_{name}', grad_param_norm, i_episode)

            for name, param in self.agent.q_net.named_parameters():
                if param.requires_grad:
                    param_norm = torch.norm(param).item()
                    self.writer.add_scalar(f'Param_q_{name}', param_norm, i_episode)
                    if param.grad is not None:
                        grad_param_norm = torch.norm(param.grad).item()
                        self.writer.add_scalar(f'Grad_q_{name}', grad_param_norm, i_episode)

        self.end_time = time.time()
        total_time = self.end_time - self.start_time
        self.start_time = time.time()

        self.writer.add_scalar('Return', eval_return, self.total_frames)

        self.writer.add_scalar('Elapsed Time', total_time, self.total_frames)

        print(f'Iteration: {i_episode} Frames: {self.total_frames} Return: {eval_return}'
              f' Training Loss: {training_loss} Elapsed Time: {total_time}')

        # plot_wei(experts_activation_weights, i_episode, self.args.path.figures_path, name='attention_weights_plot')
        # plot_wei(experts_activation_counter, i_episode, str_=self.args.path.figures_path, name='inside_counter')
        # plot_wei(outer_gates_counter, i_episode, str_=self.args.path.figures_path, name='outside_counter')
        #
        # #states = torch.cat(states).cpu().numpy()
        # plot_tsne(experts_activation_weights, states, i_episode, str_=self.args.path.figures_path, name='tsne')
        save_frames_as_video(renders, filename=f'{self.args.path.videos_path}/pacman_trajectory{i_episode}.gif')

    def save_model(self, iteration):
        if iteration % self.args.evaluation.save_every == 0:
            save_path = f'./{self.args.path.save_path}/model_checkpoint{iteration}.pth'
            torch.save({
                'model_state_dict': self.agent.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'i_episode': iteration,  # store the loss value
                'EPS': self.agent.eps_threshold,
                'frame': self.total_frames
            }, save_path)

    def load_model(self):
        if self.args.load:
            iteration = find_latest_checkpoint(self.args.save_path)
            if iteration is not None:
                checkpoint = torch.load(f'./{self.args.save_path}/model_checkpoint{iteration}.pth')
                self.agent.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_step = checkpoint['i_episode']
                self.agent.eps_threshold = checkpoint['EPS']
                self.total_frames = checkpoint['frame']

            create_directory(self.args.path.save_path)
            create_directory(self.args.path.videos_path)
            create_directory(self.args.path.logs_path)
            create_directory(self.args.path.figures_path)
        else:
            recreate_directory(self.args.path.save_path)
            recreate_directory(self.args.path.videos_path)
            recreate_directory(self.args.path.logs_path)
            recreate_directory(self.args.path.figures_path)





