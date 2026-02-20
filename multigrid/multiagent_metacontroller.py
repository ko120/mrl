"""Multi-agent metacontroller using RLlib PPO with shared policy (CTDE).

Centralized Training, Decentralized Execution:
- All agents share a single policy (centralized training via parameter sharing)
- Each agent acts independently using only its own observation (decentralized execution)

Uses Ray RLlib's PPO algorithm with multi-agent configuration.
"""

import numpy as np
import os
import torch
import wandb

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

from networks.multigrid_ppo_rl_module import MultiGridPPORLModule
from networks.multigrid_ppo_learner import MultiGridPPOLearner

from multigrid_rllib_env import MultiGridRLlibEnv
from utils import plot_single_frame, make_video
class MultiAgent():
    """Meta-agent that uses RLlib PPO to train multiple agents with a shared
    policy (CTDE - Centralized Training, Decentralized Execution).

    All agents share a single policy network. During training, experience from
    all agents is pooled to train that shared policy. During execution, each
    agent uses only its own observation to select actions.
    """

    def __init__(self, config, env, device, training=True, with_expert=None, debug=False):
        self.config = config
        self.debug = debug
        self.device = device
        self.n_agents = env.n_agents
        self.model_others = getattr(config, 'model_others', False)
        self.use_local_obs = getattr(config, 'algorithm', 'MAPPO') == 'IPPO'
        self.total_steps = 0
        self.total_episodes = 0

        # Store the raw env for visualization (RLlib manages its own env copies)
        self._raw_env = env

        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=4)

        # Register the multigrid env with RLlib
        env_name = getattr(config, 'domain', 'MultiGrid-Cluttered-Fixed-15x15')
        use_local_obs = self.use_local_obs
        register_env(
            "multigrid",
            lambda cfg: MultiGridRLlibEnv({"env_name": env_name}),
        )

        # Configure RLlib PPO - all agents share one policy (CTDE)
        num_env_runners = getattr(config, 'num_env_runners', 0)
        self.algo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment("multigrid")
            .env_runners(
                num_env_runners=num_env_runners,
            )
            .learners(
                learner_class=MultiGridPPOLearner,
                num_gpus_per_learner=1,
            )
            .training(
                lr=getattr(config, 'lr', 0.0003),
                gamma=getattr(config, 'gamma', 0.99),
                lambda_=getattr(config, 'lambda_', 0.95),
                clip_param=getattr(config, 'clip_param', 0.2),
                vf_loss_coeff=getattr(config, 'vf_loss_coeff', 0.5),
                entropy_coeff=getattr(config, 'entropy_coeff', 0.01),
                train_batch_size_per_learner=getattr(config, 'train_batch_size', 4000),
                minibatch_size=getattr(config, 'minibatch_size', 128),
                num_epochs=getattr(config, 'num_epochs', 4),
                grad_clip=getattr(config, 'grad_clip', 0.5),)
            .multi_agent(
                # CTDE: ALL agents map to the SAME shared policy
                policies={"shared_policy"},
                policy_mapping_fn=lambda agent_id, episode, **kwargs: "shared_policy",
            )
            .rl_module(
                # Use custom RLModule that follows multigrid_network.py architecture.
                # RLlib auto-wraps this in a MultiRLModule for multi-agent.
                rl_module_spec=RLModuleSpec(
                    module_class=MultiGridPPORLModule,
                    model_config={
                        "kernel_size": config.kernel_size,
                        "fc_direction": config.fc_direction,
                        "n_agents": env.n_agents,
                        "algorithm": getattr(config, "algorithm", "IPPO"),
                        "share_backbone": getattr(config, "share_backbone", False),
                        "backward": getattr(config, "backward", False),
                    },
                ),
            )
        )
        # Build the algorithm
        if training:
            self.algo = self.algo_config.build_algo()

    def get_agent_state(self, state, agent_id):
        """Extract a single agent's raw observation from the multi-agent state.

        Args:
            state: dict from multigrid env with 'image' (list) and 'direction' (list)
            agent_id: integer agent index
        Returns:
            dict with 'image' (raw for visualization) and 'direction'
        """
        return {
            'image': np.array(state['image'][agent_id], dtype=np.uint8),
            'direction': np.array(state['direction'], dtype=np.uint8),
        }

    def _encode_agent_obs(self, state, agent_id):
        """Encode agent observation matching MultiGridRLlibEnv layout.

        Layout: [local image (3)] + [global images (n*3)]
              + [local dir (4)]   + [global dir (n*4)]
        """
        from multigrid_rllib_env import NUM_DIRECTIONS
        all_images = state['image']
        directions = state['direction']
        image = np.array(all_images[agent_id], dtype=np.uint8)
        h, w = image.shape[0], image.shape[1]

        # Local image (3 ch)
        local_img = image.astype(np.float32)

        # Global images: all agents stacked by channel (n_agents*3 ch)
        global_imgs = np.concatenate(
            [np.array(all_images[a], dtype=np.float32) for a in range(self.n_agents)],
            axis=-1,
        )

        # Local direction (4 ch)
        local_dir = np.zeros(NUM_DIRECTIONS, dtype=np.float32)
        local_dir[int(directions[agent_id])] = 1.0

        # Global direction (n_agents*4 ch)
        global_dir = np.zeros(self.n_agents * NUM_DIRECTIONS, dtype=np.float32)
        for a, d in enumerate(directions):
            global_dir[a * NUM_DIRECTIONS + int(d)] = 1.0

        dir_onehot = np.concatenate([local_dir, global_dir])
        dir_channels = np.broadcast_to(
            dir_onehot[np.newaxis, np.newaxis, :], (h, w, len(dir_onehot))
        ).copy()
        return np.concatenate([local_img, global_imgs, dir_channels], axis=-1)

    def get_actions(self, state):
        """Get actions for all agents using the shared RLlib policy.

        Each agent acts based on its own observation (decentralized execution).
        """
        module = self.algo.get_module("shared_policy")

        # Build a single batched obs tensor for all agents
        obs_list = [self._encode_agent_obs(state, i) for i in range(self.n_agents)]
        obs_batch = torch.FloatTensor(np.stack(obs_list, axis=0))

        with torch.no_grad():
            result = module.forward_inference({Columns.OBS: obs_batch})

        logits = result[Columns.ACTION_DIST_INPUTS]  # (n_agents, n_actions)
        actions = torch.distributions.Categorical(logits=logits).sample()
        return [int(a) for a in actions]

    def run_one_episode(self, env, episode, log=True, train=True,
                        save_model=True, visualize=False):
        """Run a single episode manually for evaluation/visualization.

        This method steps through the environment using the trained RLlib policy.
        It does NOT train - training happens in algo.train().
        """
        state = env.reset()
        done = False
        t = 0
        rewards = []

        if visualize:
            viz_data = self.init_visualization_data(env, state)

        while not done:
            self.total_steps += 1
            t += 1

            # Each agent selects action from shared policy using own observation
            actions = self.get_actions(state)

            # Step the environment
            next_state, reward_list, done, info = env.step(actions)

            # Record per-agent rewards for this timestep
            rewards.append(reward_list)

            if visualize:
                viz_data = self.add_visualization_data(
                    viz_data, env, state, actions, next_state)

            # Advance state
            state = next_state

        rewards = np.array(rewards)  # shape: (T, n_agents)

        # Logging and checkpointing
        if log:
            self.log_one_episode(episode, t, rewards)
        self.print_terminal_output(episode, np.sum(rewards))
        if save_model:
            self.save_model_checkpoints(episode)

        if visualize:
            viz_data['rewards'] = rewards
            return viz_data

    def train(self, env):
        """Main training loop using RLlib's algo.train().

        Periodically runs manual evaluation episodes for visualization.
        """
        n_episodes = getattr(self.config, 'n_episodes', 100000)
        visualize_every = getattr(self.config, 'visualize_every', 10000)
        save_every = getattr(self.config, 'save_model_episode', 5000)
        log_every = getattr(self.config, 'log_episode', 100)
        print_every = getattr(self.config, 'print_every', 50)

        # Estimate how many algo.train() iterations we need.
        # Each train() call collects train_batch_size steps.
        train_batch_size = getattr(self.config, 'train_batch_size', 4000)
        env_max_steps = getattr(self._raw_env, 'max_steps', 100)
        est_episodes_per_iter = max(1, train_batch_size // env_max_steps)

        iteration = 0
        while self.total_episodes < n_episodes:
            iteration += 1

            # --- RLlib training step (centralized training) ---
            result = self.algo.train()

            # Extract metrics from RLlib results
            episodes_this_iter = result.get(
                "num_episodes_lifetime", self.total_episodes + est_episodes_per_iter
            ) - self.total_episodes
            if episodes_this_iter <= 0:
                episodes_this_iter = est_episodes_per_iter
            self.total_episodes += episodes_this_iter
            self.total_steps = result.get("num_env_steps_sampled_lifetime", self.total_steps)

            # Extract reward info
            env_runners = result.get("env_runners", {})
            mean_reward = env_runners.get("episode_return_mean", 0.0)
            episode_len = env_runners.get("episode_len_mean", 0.0)
            agent_returns = env_runners.get("agent_episode_returns_mean", {})

            # Log to wandb
            if self.total_episodes % log_every < episodes_this_iter:
                log_data = {
                    "episode/x_axis": self.total_episodes,
                    "episode/collective_reward_mean": mean_reward,
                    "episode/episode_length_mean": episode_len,
                    "step/x_axis": self.total_steps,
                    "step/collective_reward_mean": mean_reward,
                }
                # Log per-agent rewards
                for agent_id, agent_reward in agent_returns.items():
                    log_data[f"episode/{agent_id}_reward_mean"] = agent_reward
                wandb.log(log_data)

            # Print progress
            if iteration % max(1, print_every // est_episodes_per_iter) == 0:
                agent_str = " | ".join(
                    f"{aid}: {r:.3f}" for aid, r in sorted(agent_returns.items())
                )
                print(
                    f"Total steps: {self.total_steps} \t "
                    f"Episodes: ~{self.total_episodes} \t "
                    f"Mean reward: {mean_reward:.3f} \t "
                    f"Mean ep len: {episode_len:.1f} \t "
                    f"Per-agent: [{agent_str}]"
                )

            # Save checkpoint
            if self.total_episodes % save_every < episodes_this_iter:
                save_path = getattr(self.config, 'save_path', None)
                checkpoint_dir = self.algo.save(save_path) if save_path else self.algo.save()
                print(f"Checkpoint saved at episode ~{self.total_episodes}: {checkpoint_dir}")

            # Visualization episode
            # if (self.total_episodes % visualize_every < episodes_this_iter
            #         and self.total_episodes > 0):
            #     print(f"Running visualization at episode ~{self.total_episodes}...")
            #     viz_data = self.run_one_episode(
            #         self._raw_env, self.total_episodes, log=False,
            #         train=False, save_model=False, visualize=True)
            #     self.visualize(
            #         self._raw_env,
            #         self.config.mode + '_training_step' + str(self.total_episodes),
            #         viz_data=viz_data)

        env.close()

        # Save and upload final model to wandb
        save_path = getattr(self.config, 'save_path', None)
        final_checkpoint = self.algo.save(save_path) if save_path else self.algo.save()
        print(f"Final model saved: {final_checkpoint}")
        artifact = wandb.Artifact(name="final_model", type="model")
        artifact.add_dir(final_checkpoint)
        wandb.log_artifact(artifact)
        print("Final model uploaded to wandb.")

        self.algo.stop()

    def log_one_episode(self, episode, t, rewards):
        """Log a single episode's metrics to wandb."""
        collective_reward = np.sum(rewards)
        per_agent_rewards = np.sum(rewards, axis=0)

        log_data = {
            "episode/x_axis": episode,
            "episode/collective_reward": collective_reward,
            "episode/episode_length": t,
        }
        for i in range(self.n_agents):
            log_data[f"episode/agent_{i}_reward"] = per_agent_rewards[i]

        wandb.log(log_data)

    def save_model_checkpoints(self, episode):
        if episode % self.config.save_model_episode == 0 and episode > 0:
            save_path = getattr(self.config, 'save_path', None)
            checkpoint_dir = self.algo.save(save_path) if save_path else self.algo.save()
            print(f"Checkpoint saved at episode {episode}: {checkpoint_dir}")

    def print_terminal_output(self, episode, total_reward):
        if episode % self.config.print_every == 0:
            print('Total steps: {} \t Episode: {} \t Total reward: {}'.format(
                self.total_steps, episode, total_reward))

    def init_visualization_data(self, env, state):
        viz_data = {
            'agents_partial_images': [],
            'actions': [],
            'full_images': [],
            'predicted_actions': None,
        }
        viz_data['full_images'].append(env.render('rgb_array'))

        if self.model_others:
            predicted_actions = []
            predicted_actions.append(self.get_action_predictions(state))
            viz_data['predicted_actions'] = predicted_actions

        return viz_data

    def add_visualization_data(self, viz_data, env, state, actions, next_state):
        viz_data['actions'].append(actions)
        viz_data['agents_partial_images'].append(
            [env.get_obs_render(
                self.get_agent_state(state, i)['image'])
             for i in range(self.n_agents)])
        viz_data['full_images'].append(env.render('rgb_array'))
        if self.model_others:
            viz_data['predicted_actions'].append(
                self.get_action_predictions(next_state))
        return viz_data

    def visualize(self, env, mode, video_dir='videos', viz_data=None):
        if not viz_data:
            viz_data = self.run_one_episode(
                env, episode=0, log=False, train=False, save_model=False,
                visualize=True)

        video_path = os.path.join(
            video_dir, self.config.experiment_name, self.config.model_name)

        if not os.path.exists(video_path):
            os.makedirs(video_path)

        # Get names of actions
        action_dict = {}
        for act in env.Actions:
            action_dict[act.value] = act.name

        traj_len = len(viz_data['rewards'])
        for t in range(traj_len):
            self.visualize_one_frame(
                t, viz_data, action_dict, video_path, self.config.model_name)
            print('Frame {}/{}'.format(t, traj_len))

        make_video(video_path, mode + '_trajectory_video')

    def visualize_one_frame(self, t, viz_data, action_dict, video_path,
                            model_name):
        plot_single_frame(
            t,
            viz_data['full_images'][t],
            viz_data['agents_partial_images'][t],
            viz_data['actions'][t],
            viz_data['rewards'],
            action_dict,
            video_path,
            self.config.model_name,
            predicted_actions=viz_data['predicted_actions'],
            all_actions=viz_data['actions'])

    def load_models(self, model_path=None):
        """Load a saved RLlib checkpoint."""
        if model_path is not None:
            self.algo = self.algo_config.build_algo()
            self.algo.restore(model_path)
        else:
            print("No model path provided, using current algo state.")
