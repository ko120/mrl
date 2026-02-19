"""Custom PPO Learner with separate actor and critic optimizers.

Actor optimizer:  image_layers + pi_direction_layers + pi_trunk + pi_head
Critic optimizer: vf_direction_layers + vf_trunk + vf_head

The shared image_layers sits in the actor optimizer but receives gradients from
both the policy loss AND the value loss (since PPO uses a combined loss), so
both objectives influence the shared CNN.
"""

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override


class MultiGridPPOLearner(PPOTorchLearner):
    """PPO Learner with separate actor/critic optimizers."""

    @override(PPOTorchLearner)
    def configure_optimizers_for_module(self, module_id, **kwargs):
        module = self._module[module_id]
        lr = self.config.lr

        actor_params = (
            list(module.image_layers.parameters())
            + list(module.pi_direction_layers.parameters())
            + list(module.pi_trunk.parameters())
            + list(module.pi_head.parameters())
        )
        critic_params = (
            list(module.vf_image_layers.parameters())
            + list(module.vf_direction_layers.parameters())
            + list(module.vf_trunk.parameters())
            + list(module.vf_head.parameters())
        )

        self.register_optimizer(
            module_id=module_id,
            optimizer_name="actor",
            optimizer=torch.optim.Adam(actor_params, lr=lr, eps=1e-5),
            params=actor_params,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name="critic",
            optimizer=torch.optim.Adam(critic_params, lr=lr, eps=1e-5),
            params=critic_params,
        )
