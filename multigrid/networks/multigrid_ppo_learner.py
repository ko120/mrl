"""Custom PPO Learner with separate actor and critic optimizers.

Actor optimizer:  image_layers + pi_direction_layers + pi_trunk + pi_head
Critic optimizer: vf_direction_layers + vf_trunk + vf_head

The shared image_layers sits in the actor optimizer but receives gradients from
both the policy loss AND the value loss (since PPO uses a combined loss), so
both objectives influence the shared CNN.
"""

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override
import pdb 

class BackwardEpisode(ConnectorV2):
    """Reverses the batch along the time axis after GAE.

    Placing this after GAE means advantages and value targets are already
    computed from the correct forward trajectory, then the whole batch is
    flipped so that the dense end-of-episode rewards appear first (position 0),
    receiving stronger gradient signal due to less discounting.
    """

    def __call__(self, *, rl_module, batch, episodes, explore=None, shared_data=None, **kwargs):
        for module_batch in batch.values():
            for key, val in module_batch.items():
                if isinstance(val, torch.Tensor):
                    module_batch[key] = val.flip(0)
        return batch


class MultiGridPPOLearner(PPOTorchLearner):
    """PPO Learner with separate actor/critic optimizers."""

    @override(PPOTorchLearner)
    def build(self):
        super().build()  # appends GAE last via PPOLearner.build()
        if (
            self._learner_connector is not None
            and self.config.add_default_connectors_to_learner_pipeline
        ):
            module_ids = list(self._module.keys())
            backward = (
                self._module[module_ids[0]].model_config.get("backward", False)
                if module_ids else False
            )
            if backward:
                self._learner_connector.append(BackwardEpisode())

    @override(PPOTorchLearner)
    def configure_optimizers_for_module(self, module_id, **kwargs):
        module = self._module[module_id]
        lr = self.config.lr

        if module.model_config.get("share_backbone", False):
            # Single optimizer for all params: backbone gradients flow from
            # both policy and value losses and are updated together.
            all_params = list(module.parameters())
            self.register_optimizer(
                module_id=module_id,
                optimizer_name="shared",
                optimizer=torch.optim.Adam(all_params, lr=lr, eps=1e-5),
                params=all_params,
            )
        else:
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
