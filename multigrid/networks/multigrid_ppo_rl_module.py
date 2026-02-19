"""Custom RLModule for PPO with separate policy and value networks.

Policy network (actor) — always uses LOCAL observation:
    image_layers:        CNN on local image (3 ch).
    pi_direction_layers: local direction one-hot (4 ch).
    pi_trunk + pi_head:  produces action logits.

Value network (critic) — obs depends on algorithm:
    MAPPO: vf_image_layers (n_agents*3 ch) + vf_direction_layers (n_agents*4 ch)
    IPPO:  vf_image_layers (3 ch)          + vf_direction_layers (4 ch)

Observation layout (from MultiGridRLlibEnv):
    obs[:, :, :,  0 : 3             ] = local image          (actor)
    obs[:, :, :,  3 : 3+n*3         ] = all agents' images   (MAPPO critic)
    obs[:, 0, 0,  3+n*3 : 3+n*3+4  ] = local direction       (actor + IPPO critic)
    obs[:, 0, 0,  3+n*3+4 : end     ] = global direction      (MAPPO critic)
"""

from typing import Any, Dict, Optional

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()

NUM_DIRECTIONS = 4
_IMG_CH = 3


class MultiGridPPORLModule(TorchRLModule, ValueFunctionAPI):
    """Separate actor/critic networks; critic obs depends on MAPPO vs IPPO."""

    @override(TorchRLModule)
    def setup(self):
        kernel_size = self.model_config.get("kernel_size", 3)
        fc_direction = self.model_config.get("fc_direction", 8)
        n_agents = self.model_config.get("n_agents", 3)
        algorithm = self.model_config.get("algorithm", "MAPPO")

        self._use_global_vf = (algorithm == "MAPPO")

        # ------------------------------------------------------------------
        # Channel slice indices (depend on n_agents)
        # ------------------------------------------------------------------
        self._global_img_end  = _IMG_CH + n_agents * _IMG_CH       # 3 + n*3
        self._local_dir_start = self._global_img_end                # 3 + n*3
        self._local_dir_end   = self._local_dir_start + NUM_DIRECTIONS
        self._global_dir_end  = self._local_dir_end + n_agents * NUM_DIRECTIONS

        vf_img_ch = n_agents * _IMG_CH if self._use_global_vf else _IMG_CH
        vf_dir_ch = n_agents * NUM_DIRECTIONS if self._use_global_vf else NUM_DIRECTIONS

        # ------------------------------------------------------------------
        # Policy network (actor) — local image + local direction
        # ------------------------------------------------------------------
        self.image_layers = nn.Sequential(
            nn.Conv2d(_IMG_CH, 32, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        self.pi_direction_layers = nn.Sequential(
            nn.Linear(NUM_DIRECTIONS, fc_direction),
            nn.ReLU(),
        )
        self.pi_trunk = nn.Sequential(
            nn.Linear(64 + fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
        )
        self.pi_head = nn.Linear(64, self.action_space.n)

        # ------------------------------------------------------------------
        # Value network (critic) — separate image extractor
        # MAPPO: all agents' images (n*3 ch) + global dir (n*4 ch)
        # IPPO:  local image (3 ch)           + local dir  (4 ch)
        # ------------------------------------------------------------------
        self.vf_image_layers = nn.Sequential(
            nn.Conv2d(vf_img_ch, 32, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, (kernel_size, kernel_size)),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
        )
        self.vf_direction_layers = nn.Sequential(
            nn.Linear(vf_dir_ch, fc_direction),
            nn.ReLU(),
        )
        self.vf_trunk = nn.Sequential(
            nn.Linear(64 + fc_direction, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
        )
        self.vf_head = nn.Linear(64, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pi_features(self, batch):
        """Actor: local image + local direction."""
        obs = batch[Columns.OBS]
        image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
        img = self.image_layers(image)
        local_dir = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        dir_feat = self.pi_direction_layers(local_dir)
        return self.pi_trunk(torch.cat([img, dir_feat], dim=-1))

    def _vf_features(self, batch):
        """Critic: global (MAPPO) or local (IPPO) image + direction."""
        obs = batch[Columns.OBS]
        if self._use_global_vf:
            image = obs[:, :, :, _IMG_CH:self._global_img_end].permute(0, 3, 1, 2).float()
            dir_input = obs[:, 0, 0, self._local_dir_end:self._global_dir_end].float()
        else:
            image = obs[:, :, :, :_IMG_CH].permute(0, 3, 1, 2).float()
            dir_input = obs[:, 0, 0, self._local_dir_start:self._local_dir_end].float()
        img = self.vf_image_layers(image)
        dir_feat = self.vf_direction_layers(dir_input)
        return self.vf_trunk(torch.cat([img, dir_feat], dim=-1))

    # ------------------------------------------------------------------
    # RLlib API
    # ------------------------------------------------------------------

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        """Inference: policy network only."""
        return {Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch))}

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        """Training: run both networks, store vf features as embeddings."""
        return {
            Columns.ACTION_DIST_INPUTS: self.pi_head(self._pi_features(batch)),
            Columns.EMBEDDINGS: self._vf_features(batch),
        }

    @override(ValueFunctionAPI)
    def compute_values(
        self,
        batch: Dict[str, Any],
        embeddings: Optional[Any] = None,
    ) -> TensorType:
        if embeddings is None:
            embeddings = self._vf_features(batch)
        return self.vf_head(embeddings).squeeze(-1)
