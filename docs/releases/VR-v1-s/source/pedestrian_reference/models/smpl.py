"""SMPL body model for the optional human nodes.

Only what linear blend skinning needs: the template mesh, the first 10 shape
directions, the joint regressor, the skinning weights and the kinematic tree.
Pose-dependent blend shapes are not applied, matching DriveStudio/OmniRe's SMPL
Gaussians, which skin a fixed canonical template.

The official SMPL pickle stores ``shapedirs`` as a chumpy array, so it is
converted once to ``.npz`` by ``scripts/convert_smpl_model.py``; reading the npz
needs only numpy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

NUM_JOINTS = 24
NUM_VERTICES = 6890
NUM_BETAS = 10


class SMPL(nn.Module):
    """Shape and skin the SMPL template.

    Buffers are non-persistent: they come from the model file, not from
    checkpoints, so enabling human nodes adds nothing to the state dict.

    Args:
        model_path: ``.npz`` written by ``scripts/convert_smpl_model.py``.
    """

    def __init__(self, model_path: str) -> None:
        super().__init__()
        data = np.load(model_path)

        def buffer(name: str, value: np.ndarray, dtype: torch.dtype) -> None:
            self.register_buffer(name, torch.as_tensor(value, dtype=dtype), persistent=False)

        buffer("v_template", data["v_template"], torch.float32)  # (6890, 3)
        buffer("shapedirs", data["shapedirs"][:, :, :NUM_BETAS], torch.float32)  # (6890, 3, 10)
        buffer("j_regressor", data["j_regressor"], torch.float32)  # (24, 6890)
        buffer("weights", data["weights"], torch.float32)  # (6890, 24)
        buffer("faces", data["faces"], torch.long)  # (13776, 3)
        # Walked once per call while chaining transforms; a Python list avoids a
        # device sync per joint.
        self.parents: list[int] = [int(p) for p in data["parents"]]

    def shape(self, betas: Tensor) -> tuple[Tensor, Tensor]:
        """Rest-pose (T-pose) vertices and joints.

        Args:
            betas: (P, 10) shape coefficients.

        Returns:
            vertices (P, 6890, 3), joints (P, 24, 3).
        """
        vertices = self.v_template + torch.einsum("vcb,pb->pvc", self.shapedirs, betas)
        joints = torch.einsum("jv,pvc->pjc", self.j_regressor, vertices)
        return vertices, joints

    def joint_transforms(self, pose: Tensor, joints: Tensor) -> Tensor:
        """Per-joint rigid transforms that carry the rest pose into ``pose``.

        Args:
            pose:   (..., P, 24, 3, 3) joint rotations; joint 0 is the global
                    orientation, joints 1-23 are relative to their parent.
            joints: (P, 24, 3) rest-pose joints from :meth:`shape`.

        Returns:
            (..., P, 24, 4, 4) transforms A_j.  A rest-pose point rigidly attached
            to joint j is posed as A_j @ [x, 1]; the global orientation rotates
            about the root joint, and no translation is applied.
        """
        joints = joints.expand(*pose.shape[:-3], NUM_JOINTS, 3)
        offsets = joints.clone()
        offsets[..., 1:, :] = joints[..., 1:, :] - joints[..., self.parents[1:], :]

        local = torch.zeros(*pose.shape[:-2], 4, 4, dtype=pose.dtype, device=pose.device)
        local[..., :3, :3] = pose
        local[..., :3, 3] = offsets
        local[..., 3, 3] = 1.0

        chain = [local[..., 0, :, :]]
        for j in range(1, NUM_JOINTS):
            chain.append(chain[self.parents[j]] @ local[..., j, :, :])
        world = torch.stack(chain, dim=-3)

        # World transforms place joint j at its posed position; subtract the
        # rotated rest position so they act on rest-pose coordinates instead.
        out = world.clone()
        out[..., :3, 3] = world[..., :3, 3] - (world[..., :3, :3] @ joints.unsqueeze(-1)).squeeze(-1)
        return out

    def blend(self, transforms: Tensor) -> tuple[Tensor, Tensor]:
        """Linear blend of joint transforms per vertex.

        Args:
            transforms: (..., P, 24, 4, 4) from :meth:`joint_transforms`.

        Returns:
            rots (..., P, 6890, 3, 3) and trans (..., P, 6890, 3); a rest vertex v
            is posed as rots @ v + trans.
        """
        blended = torch.einsum("vj,...jrc->...vrc", self.weights, transforms)
        return blended[..., :3, :3], blended[..., :3, 3]

    def pose_to_world(self, betas: Tensor, pose: Tensor, centres: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Skin the rest mesh into ``pose`` and place it in the world.

        Placement follows DriveStudio: the posed mesh is shifted so its
        bounding-box centre sits on the box centre.

        Args:
            betas:   (P, 10).
            pose:    (K, P, 24, 3, 3).
            centres: (K, P, 3) box centres.

        Returns:
            rest  (P, 6890, 3) rest-pose vertices, the canonical coordinates.
            rots  (K, P, 6890, 3, 3) and trans (K, P, 6890, 3): a canonical point
                  x attached to vertex v is placed at rots[v] @ x + trans[v].
            All float32.
        """
        # World coordinates run to tens of metres, which bf16 under the training
        # autocast would quantise to ~0.2 m, so skinning always runs in fp32.
        with torch.autocast(device_type=betas.device.type, enabled=False):
            rest, joints = self.shape(betas.float())
            rots, trans = self.blend(self.joint_transforms(pose.float(), joints))
            posed = (rots @ rest.unsqueeze(-1)).squeeze(-1) + trans
            shift = (posed.amin(dim=-2) + posed.amax(dim=-2)) / 2
            return rest, rots, trans + (centres.float() - shift).unsqueeze(-2)

    def posed_vertices(self, betas: Tensor, pose: Tensor) -> Tensor:
        """Convenience: skin the shaped rest mesh into ``pose``.

        Args:
            betas: (P, 10).
            pose:  (..., P, 24, 3, 3).

        Returns:
            (..., P, 6890, 3), without translation.
        """
        vertices, joints = self.shape(betas)
        rots, trans = self.blend(self.joint_transforms(pose, joints))
        return (rots @ vertices.unsqueeze(-1)).squeeze(-1) + trans
