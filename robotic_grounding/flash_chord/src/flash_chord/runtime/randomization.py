# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-world physics randomization: contact friction and object mass.

Each world draws its scales once, when the environment is built, and keeps them across resets, so a
batch of worlds covers the configured ranges. The scaled values are written to the Newton model and
pushed to MuJoCo-Warp's per-world model fields through ``SolverMuJoCo.notify_model_changed``.

With both ranges at ``(1, 1)`` (the default) nothing is touched, so existing checkpoints and the
evaluator behave exactly as before. A range with equal ends sets every world to that value, which is
how a robustness evaluation applies one fixed perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass

import newton
import numpy as np


@dataclass
class DomainRandomizationConfig:
    """Uniform per-world scale ranges; ``(1, 1)`` leaves that quantity unchanged."""

    friction_scale: tuple[float, float] = (1.0, 1.0)  # scales every world-owned shape's friction
    object_mass_scale: tuple[float, float] = (1.0, 1.0)  # scales object body mass and inertia
    seed: int = 0

    def __post_init__(self) -> None:
        for name in ("friction_scale", "object_mass_scale"):
            low, high = (float(value) for value in getattr(self, name))
            if not (np.isfinite(low) and np.isfinite(high)) or low <= 0.0 or high < low:
                raise ValueError(f"{name} must satisfy 0 < low <= high, got ({low}, {high})")
            setattr(self, name, (low, high))

    @property
    def enabled(self) -> bool:
        return self.friction_scale != (1.0, 1.0) or self.object_mass_scale != (1.0, 1.0)


def sample_world_scales(low_high: tuple[float, float], world_count: int, rng: np.random.Generator) -> np.ndarray:
    """One scale per world, uniform in ``[low, high]``."""
    low, high = low_high
    return rng.uniform(low, high, size=world_count) if high > low else np.full(world_count, low)


def apply_domain_randomization(scene, solver, config: DomainRandomizationConfig) -> dict[str, np.ndarray]:
    """Scale each world's friction and object mass in place; return the per-world scales used."""
    if not config.enabled:
        return {}
    model = scene.model
    world_count = scene.world_count
    rng = np.random.default_rng(config.seed)
    friction = sample_world_scales(config.friction_scale, world_count, rng)
    mass = sample_world_scales(config.object_mass_scale, world_count, rng)
    flags = 0

    if config.friction_scale != (1.0, 1.0):
        # Shapes shared by all worlds (world -1, e.g. the ground plane) keep their friction; MuJoCo
        # combines a contact pair's friction by maximum, so the hand-object contacts carry the scale.
        shape_world = model.shape_world.numpy()
        mu = model.shape_material_mu.numpy()
        owned = shape_world >= 0
        mu[owned] *= friction[shape_world[owned]]
        model.shape_material_mu.assign(mu)
        flags |= newton.solvers.SolverNotifyFlags.SHAPE_PROPERTIES

    if config.object_mass_scale != (1.0, 1.0):
        bodies_per_world = model.body_count // world_count
        local_ids = np.array([body.body_id for binding in scene.objects for body in binding.bodies], dtype=np.int64)
        ids = (local_ids[None, :] + np.arange(world_count)[:, None] * bodies_per_world).ravel()
        scale = np.repeat(mass, len(local_ids))
        body_mass = model.body_mass.numpy()
        body_inv_mass = model.body_inv_mass.numpy()
        body_inertia = model.body_inertia.numpy()
        body_inv_inertia = model.body_inv_inertia.numpy()
        body_mass[ids] *= scale
        body_inv_mass[ids] /= scale
        body_inertia[ids] *= scale[:, None, None]
        body_inv_inertia[ids] /= scale[:, None, None]
        model.body_mass.assign(body_mass)
        model.body_inv_mass.assign(body_inv_mass)
        model.body_inertia.assign(body_inertia)
        model.body_inv_inertia.assign(body_inv_inertia)
        flags |= newton.solvers.SolverNotifyFlags.BODY_INERTIAL_PROPERTIES

    solver.notify_model_changed(flags)
    return {"friction_scale": friction, "object_mass_scale": mass}
