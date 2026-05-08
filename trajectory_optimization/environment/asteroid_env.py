import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import jax.numpy as jnp
from typing import NamedTuple


class Asteroid(NamedTuple):
    center:   jnp.array
    radius:   jnp.array
    velocity: jnp.array = 0

    def at_time(self, time):
        return self._replace(center=self.center + self.velocity * time)


class Environment(NamedTuple):
    asteroids:      Asteroid
    ship_radius:    jnp.array
    sensing_radius: jnp.array
    bounds:         jnp.array

    @classmethod
    def create(cls, num_asteroids, ship_radius=1.0, sensing_radius=5, bounds=(50, 40)):
        bounds = np.array(bounds)
        return cls(
            Asteroid(
                np.random.rand(num_asteroids, 2) * bounds,
                1 + 2 * np.random.rand(num_asteroids),
                np.random.randn(num_asteroids, 2),
            ),
            ship_radius,
            sensing_radius,
            bounds,
        )

    def at_time(self, time):
        return self._replace(asteroids=self.asteroids.at_time(time))

    def wrap_vector(self, vector):
        return (vector + self.bounds / 2) % self.bounds - self.bounds / 2

    def sense(self, position):
        """Return environment with unseen asteroid radii set to NaN."""
        deltas = self.wrap_vector(position - self.asteroids.center)
        return self._replace(
            asteroids=self.asteroids._replace(
                radius=jnp.where(
                    jnp.linalg.norm(deltas, axis=-1) - self.asteroids.radius
                    < self.sensing_radius,
                    self.asteroids.radius,
                    np.nan,
                )
            )
        )

    def plot(self, state=None, plan=None, history=None,
             sensor=False, scaled_thrust=None, ax=None):
        pose         = np.full(3, np.nan) if state is None else state[:3]
        plan         = np.full((0, 2), np.nan) if plan is None else plan[:, :2]
        history      = np.full((0, 2), np.nan) if history is None else history[:, :2]
        scaled_thrust = np.full((), np.nan) if scaled_thrust is None else scaled_thrust

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.set_xlim(0, self.bounds[0])
            ax.set_ylim(0, self.bounds[1])
            ax.set_aspect(1)
            asteroids = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Circle(np.zeros(2), r) for r in self.asteroids.radius] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="black",
                ))
            ship = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Polygon(self.ship_radius * np.array([[-.2, -.4], [1., 0], [-.2, .4]]))] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="orange",
                    zorder=10,
                ))
            circle = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Circle(np.zeros(2), self.sensing_radius)] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    facecolor=(0, 0, 0, 0),
                    edgecolor="black",
                    linestyle="--",
                    zorder=10,
                ))
            thruster = ax.add_collection(
                matplotlib.collections.PatchCollection(
                    [plt.Polygon(self.ship_radius * np.array([[-1., 0.], [0., -.25], [0., .25]]))] * 4,
                    offsets=np.zeros(2),
                    transOffset=matplotlib.transforms.AffineDeltaTransform(ax.transData),
                    color="red",
                    zorder=5,
                ))
            plan_line    = ax.plot(plan[:, 0],    plan[:, 1],    color="green")[0]
            history_line = ax.plot(history[:, 0], history[:, 1], color="blue")[0]
        else:
            fig = ax.figure
            asteroids, ship, circle, thruster = ax.collections
            plan_line, history_line = ax.lines

        screen_offsets = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        asteroids.set_offsets(
            (self.wrap_vector(self.asteroids.center)
             + self.bounds * screen_offsets[:, None, :]).reshape(-1, 2)
        )
        if sensor:
            asteroids.set_alpha(
                np.where(np.isnan(self.sense(pose[:2]).asteroids.radius), 0.1, 1.0)
            )
        ship.set_offsets(self.wrap_vector(pose[:2]) + self.bounds * screen_offsets)
        ship.set_transform(
            matplotlib.transforms.Affine2D().rotate(pose[2]) + ax.transData
        )
        circle.set_offsets(
            self.wrap_vector(pose[:2] if sensor else np.full(2, np.nan))
            + self.bounds * screen_offsets
        )
        thruster.set_offsets(self.wrap_vector(pose[:2]) + self.bounds * screen_offsets)
        thruster.set_transform(
            matplotlib.transforms.Affine2D()
            .scale(0.2 + 0.8 * scaled_thrust, 1)
            .rotate(pose[2])
            + ax.transData
        )

        def tile_line(line):
            if line.shape[0] == 0:
                return line
            irange, jrange = [
                range(int(x[0]), int(x[1] + 1))
                for x in zip(
                    np.min(line, 0) // self.bounds,
                    np.max(line, 0) // self.bounds,
                )
            ]
            return np.concatenate([
                np.pad(
                    line - np.array([i, j]) * self.bounds,
                    ((0, 1), (0, 0)),
                    constant_values=np.nan,
                )
                for i in irange
                for j in jrange
            ], 0)

        plan_line.set_data(*tile_line(plan).T)
        history_line.set_data(*tile_line(history).T)
        return fig, ax