"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self, x_range=51, y_range=31, obs_coords=None):
        self.x_range = x_range  # size of background
        self.y_range = y_range
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map(obs_coords)

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self, obs_coords=None):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        if obs_coords is not None:
            for coord in obs_coords:
                obs.add(tuple(coord))
        else:
            # pre-defined map
            for i in range(x):
                obs.add((i, 0))
            for i in range(x):
                obs.add((i, y - 1))

            for i in range(y):
                obs.add((0, i))
            for i in range(y):
                obs.add((x - 1, i))

            for i in range(10, 21):
                obs.add((i, 15))
            for i in range(15):
                obs.add((20, i))

            for i in range(15, 30):
                obs.add((30, i))
            for i in range(16):
                obs.add((40, i))

        return obs