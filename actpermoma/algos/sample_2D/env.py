"""
Environment for rrt_2D
@author: huiming zhou
"""

class Env:
    def __init__(self, x_range=50, y_range=30, obs_coords=None):
        self.x_range = (0, x_range)
        self.y_range = (0, y_range)
        if obs_coords is not None:
            self.obs_boundary = []
            self.obs_circle = [[x, y, 1] for x, y in zip(obs_coords[:,0], obs_coords[:,1])]
            self.obs_rectangle = []
        else:
            self.obs_boundary = self.obs_boundary()
            self.obs_circle = self.obs_circle()
            self.obs_rectangle = self.obs_rectangle()

    @staticmethod
    def obs_boundary():
        obs_boundary = [
            [0, 0, 1, 30],
            [0, 30, 50, 1],
            [1, 0, 50, 1],
            [50, 1, 1, 30]
        ]
        return obs_boundary

    @staticmethod
    def obs_rectangle():
        obs_rectangle = [
            [14, 12, 8, 2],
            [18, 22, 8, 3],
            [26, 7, 2, 12],
            [32, 14, 10, 2]
        ]
        return obs_rectangle

    @staticmethod
    def obs_circle():
        obs_cir = [
            [7, 12, 3],
            [46, 20, 2],
            [15, 5, 2],
            [37, 7, 3],
            [37, 23, 3]
        ]

        return obs_cir