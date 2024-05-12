import numpy as np
import itertools
from vgn.utils import look_at, spherical_to_cartesian

class ViewHalfSphere:
    #src: https://github.com/ethz-asl/active_grasp/blob/devel/src/active_grasp/controller.py
    def __init__(self, bbox, min_z_dist):
        self.center = bbox.center
        self.r = 0.5 * bbox.size[2] + min_z_dist

    def get_view(self, theta, phi):
        eye = self.center + spherical_to_cartesian(self.r, theta, phi)
        up = np.r_[1.0, 0.0, 0.0]
        return look_at(eye, self.center, up)

    def sample_view(self):
        raise NotImplementedError

class AABBox:
    # src: https://github.com/ethz-asl/active_grasp/blob/devel/src/active_grasp/bbox.py
    def __init__(self, bbox_min, bbox_max):
        self.min = np.asarray(bbox_min)
        self.max = np.asarray(bbox_max)
        self.center = 0.5 * (self.min + self.max)
        self.size = self.max - self.min

    @property
    def corners(self):
        return list(itertools.product(*np.vstack((self.min, self.max)).T))

    def is_inside(self, p):
        return np.all(p > self.min) and np.all(p < self.max)