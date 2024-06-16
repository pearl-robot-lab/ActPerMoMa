"""Installation scripts for the 'actpermoma' python package."""

from setuptools import setup, find_packages

INSTALL_REQUIRES = [
    "protobuf",
    "omegaconf",
    "hydra-core",
]

# Installation operation
setup(
    name="actpermoma",
    author="Sophie Lueth & Snehal Jauhri",
    version="1.0.0",
    description="Active-Perceptive Motion Generation for Mobile Manipulation in NVIDIA Isaac Sim. Adapted from omniisaacgymenvs (https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs)",
    keywords=["robotics", "active perception"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.8, 3.10"],
    zip_safe=False,
)