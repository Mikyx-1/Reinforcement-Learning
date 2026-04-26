from setuptools import find_packages, setup

setup(
    name="rl_zoo",
    version="0.1.0",
    description="Clean RL algorithm implementations for research and demonstration.",
    packages=find_packages(exclude=["tests", "notebooks", "scripts"]),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest", "matplotlib", "pandas"],
        "wandb": ["wandb>=0.15.0"],
    },
)
