# Warp-Surgical

## Installation

Follow these steps to set up the environment:

1. Navigate to the IsaacLab folder:
```bash
cd isaacLab
```

2. Create a conda environment from the environment file:
```bash
conda env create -f environment.yml
```

3. Activate the conda environment:
```bash
conda activate isaaclab
```

4. Install PyTorch with CUDA 12.8 support:
```bash
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

5. Install Isaac Sim:
```bash
pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

6. Install IsaacLab dependencies:
```bash
./isaaclab.sh -i
```

7. Navigate to the Newton folder and install it:
```bash
cd ../newton
pip install -e .
```

8. Return to the IsaacLab folder:
```bash
cd ../isaacLab
```

9. Run the test environment:
```bash
./isaaclab.sh -p scripts/environments/zero_agent.py --task Isaac-Reach-STAR-v0 --num_envs 1
```

