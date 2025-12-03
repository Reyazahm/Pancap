## README for environment configuration

For details other than the environment configuration, please refer to README.md. 

### Install

1. Clone this repository and navigate to pancap folder
```bash
git clone https://github.com/Visual-AI/Pancap/
cd Pancap
```

2. Install Package
```Shell
conda create -n pancap python=3.10.14 -y
conda activate pancap
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn==2.5.6 --no-build-isolation
pip install numpy==1.26.4
pip install tensorboardX==2.6.2.2
```


