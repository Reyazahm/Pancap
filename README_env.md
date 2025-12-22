## README for environment configuration

For details other than the environment configuration, please refer to README.md. 

### Install (for training and inference)

1. Clone this repository and navigate to pancap folder
```bash
git clone https://github.com/Visual-AI/Pancap/
cd Pancap
```

2. Create environment
```Shell
conda create -n pancap python=3.10.14 -y
conda activate pancap
```

3. Install packages
```Shell
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

4. Install additional packages for training cases
```Shell
pip install -e ".[train]"
pip install flash-attn==2.5.6 --no-build-isolation
pip install numpy==1.26.4
pip install tensorboardX==2.6.2.2
```


### Install for Evaluation Metric

1. Create environment
```Shell
conda create -n pancap_eval python=3.12.7 -y
conda activate pancap_eval
```

2. Install packages for Qwen inference
```Shell
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.45.1 accelerate==1.1.1 "bitsandbytes>=0.41.0" sentencepiece protobuf
pip install flash-attn==2.6.1 --no-build-isolation
```

3. Install additional packages for evaluation functions
```Shell
pip install scipy==1.13.1
pip install nltk==3.9.1
pip install sentence-transformers==3.3.1
pip install factualscenegraph==0.5.0
```

