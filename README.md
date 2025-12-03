
<div align="center">
<h1>Panoptic Captioning: An Equivalence Bridge for Image and Text</h1>
<h3 align="center">NeurIPS 2025</h3>


<a href="https://openreview.net/forum?id=Kq08RIeXxI" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2505.16334"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://visual-ai.github.io/pancap/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>


**[Visual AI Lab, HKU](https://visailab.github.io/people.html)**

[Kun-Yu Lin](https://kunyulin.github.io/), [Hongjun Wang](https://whj363636.github.io/), [Weining Ren](https://github.com/rwn17), [Kai Han](https://www.kaihan.org/)
</div>

## 📢 Updates
- [2025/12/03] 🔥Released training and inference code.
- [2025/09/18] 🎉The paper was accepted by NeurIPS'25.

## 🌈 Overview

### TL;DR
- A new image captioning task to seek the minimum text equivalent of images

![alt text](./assets/teasor.png)

- Panoptic captioning aims to generate a comprehensive textual description for an image, which encapsulates all entities, their respective locations and attributes, relationships among entities, as well as global image state. 
- Through an extensive evaluation, our work reveals that state-of-the-art Multi-modal Large Language Models (MLLMs) have limited performance in solving panoptic captioning.
- To address this task, we propose a effective data engine, contribute a new benchmark, and develop a novel decoupling method. 

### Contributions
- New task with new metric
- New data engine and new benchmark
- New model, that beats Qwen2.5-VL-72B, InternVL-2.5-78B, Gemini-2.0-Pro with only 13B parameters


## 💪 Environment
Please refer to [README_env.md](README_env.md) for environment configuration.

## 📚 Data Preparation
Our SA-Pancap benchmark is based on SA-1B, so you should download the required images from SA-1B. Our adopted images come from the first 64 subsets of SA-1B. 

- Download the first 63 subsets of the dataset, i.e., sa_000000 ~ sa_000063. Totally, this part roughly consists of 734243 images. 
- After downloading all of them, organize the data in a specific DATA_ROOT as follows:
```
├── sam
│   ├── sa_000000
│   ├── sa_000001
│   ├── ...
│   └── sa_000063
```
- The paths of training, validation and test images are summarized in [sapancap_train_data_list.json](playground/data/pancap/sapancap_train_data_list.json), [sapancap_val_data_list.json](playground/data/pancap/sapancap_val_data_list.json) and [sapancap_test_data_list.json](playground/data/pancap/sapancap_test_data_list.json). 


## 🚀 Training
We use the pretrained ASMv2 model as initialization, so users should first download the [stage2-trained checkpoint](https://huggingface.co/OpenGVLab/ASMv2) from ASMv2. Then, use the following script to run the training code. You should modify the paths of DATA_ROOT and SAVE_CKPT before running the code. 

```shell
bash scripts/pancapchain_train.sh
```

After finish training, you can use the following script to merge LoRA weights. 

```shell
bash scripts_pancap/eval/merge_lora.sh
```


## 🚝 Inference

You can use the following script to do inference on the *validation* set. 

```shell
bash scripts_pancap/eval/inference_pancapchain_val.sh
```

You can use the following script to do inference on the *test* set.

```shell
bash scripts_pancap/eval/inference_pancapchain_test.sh
```


## 🛸 Evaluation

The metric code will be released very soon. 


## 📌 Citation

For any question, please contact [Kun-Yu Lin](kunyulin14@outlook.com). If you find this work useful, please star this repo and cite our work as follows:

```bibtex
@inproceedings{lin2025pancap,
    title={Panoptic Captioning: An Equivalence Bridge for Image and Text},
    author={Lin, Kun-Yu and Wang, Hongjun and Ren, Weining and Han, Kai},
    journal={The Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025}
}
```

## 🌟 Acknowledgements
Thanks to these great repositories: [LLaVA](https://github.com/haotian-liu/LLaVA) and [All-Seeing](https://github.com/OpenGVLab/all-seeing), and many other inspiring works in the community.
