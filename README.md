# Learning to Parallel Decoding

[![arXiv](https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv)](https://arxiv.org/abs/2509.25188) [![arXiv](https://img.shields.io/badge/Project%20Page-red)](https://ims-kdks.github.io/learning-to-parallel/)

<img width="1806" height="373" alt="introduction" src="https://github.com/user-attachments/assets/87b9994a-dc69-42e6-a5f8-6e8276b98883" />

https://github.com/user-attachments/assets/09c67a58-b425-463d-a998-c1a6049bc171

## 🔥News
- [2025-10-14] [Dream](https://github.com/DreamLM/Dream) integration coming soon!

## 💡Methods
### 1. Learning to Parallel Decoding
<img width="1520" height="547" alt="overview" src="https://github.com/user-attachments/assets/8e475aaa-b032-4f3a-8ee9-6eed02ae184b" />

Extremely Greedy Parallel strategy: compares the predicted tokens with the reference answer and only remasks the tokens that do not match in these comparisons.
Use a trained filter $f_\theta$ that simulate the Extremely Greedy Parallel strategy after each decoding step to select tokens and decide whether to remask them.

### 2. End-of-Text Prediction
<img width="35%" alt="eot" src="https://github.com/user-attachments/assets/ec85eba5-6bab-44e2-82cd-51c5431674d8" />

Upon detection of an $[EoT]$ token, we throw away all the tokens after the $[EoT]$ token in the next diffusion step. When the specified output length is very long (for example, 1024), this method can significantly reduce computation by dynamically reducing the input size during the diffusion process.

## 🏎️Performance
Experiments on **GSM8K**, **MATH**, **HumanEval**, and **MBPP** show that our approach significantly improves throughput (by **up to 22.58 times faster**) while maintaining model accuracy, demonstrating outstanding generalization and practicality. Each method was evaluated using two generation lengths (256 and 1024) across four datasets. Performance is measured using three metrics: TPS (tokens/sec), speedup, and accuracy score. The highest throughput and speedup values for each configuration are highlighted in bold.
<p align="center">
   <img width="70%" alt="performance" src="https://github.com/user-attachments/assets/529e3272-4714-4299-9a4e-ed0674d72b89" />
</p>

## How to run
1. Install dependencies
```
pip install -r requirements.txt
```
2. Run the program
   1. Test single questions
   ```
   python generate.py
   ```
   2. Run evaluations
   ```
   ./eval_llada.sh
   ```
## Generate data for training
```
./generate_training_data.sh
```

## Training Filter
You can directly use `training.ipynb` to train new filter models with your own datasets.

## Acknowledgments
We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada) and [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) for their excellent work and open-source contributions. 

## Citation
If you find our work useful, please consider citing our paper.
```
@misc{bao2025learningparallelacceleratingdiffusion,
      title={Learning to Parallel: Accelerating Diffusion Large Language Models via Learnable Parallel Decoding}, 
      author={Wenrui Bao and Zhiben Chen and Dan Xu and Yuzhang Shang},
      year={2025},
      eprint={2509.25188},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.25188}, 
}
```
