# Learning to Parallel Decoding

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
