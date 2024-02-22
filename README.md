# Hint-before-Solving Prompting (HSP)
This is the Source Code of Paper: [Hint-before-Solving Prompting: Guiding LLMs to
Effectively Utilize Encoded Knowledge](https://arxiv.org/).


## What is Hint-before-Solving Prompting (HSP)?


## Get started
We suggest using miniconda/conda to set up the environment based on python3.10. 
After create your env, you'll need install pytorch, transformers and vllm.
```
conda activate your_env
pip install torch torchvision torchaudio
pip install transformers
pip install vllm

```

 

## Repo Structure
- `data/`: Include six evaluation datasets. 
- `icl_robust/`: robust analysis examples.
- `prompt/`: SD,LtM,PS,CoT,*Hint prompts for the LM to generate the reasoning chain.
- `sft/`:
  - `deepspeed_config.json`: We use deepspeed to accelerate model training.
  - `sft_data_*.jsonl`: Sft datasets.
  - `sft_train.py`: The main train class.
  - `sft_train.sh`: Run this file to train model.  
- `evaluate.py`: Use this script evaluate the model output file to get accuracy.
- `inference.py`: Run the model to make predictions on a dataset.
- `produce_hintV2.py`: First generate hint and then generate solution.
- `produce_hintV2_gpt.py`: Use gpt4's hint to generate solution.
- `math/`: evaluation on Math datasets.
  - `data/`: Math datasets.
  - `prompt/`: The prompts used to in experiment.
  -  `evaluate.py`: evaluation script.
  -  `produce.py`: inference script.
  -  `utils.py`: Used for handle data.


## Usage

### Make predictions

Run `inference.py`:

Example:
```
python inference.py --model {model_path} --tp_degree 4 --dataset_name GSM8K --output_path gsm8k_output.json --hint
```

### Evaluate the model predictions
Run `evaluate.py`

The accuracy will be printed to stdout.

Example:
```
python evaluate.py --dataset_name GSM8K --file gsm8k_output.json
```

### SFT on HSPMATH dataset
- We constructed a HSPMATH1 dataset with 7.5k pieces of data (./sft/HSPMATH1.jsonl) based on the gsm8k train dataset with the assistance of gpt-4,
- Expansion of HSPMATH dataset based on metamathqa dataset to 75K pieces of data (./sft/HSPMATH. jsonl)
  - Expansion method: Extract data from the metamath dataset that has the same question and answer as the HSPMATH1 dataset, and apply the hint of HSPMATH1 to the corresponding problem data.
- We conducted SFT training on 8xA100 (80G) GPUs using Slurm scheduling system. The training script path is: ./sft/sft_train.sh
  - Please pay attention to modifying the save path for ckpt: epoch_output_dir_llemma_7b


## Citation
If you find this repository useful, please cite our paper:
```
@article{,
  title={Hint-before-Solving Prompting: Guiding LLMs to
Effectively Utilize Encoded Knowledge},
  author={},
  journal={},
  year={}
}
```