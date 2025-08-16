# MoBiPlant

This repository provides tools for evaluating and benchmarking large language models (LLMs) on the **MoBiPlant benchmark🌱**: PhD-level plant molecular biology questions from researchers all over the globe. 

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/manolofer10/mobiplant.git
   cd mobiplant
   ```

2. **Set environment:**
   *(Create and activate a virtual environment + install dependencies.)*
   ```sh
   conda env create -f environment.yml
   conda activate mobiplant
   python -m pip install -r requirements.txt
   ```

3. **API Keys:**
   - Place your model API keys in a `tokens.json` file in the root directory. Example:
     ```json
     {
       "gemini": "YOUR_GEMINI_KEY",
       "chatgpt": "YOUR_OPENAI_KEY",
       "claude": "YOUR_ANTHROPIC_KEY",
       "llama": "YOUR_LLAMA_KEY",
       "o1-mini": "YOUR_OPENAI_KEY",
       "v3": "YOUR_DEEPSEEK_KEY",
       "r1": "YOUR_DEEPSEEK_KEY"
     }
     ```
     *Only these models are supported for evaluation. If you want to include others, you may have to change how models are initialized in predict_answers.py*.
     
     (Note that you can set a subset of these if you want to test less models).

## Usage

The main entry point is `predict_answers.py`. This script runs inference with selected models on MoBiPlant.

### Example: Quick Multiple-Choice Question Answering
*Once you have set up a tokens.json file;*

```sh
python predict_answers.py \
  --data_path manufernandezbur/MoBiPlant \
  --models model_name \ 
  --setting mcq-answering \ #for mcq answering
  --results_dataset_path where_to_save_results \ #will save as json
  --evaluation_style CoT \ # for thinking before answering
```

- `--data_path`: Path to your input CSV file.
- `--models`: Space-separated list of model names (must be from: `gemini`, `chatgpt`, `claude`, `llama`, `o1-mini`, `v3`, `r1`).
- `--setting`: `mcq-answering` or `long-form-answering`.
- `--results_dataset_path`: Directory to save results.
- `--evaluation_style`: `direct` (letter only) or `CoT` (chain-of-thought with answer tags).

### Example: Quick Long-Form Answering

```sh
python predict_answers.py \
  --data_path manufernandezbur/MoBiPlant \
  --models model_name \
  --setting long-form-answering \
  --results_dataset_path where_to_save_results \
```

## Evaluation and Plotting
### Example: Quick metrics from previous results
```sh
python eval.py \
  --results_dataset inference_results_path \ #i.e. where_to_save_results
  --output_folder evaluation_results \ # will save metrics to this directory
  --evaluation_style automatic-metrics \ 
```

- You can use `eval_utils.py` for advanced metrics calculation and figure generation (i.e. --evaluation_style automatic-plotting).

## Citation

If you use this codebase, please cite the corresponding publication *wip*.

---
