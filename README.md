# LLMs-in-PMB

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.08.31.672925-b31b1b.svg)](https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-manufernandezbur%2FMoBiPlant-yellow)](https://huggingface.co/datasets/manufernandezbur/MoBiPlant)



This repository provides tools for evaluating and benchmarking large language models (LLMs) on plant molecular biology (PMB) question answering tasks. It supports both multiple-choice (MCQ) and long-form answer generation, as well as human and automatic evaluation, data preprocessing, and figure creation for analysis.


## Installation

1. **Clone the repository:** (anonymous submission!)
   ```sh
   git clone -----
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
  --data_path anonymousSub10/mobiplant \
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
  --data_path anonymousSub10/mobiplant \
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

If you use this repository, please cite:

````bibtex
@article {Fernandez Burda2025.08.31.672925,
    author = {Fernandez Burda, Manuel and Ferrero, Lucia and Gaggion, Nicol{\'a}s and Fonouni-Farde, Camille and The MoBiPlant Consortium and Crespi, Mart{\'\i}n and Ariel, Federico and Ferrante, Enzo},
    title = {What Large Language Models Know About Plant Molecular Biology},
    elocation-id = {2025.08.31.672925},
    year = {2025},
    doi = {10.1101/2025.08.31.672925},
    publisher = {Cold Spring Harbor Laboratory},
    abstract = {Large language models (LLMs) are rapidly permeating scientific research, yet their capabilities in plant molecular biology remain largely uncharacterized. Here, we present MoBiPlant, the first comprehensive benchmark for evaluating LLMs in this domain, developed by a consortium of 112 plant scientists across 19 countries. MoBiPlant comprises 565 expert-curated multiple-choice questions and 1,075 synthetically generated questions, spanning core topics from gene regulation to plant-environment interactions. We benchmarked seven leading chat-based LLMs using both automated scoring and human evaluation of open-ended answers. Models performed well on multiple-choice tasks (exceeding 75% accuracy), although most of them exhibited a consistent bias towards option A. In contrast, expert reviews exposed persistent limitations, including factual misalignment, hallucinations, and low self-awareness. Critically, we found that model performance strongly correlated with the citation frequency of source literature, suggesting that LLMs do not simply encode plant biology knowledge uniformly, but are instead shaped by the visibility and frequency of information in their training corpora. This understanding is key to guiding both the development of next-generation models and the informed use of current tools in the everyday work of plant researchers. MoBiPlant is publicly available online in this link.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925},
    eprint = {https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925.full.pdf},
    journal = {bioRxiv}
}
````

