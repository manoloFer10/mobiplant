# MoBiPlant

[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.08.31.672925-b31b1b.svg)](https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-manufernandezbur%2FMoBiPlant-yellow)](https://huggingface.co/datasets/manufernandezbur/MoBiPlant)

This repository provides tools for evaluating and benchmarking large language models (LLMs) on the **MoBiPlant benchmark🌱**: PhD-level plant molecular biology questions from researchers all over the globe. 


## System Requirements

### Operating systems (tested)
- **Windows 11** (development environment; `environment.yml` was exported from a Windows Miniconda install).
- **Linux (x86-64)** (used for the local `ChatNT` runs via `run_mobiplant_ChatNT.sh`).

Any OS with a working Conda + Python 3.11 toolchain should work, since all closed-source models are accessed over HTTP APIs.

### Software dependencies
- **Python 3.11.11** (pinned in `environment.yml`).
- The full pinned list is in `environment.yml`. Key packages:
  - Dataset loading: `datasets==3.6.0`, `huggingface-hub==0.32.4`
  - Model clients: `openai==1.76.0`, `anthropic==0.50.0`, `google-genai==1.12.1`
  - LangChain wrappers: `langchain-core==0.3.56`, `langchain-openai==0.3.14`, `langchain-anthropic==0.3.12`, `langchain-google-genai==2.1.3`
  - Analysis & plotting: `pandas==2.2.3`, `numpy==2.2.5`, `matplotlib==3.10.1`, `seaborn==0.13.2`, `plotly==6.0.1`
  - Utilities: `tenacity==9.1.2`, `tqdm==4.67.1`, `tiktoken==0.9.0`
- **Model backends** (for the local `ChatNT` model): `torch==2.11.0`, `transformers==4.38.2` — both pinned in `environment.yml` and installed by `conda env create`. Note that `benchmarking_utils/answering.py` imports `torch` at module load, so it is required for **every** run, not just `ChatNT`. See the note in *Installation* for the CUDA build.

### Hardware
- **API-based models** (`gemini`, `chatgpt`, `gpt-5.2`, `o1-mini`, `claude`, `sonnet-4.6`, `llama`, `v3`, `r1`): **no special hardware.** A standard desktop/laptop with an internet connection and valid API keys is sufficient as inference runs on the provider's servers.
- **`ChatNT`**: requires a **CUDA-capable NVIDIA GPU**. The model is loaded in `bfloat16` (`device='cuda'`) and was instantiated on a **24GB VRAM NVIDIA GPU**.


## Installation

1. **Clone the repository:** 
   ```sh
   git clone https://github.com/manoloFer10/mobiplant
   cd mobiplant
   ```

2. **Set environment:**
   *(Create and activate a virtual environment + install dependencies.)*
   ```sh
   conda env create -f environment.yml
   conda activate mobiplant
   ```
   This single command installs the entire dependency set (conda packages + the pinned `pip` section, including `torch` and `transformers`).

   > **Note on `torch` / `transformers`.** Both are pinned in `environment.yml` (`torch==2.11.0`, `transformers==4.38.2`) and installed by `conda env create`. `benchmarking_utils/answering.py` imports `torch` at module load, so it is required for **every** run, not just `ChatNT`. `transformers==4.38.2` is intentionally old: it is the version compatible with ChatNT's `trust_remote_code` model definition.
   >
   > For the local `ChatNT` model you need a **CUDA-capable** torch build. The plain `torch==2.11.0` above resolves the default PyPI wheel; the tested GPU build was `torch==2.11.0+cu130`, installable from the PyTorch index:
   > ```sh
   > pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
   > ```

   **Typical install time on a normal desktop:** ~5–15 min, dominated by the Conda dependency solve and package downloads (add a few minutes for `torch`).

3. **API Keys:**
   - Place your model API keys in a `tokens.json` file in the root directory. **Each key must match the exact model name you pass to `--models`** (the code looks up `keys[model_name]`). Example:
     ```json
     {
       "gemini": "YOUR_GEMINI_KEY",
       "chatgpt": "YOUR_OPENAI_KEY",
       "gpt-5.2": "YOUR_OPENAI_KEY",
       "o1-mini": "YOUR_OPENAI_KEY",
       "claude": "YOUR_ANTHROPIC_KEY",
       "sonnet-4.6": "YOUR_ANTHROPIC_KEY",
       "llama": "YOUR_LLAMA_KEY",
       "v3": "YOUR_DEEPSEEK_KEY",
       "r1": "YOUR_DEEPSEEK_KEY"
     }
     ```
     Supported model names are defined by `SUPPORTED_MODELS` in `benchmarking_utils/instantiation.py`: `llama`, `chatgpt`, `gpt-5.2`, `o1-mini`, `gemini`, `claude`, `sonnet-4.6`, `r1`, `v3`, `ChatNT`. `ChatNT` runs locally and needs **no** API key. To add other models, edit `instantiate_models` in `benchmarking_utils/instantiation.py`.

     (You can set only the subset of keys for the models you intend to run.)



## Usage

The main entry point is `predict_answers.py`. This script runs inference with selected models on MoBiPlant.

### Example: Multiple-Choice Question Answering
*Once you have set up a `tokens.json` file:*

```sh
python predict_answers.py \
  --data_path manufernandezbur/mobiplant \
  --models chatgpt \
  --setting mcq-answering \
  --evaluation_style CoT \
  --num_samples all \
  --results_dataset_path where_to_save_results
```

- `--data_path` *(required)*: Hugging Face dataset repo id (or a local dataset directory) loaded via `datasets.load_dataset(...)['train']` — **not** a raw CSV path. E.g. `manufernandezbur/mobiplant`.
- `--models` *(required)*: Space-separated list of model names from `SUPPORTED_MODELS` (`llama`, `chatgpt`, `gpt-5.2`, `o1-mini`, `gemini`, `claude`, `sonnet-4.6`, `r1`, `v3`, `ChatNT`).
- `--setting` *(required)*: `mcq-answering` or `long-form-answering`.
- `--results_dataset_path` *(required)*: Directory to save results (created if missing).
- `--num_samples` *(required)*: `all`, or an integer to evaluate a random subset of that size.
- `--evaluation_style`: `direct` (letter only) or `CoT` (chain-of-thought with `<ANSWER> X </ANSWER>` tags). Required for `mcq-answering`; ignored for `long-form-answering`.
- `--subset` *(optional)*: `expert` or `synthetic` (filters on the `is_expert` column when present).
- `--with_search` *(optional flag)*: enable web-search tool use for models that support it (`gpt-5.2`, `sonnet-4.6`).
- `--restart_from` / `--restart_path` *(optional)*: resume a partially completed run from a previous results JSON.

### Example: Long-Form Answering

```sh
python predict_answers.py \
  --data_path manufernandezbur/mobiplant \
  --models chatgpt \
  --setting long-form-answering \
  --num_samples all \
  --results_dataset_path where_to_save_results
```

## Demo

A minimal end-to-end run on a small random subset (5 questions) using an API model. This exercises the full pipeline — dataset download → inference → JSON output — without a large API bill.

```sh
python predict_answers.py \
  --data_path manufernandezbur/mobiplant \
  --models chatgpt \
  --setting mcq-answering \
  --evaluation_style CoT \
  --num_samples 5 \
  --results_dataset_path demo_results
```

**Expected output:** a JSON file at
`demo_results/inference/mcq-answering_CoT_chatgpt_results.json`,
with one record per question containing the model's chain-of-thought, the extracted answer as a 0-indexed integer (`A`→0, `B`→1, `C`→2), and the ground-truth label. A `tqdm` progress bar is printed to the console; intermediate checkpoints are written under `demo_results/temp/`.

**Expected run time:** ~1–2 min for 5 questions on a normal desktop. Runtime is dominated by API latency (requests are issued sequentially, `MAX_API_WORKERS = 1`); the local machine is idle while waiting on the provider.

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
    abstract = {Large language models (LLMs) are rapidly permeating scientific research, yet their capabilities in plant molecular biology remain largely uncharacterized. Here, we present MoBiPlant, the first comprehensive benchmark for evaluating LLMs in this domain, developed by a consortium of 112 plant scientists across 19 countries. MoBiPlant comprises 565 expert-curated multiple-choice questions and 1,075 synthetically generated questions, spanning core topics from gene regulation to plant-environment interactions. We benchmarked seven leading chat-based LLMs using both automated scoring and human evaluation of open-ended answers. Models performed well on multiple-choice tasks (exceeding 75% accuracy), although most of them exhibited a consistent bias towards option A. In contrast, expert reviews exposed persistent limitations, including factual misalignment, hallucinations, and low self-awareness. Critically, we found that model performance strongly correlated with the citation frequency of source literature, suggesting that LLMs do not simply encode plant biology knowledge uniformly, but are instead shaped by the visibility and frequency of information in their training corpora. This understanding is key to guiding both the development of next-generation models and the informed use of current tools in the everyday work of plant researchers. MoBiPlant is publicly available online in this link.Competing Interest StatementThe authors have declared no competing interest.},
    URL = {https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925},
    eprint = {https://www.biorxiv.org/content/early/2025/09/04/2025.08.31.672925.full.pdf},
    journal = {bioRxiv}
}
````

