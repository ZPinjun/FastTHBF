# Fast Tri-Hybrid Beamforming via Deep Unfolding

This repository provides the code for reproducing the main results presented in the following paper:

> [1] Pinjun Zheng, Md. Jahangir Hossain, and Anas Chaaban, “Fast Tri-Hybrid Beamforming via Deep Unfolding,” accepted for publication in IEEE Transactions on Signal Processing.

- **IEEE Xplore:** coming soon
- **Preprint (arXiv):** [arXiv:2608.27759](https://arxiv.org/abs/2608.27759)


This repository supports the experiments listed below and is licensed under MIT. OpenAI Codex assisted with refactoring this codebase, improving its documentation, and preparing it for public release. The authors remain responsible for the reported results.

## Supported experiments

| Notebook | Manuscript result | Required inputs |
| --- | --- | --- |
| `Fig00.ipynb` | Fig. 2: convergence and runtime | System parameters and Miami channels |
| `Fig01.ipynb` | Fig. 6: initialization/training-loss comparison | Training channels |
| `Fig02.ipynb` | Fig. 7: classical and unfolded CDF curves | Test instances; L=2,3,4 weights |
| `Fig03.ipynb` | Fig. 8 and Table III: runtime comparison | Test instances; L=2,4,6 weights |
| `Fig04.ipynb` | Fig. 9: performance versus unfolding depth | Test instances; L=2,...,7 weights |
| `Fig05.ipynb` | Fig. 10: noisy-CSI evaluation | Test instances; L=2,4 weights |
| `Fig06.ipynb` | Fig. 11: cross-city generalization | Four city datasets; L=2,4,6 weights |

Fig. 5, the direct-GNN curve of Fig. 7, and the scalability experiments in
Table IV were added in response to the reviewers’ requests. These results are not reproduced here, as the first author considers such repetition unnecessary. 

## Two workflows: network training and numerical evaluation

The code provides two parts:

1. **Network training:** this part prepares channel data, train the unfolded WMMSE algorithm, and save model checkpoints. The main notebooks are
   `A00_Data_generation.ipynb` and `A01_Train_UWMMSE.ipynb`.
2. **Numerical simulation and evaluation:** With the trained models, this part evaluates the performance of the considered methods using `Fig00.ipynb` through `Fig06.ipynb`. 

**If you are interested only in numerical evaluation, you can use the trained weights we provided.** After completing the [environment setup](#environment-setup), skip directly
to ['Part 2: Numerical simulation and evaluation'](#part-2-numerical-simulation-and-evaluation) using the provided model weights.
To run your own model training, follow ['Part 1: Network training'](#part-1-network-training).

## Environment setup

Install Python 3.11 first (the checked environment uses 3.11.13). The `venv`
command creates an environment from an installed interpreter; it does not install
Python itself. The **repository root** is the `FTHBF-public` folder containing this
`README.md`, `requirements.txt`, and the notebooks. Run terminal commands there.

### macOS / Linux (bash or zsh)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If `python3.11` is not found, use the full path to an installed Python 3.11
interpreter in the first command. If `python --version` already reports 3.11,
`python -m venv .venv` is also suitable. A Conda base environment may use a
different Python version, so check its version before creating the environment.

### Windows (PowerShell)

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The first command must report Python 3.11. If `py` is not recognized but
`python --version` reports 3.11, use `python -m venv .venv` instead. Otherwise,
install/configure a 64-bit Python 3.11 interpreter and its launcher, then reopen
PowerShell. See the [Python on Windows guide](https://docs.python.org/3.11/using/windows.html).

If PowerShell blocks activation, invoke the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In that case, replace `python` in later commands with
`.\.venv\Scripts\python.exe`. See the
[virtual environment guide](https://docs.python.org/3.11/library/venv.html).

**Once the environment is activated, the `python -m ...` commands in both workflow
sections are the same on Windows, macOS, and Linux.** Check that `python --version`
reports 3.11 before continuing.

The requirements pin versions checked on macOS arm64. A fresh installation on
other platforms has not been validated. The solver uses CPU tensors; GPU/MPS
execution is not a supported configuration. DeepMIMO dependencies are needed only
when generating channel data, as described in the relevant sections below.


## Part 1: Network training

### Prepare training data

Training requires `data_Probs/parameters.pt` and
`data_Probs/channels_deepMIMO_train.pt`. Choose **one** of the following ways to
prepare them:

**Option A: Use the supplied training data.** Download `fthbf-data-v1.zip` from the
[Google Drive download folder](https://drive.google.com/drive/folders/1zEdSEqWkpc1nvKUOKtuseE9iOGGQOOEe)
and extract it into the repository root, following the
[folder layout](#obtain-supplied-data-and-pretrained-weights) below. 
This option skips DeepMIMO scene downloads and channel generation.

**Option B: Generate the training data yourself.** Install the additional dependencies, download the training scenes,
and generate the required files:

```bash
python -m pip install -r requirements-data.txt
python -m scripts.download_scenarios --group train
python -m scripts.generate_data --group train
```

For interactive data generation, `A00_Data_generation.ipynb` contains training
and Miami test-data cells. Run the training-data cell after downloading the
training scenes; the test-data cell separately requires the Miami test scene.
See [data preparation](docs/DATA.md) for exact scenarios and parameters.

After completing either option, continue to training. **Training from scratch
requires no pretrained weights.** The warm-start example below uses the six-layer
checkpoint supplied in `fthbf-weights-v1.zip`.

### Train the unfolded networks

For an interactive training run, open `A01_Train_UWMMSE.ipynb` in JupyterLab or
VS Code, select the project environment as the notebook kernel, inspect the
configuration, and run the cells in order. The notebook defaults to
`configs/train_scratch.json`. The [notebook execution instructions](#run-the-experiments)
below apply to training as well as evaluation.

The same training routine is available from the terminal:

```bash
python -m scripts.train --config train_scratch.json --output-dir results/my-training
```

This example starts a two-layer model from scratch and saves its checkpoint,
configuration, and train/validation split in `results/my-training/`. To first
check a single optimizer batch on four instances:

```bash
python -m scripts.train --smoke --output-dir results/training-smoke
```

A warm-start example initializes a seven-layer model from the supplied six-layer
weights and creates a new optimizer:

```bash
python -m scripts.train --config train_warmstart.json --output-dir results/my-warmstart
```

## Part 2: Numerical simulation and evaluation

If you have trained your own network through the above steps, you can skip to ['Prepare and verify evaluation inputs'](#prepare-and-verify-evaluation-inputs). Otherwise, you can use the model weights we have trained. You can access these trained model weights using the following steps.

### Obtain Supplied data and pretrained weights

Download `fthbf-data-v1.zip` and `fthbf-weights-v1.zip` from the
[Google Drive download folder](https://drive.google.com/drive/folders/1zEdSEqWkpc1nvKUOKtuseE9iOGGQOOEe).
**Extract each ZIP into the repository root, `FTHBF-public`.** For
example, if your project is `H:\FTHBF-public`, choose that folder as the extraction
destination for both archives.  

| Archive | Extracted folder | Used for |
| --- | --- | --- |
| `fthbf-data-v1.zip` | `FTHBF-public/data_Probs/` | Training channels, system parameters, and evaluation datasets |
| `fthbf-weights-v1.zip` | `FTHBF-public/checkpoints/` | Pretrained unfolded models with depths 2–7 |

The folder architecture has to be as follows.

```text
FTHBF-public/
├── README.md
├── requirements.txt
├── A00_Data_generation.ipynb
├── A01_Train_UWMMSE.ipynb
├── A02_Data_MultipleCities.ipynb
├── Fig00.ipynb ... Fig06.ipynb
├── scripts/
├── configs/
├── data_Probs/
│   ├── parameters.pt
│   ├── channels_deepMIMO_train.pt
│   ├── channels_deepMIMO_test.pt
│   ├── probInsts_test.pt
│   ├── probInsts_Miami.pt
│   ├── probInsts_Austin.pt
│   ├── probInsts_Columbus.pt
│   └── probInsts_Dallas.pt
└── checkpoints/
    ├── unfolded_wmmse_model_2Iter.pt
    ├── unfolded_wmmse_model_3Iter.pt
    ├── unfolded_wmmse_model_4Iter.pt
    ├── unfolded_wmmse_model_5Iter.pt
    ├── unfolded_wmmse_model_6Iter.pt
    └── unfolded_wmmse_model_7Iter.pt
```


### Prepare and verify evaluation inputs

Verify the files and run a
small evaluation:

```bash
python -m scripts.verify_artifacts
python -m scripts.smoke_check --instances 2 --layers 2
```

The first command should report `Verified 14 release files.` A `Missing` message
identifies an absent file; check the extracted directory structure. The second
command evaluates two instances and writes `results/smoke-check.json`.

If you prefer to regenerate evaluation data, install `requirements-data.txt`
and run the following instead of using the supplied data archive:

```bash
python -m scripts.download_scenarios --group test
python -m scripts.generate_data --group test
python -m scripts.download_scenarios --group generalization
python -m scripts.generate_data --group generalization
```

The generalization group is needed for Fig06. Fig01 additionally needs training
channels; generate the training group as described in Part 1.
`A00_Data_generation.ipynb` contains the Miami test-data cell, and `A02_Data_MultipleCities.ipynb` generates the
cross-city datasets. Hash verification above applies to the supplied release
files; regenerated data uses the separate checks described in [data preparation](docs/DATA.md).

### Run the experiments

After the checks pass, you can run the supported paper experiments.
Open a notebook in JupyterLab or VS Code, select the project environment as its
kernel, and run the cells in order. Alternatively, run it from the repository root:

```bash
python -m scripts.run_notebook Fig02.ipynb
```

Replace `Fig02.ipynb` with another supported notebook to run a different experiment.

## Configuration and outputs

- `configs/data.json`: system, channel, filtering, and instance-generation settings.
- `configs/experiments.json`: seeds, layer/iteration counts, and sample counts.
- `configs/train_scratch.json` and `train_warmstart.json`: training examples.
- `configs/scenarios.json`: exact scenario names and reference metadata hashes.
- `configs/artifacts.json`: checkpoint/data sizes, architectures, and SHA-256 hashes.
- `reference_results/`: numeric curves extracted from the manuscript, labeled as references.


## Citation and attribution

If you use this code, or any modified part of it, in your work, please cite the following paper:

```bibtex
@article{zheng2026fast,
  author  = {Zheng, Pinjun and Hossain, Md Jahangir and Chaaban, Anas},
  title   = {Fast Tri-Hybrid Beamforming via Deep Unfolding},
  journal = {IEEE Transactions on Signal Processing},
  year    = {2026},
  note    = {in press}
}
