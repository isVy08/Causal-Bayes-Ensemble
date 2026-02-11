# Causal Discovery Ensemble

An ensemble learning framework for causal graph discovery that aggregates predictions from multiple causal discovery algorithms.

## Pipeline Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Step 1         │     │  Step 2         │     │  Step 3         │
│  run_algo.py    │ ──▶ │  main.py        │ ──▶ │  predict.py     │
│  (Run Experts)  │     │  (Train Model)  │     │  (Aggregate)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Step 1: Generate Expert Predictions (`run_algo.py`)

Run individual causal discovery algorithms to generate graph predictions.

```bash
python run_algo.py <sem_type> <method>
```

**Arguments:**
- `sem_type`: Dataset name/type (e.g., `sachs`, `MLP`, `GP`)
- `method`: Algorithm name (`DAGMA`, `NOTEARS`, `GES-BIC`, `PC`, `LiNGAM`)

**Example:**
```bash
python run_algo.py sachs DAGMA
python run_algo.py sachs NOTEARS
python run_algo.py sachs GES-BIC
python run_algo.py sachs PC
python run_algo.py sachs LiNGAM
```

**Output:** `output/<sem_type>/<method>.pkl` - List of predicted adjacency matrices

**Supported Algorithms:**
| Method | Description |
|--------|-------------|
| `DAGMA` | DAG learning with M-matrices and Acyclicity |
| `NOTEARS` | Non-combinatorial structure learning via continuous optimization |
| `GES-BIC` | Greedy Equivalence Search with BIC score |
| `PC` | PC algorithm with partial correlation |
| `PC-KCI` | PC algorithm with KCI conditional independence test |
| `LiNGAM` | Linear Non-Gaussian Acyclic Model |

---

## Step 2: Train Ensemble Model (`main.py`)

Learn expert competence matrices and prior distributions from expert predictions.

```bash
python main.py <ngraph> <sem_type>
```

**Arguments:**
- `ngraph`: Number of expert graphs to sample per method
- `sem_type`: Dataset name/type

**Example:**
```bash
python main.py 50 sachs
```

**Process:**
1. Load expert predictions from Step 1
2. Train Forwarder (transition matrices) and Backwarder (prior inference) models
3. Repeat training 30 times for ensemble averaging
4. Save learned parameters

**Output:** `output/<sem_type>/graph-<ngraph>.pkl` - Tuple of (logT, logPr)
- `logT`: Log transition matrices `[num_features, num_methods, num_classes, num_classes]`
- `logPr`: Log prior probabilities `[num_features, num_classes]`

**Hyperparameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_epochs` | 500 | Training epochs per run |
| `batch_size` | 10000 | Batch size |
| `eta` | 0.001 | KL divergence regularization weight |
| `rpts` | 30 | Number of ensemble repetitions |

---

## Step 3: Aggregate Predictions (`predict.py`)

Combine expert predictions using learned parameters to produce final graph estimate.

```bash
python predict.py <ngraph> <sem_type>
```

**Arguments:**
- `ngraph`: Number of graphs (must match Step 2 to load correct model)
- `sem_type`: Dataset name/type

**Input:** Loads `output/<sem_type>/graph-<ngraph>.pkl` from Step 2

**Example:**
```bash
python predict.py 50 sachs
```

**Aggregation Strategies:**

The script evaluates two multi-step ensembling options:

**Option 1: Inter-algo → Intra-algo**
1. Aggregate graphs within each algorithm (majority/rank voting)
2. Combine across algorithms (Bayes/majority/rank voting)

**Option 2: Intra-algo → Inter-algo**
1. Combine across algorithms per voting profile (Bayes/majority/rank)
2. Aggregate across profiles (majority/rank voting)

**Voting Methods:**
| Method | Description |
|--------|-------------|
| `majority` | Mode of edge labels across graphs |
| `rank` | DAG aggregation via greedy center (minimizes DAG distance) |
| `bayes` | Weighted voting using learned T and Pr parameters |

**Output:** Prints SHD (Structural Hamming Distance) and F1 scores for each strategy combination.

---

## Quick Start

```bash
# 1. Run all expert algorithms
for method in DAGMA NOTEARS GES-BIC PC LiNGAM; do
    python run_algo.py sachs $method
done

# 2. Train ensemble model (using 50 graphs per method)
python main.py 50 sachs

# 3. Evaluate aggregation strategies
python predict.py 50 sachs
```

**Note:** The `ngraph` argument in Steps 2 and 3 must match to load the correct trained parameters.

---

## Project Structure

```
.
├── main.py              # Step 2: Train ensemble model
├── run_algo.py          # Step 1: Run expert algorithms
├── predict.py           # Step 3: Aggregate predictions
├── trainer.py           # Trainer class and model utilities
├── baseline.py          # DAG aggregation algorithms
├── simulation.py        # Simulation experiments
├── synthetic.py         # Synthetic dataset generation
├── real.py              # Real dataset loading
├── data_generator.py    # Dataset configuration
├── utils_ensb.py        # Ensemble utilities
├── models/
│   ├── ensemble.py      # Forwarder/Backwarder neural networks
│   ├── dagma.py         # DAGMA algorithm
│   └── notears.py       # NOTEARS algorithm
├── utils/
│   ├── eval.py          # Evaluation metrics (SHD, F1)
│   ├── graph.py         # Graph utilities
│   ├── data.py          # Data processing
│   ├── io.py            # I/O utilities
│   ├── trainer.py       # Training utilities
│   └── sampler.py       # Sampling utilities
├── dataset/             # Data storage
└── output/              # Results storage
```

---

## Requirements

```bash
pip install -r requirements.txt
```

See `requirements.txt` for full list of dependencies.
