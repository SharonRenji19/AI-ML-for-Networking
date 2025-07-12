# Network Intelligence using AI & ML

> **Modern traffic insight & encrypted anomaly detection using Machine Learning**  
> • **Random Forest** for flow categorization • **Isolation Forest** for unknown threat discovery • **Privacy-aware feature engineering**

---

## Contents

1. [Introduction](#introduction)  
2. [Highlights](#highlights)  
3. [Setup Guide](#setup-guide)  
4. [Getting Started](#getting-started)  
5. [Directory Layout](#directory-layout)  
6. [Usage Instructions](#usage-instructions)  
7. [Performance Results](#performance-results)  
8. [Planned Enhancements](#planned-enhancements)  
9. [Contributions](#contributions)  
10. [Legal Info](#legal-info)  

---

## Introduction

`Network Intelligence using AI & ML` offers a lightweight Python toolkit designed to analyze encrypted or complex network traffic.  
Utilizing classical ensemble techniques and fast preprocessing, it enables:

- **Flow labeling** as either *normal* or *attack* with multi-label support.  
- **Zero-day threat identification** in real-time scenarios.  
- **Data confidentiality** by transforming raw packets into anonymized features locally.

The solution achieves a **Macro F1 score of ~0.92** on NSL‑KDD while managing approximately 10,000 flows per second on a standard CPU.

---

## Highlights

- 🎯 **Supervised Classification** — Optimized Random Forest (50–100 estimators, max depth of 10).  
- ⚠️ **Outlier Detection** — Unsupervised Isolation Forest tuned with ≤1% contamination.  
- 🧩 **Adaptable Framework** — Easily reconfigurable for different models or datasets.  
- 💡 **Lightweight Execution** — Requires less than 2 GB RAM; works without GPUs.  
- 🔐 **Secure Design** — On-site packet abstraction ensures sensitive data never leaves the origin.

---

## Setup Guide

```bash
# Clone the project
$ git clone https://github.com/sriya-vadla/AI-ML-for-Networking.git
$ cd AI-ML-for-Networking

# Activate a virtual environment
$ python -m venv venv && source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the required packages
$ pip install -r requirements.txt
```

> **Note**: NSL-KDD data is downloaded automatically on the initial run.  
> To integrate your PCAP or CSV data, consult [`docs/DATASET.md`](docs/DATASET.md).

---

## Getting Started

Use the following commands to train and test the pipeline:

```bash
# Train models and assess performance
$ python run_analysis.py --mode train

# Make predictions on new data
$ python run_analysis.py --mode predict --input sample/flows.csv --output predictions.csv
```

---

## Directory Layout

```text
AI-ML-for-Networking/
│
├── preprocessing.py       # Handles input cleanup and feature setup
├── models.py              # ML logic and hyperparameter configurations
├── run_analysis.py        # Main entry point with CLI
│
├── sample/                # Sample data and configuration files
├── tests/                 # Testing scripts
├── requirements.txt       # List of Python packages
└── docs/                  # Additional help and guides
    └── DATASET.md         # Instructions for using custom data
```

---

## Usage Instructions

Available CLI flags:

```bash
--mode {train,predict,benchmark}
--input  path/to/flows.csv      # mandatory for predict/benchmark modes
--output path/to/out.csv        # default is predictions.csv
--model_dir models/             # model storage directory
--log_level {info,debug}
```

To see all options, run:

```bash
python run_analysis.py --help
```

### Real-Time Monitoring (Beta)

```bash
$ python run_analysis.py --mode live --iface eth0
```

> Requires `scapy` or `pyshark`. Real-time packets are translated into flow summaries before predictions.

---

## Performance Results

```
               precision    recall  f1-score   support

        Benign     0.95      0.96      0.96     9711
       DoS_Hulk     0.93      0.91      0.92     2315
   Probe_PortScan   0.89      0.88      0.88     1337
          …

    Macro-F1 ≈ 0.92
```

A comprehensive performance report (including confusion matrix) is saved in `reports/latest/`.

---

## Planned Enhancements

Some upcoming goals:

- 🔬 Add neural network-based approaches (Autoencoders, Transformers)  
- 📊 Include additional datasets (e.g., CICIDS, UNSW-NB15)  
- 🐳 Create Docker container for deployment  
- 📈 Real-time dashboard with Grafana integration

---

## Contributions
Want to contribute?

1. **Fork** the repository and create a branch (`git checkout -b feature/your-feature`).  
2. **Commit** changes (`git commit -m "feature: description"`).  
3. **Push** to GitHub (`git push origin feature/your-feature`).  
4. **Open a Pull Request** with a short explanation.

## Legal Info

This project is covered under the **MIT License**.  
Read [`LICENSE`](LICENSE) for the full license text.
