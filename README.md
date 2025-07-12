# AI‑ML‑Driven Network Security

> **Smarter traffic analysis & threat detection for encrypted, high-throughput networks.**  
> • **Random Forest**-based flow classification • **Isolation Forest** anomaly discovery • **Privacy-first feature handling**

---

## Table of Contents

1. [Overview](#overview)  
2. [Core Highlights](#core-highlights)  
3. [Setup Instructions](#setup-instructions)  
4. [Getting Started](#getting-started)  
5. [Project Layout](#project-layout)  
6. [How to Use](#how-to-use)  
7. [Example Results](#example-results)  
8. [Future Plans](#future-plans)  
9. [How to Contribute](#how-to-contribute)  
10. [License](#license)  

---

## Overview

`AI‑ML‑Driven Network Security` is a Python-based solution tailored to the challenges of traffic visibility in encrypted and IoT-rich environments.  
It leverages ensemble ML methods and streamlined preprocessing to:

- **Identify** network sessions as either *legitimate* or *malicious*, supporting multi-class output.  
- **Spot** unseen (zero-day) anomalies with minimal delay.  
- **Protect privacy** by locally summarizing sensitive data before export.

The toolkit delivers a **Macro‑F1 score ~0.92** on NSL‑KDD and handles ~10k flows/sec on standard laptop hardware.

---

## Core Highlights

- 🔍 **Flow Classification** — Optimized Random Forest (up to 100 trees, depth capped at 10).  
- 🚨 **Anomaly Detection** — Isolation Forest with contamination rate ≤1%.  
- 🛠️ **Modular & Reproducible** — Easily switch models or datasets via config flags.  
- ⚡ **Low Resource Usage** — Operates efficiently within 2 GB RAM; no GPU needed.  
- 🔐 **Data Privacy** — Optional local aggregation to avoid raw packet leakage.

---

## Setup Instructions

```bash
# 1. Clone the repository
$ git clone https://github.com/sriya-vadla/AI-ML-for-Networking.git
$ cd AI-ML-for-Networking

# 2. Set up a virtual environment
$ python -m venv venv && source venv/bin/activate  # For Windows: venv\Scripts\activate

# 3. Install Python dependencies
$ pip install -r requirements.txt
```

> **Dataset** — NSL‑KDD will auto-download on first use.  
> For custom CSV or PCAP flows, refer to [`docs/DATASET.md`](docs/DATASET.md).

---

## Getting Started

Run the training and prediction steps:

```bash
# Build and evaluate models
$ python run_analysis.py --mode train

# Classify flows using a CSV input
$ python run_analysis.py --mode predict --input sample/flows.csv --output predictions.csv
```

---

## Project Layout

```text
AI-ML-for-Networking/
│
├── preprocessing.py       # Data transformation & feature creation
├── models.py              # ML pipeline definitions and tuning
├── run_analysis.py        # Main controller script
│
├── sample/                # Demo configs & sample inputs
├── tests/                 # Test cases for validation
├── requirements.txt       # Dependency list
└── docs/                  # Documentation files
    └── DATASET.md         # Custom dataset usage guide
```

---

## How to Use

Commonly used CLI flags:

```bash
--mode {train,predict,benchmark}
--input  path/to/flows.csv      # required for predict/benchmark
--output path/to/out.csv        # default: predictions.csv
--model_dir models/             # directory for saved models
--log_level {info,debug}
```

For a full list, run:

```bash
python run_analysis.py --help
```

### Real-Time Capture (Beta)

```bash
$ python run_analysis.py --mode live --iface eth0
```

> Requires `scapy` or `pyshark`. Real-time packets are processed into flow summaries.

---

## Example Results

```
               precision    recall  f1-score   support

        Benign     0.95      0.96      0.96     9711
       DoS_Hulk     0.93      0.91      0.92     2315
   Probe_PortScan   0.89      0.88      0.88     1337
          …

    Macro‑F1 ≈ 0.92
```

A detailed HTML evaluation report is saved to `reports/latest/` after each training run.

---

## Future Plans

Planned improvements:

- 🧠 Incorporation of deep learning (e.g., autoencoders or transformers)  
- 🧪 Broader dataset benchmarks (CICIDS, UNSW-NB15)  
- 📦 Dockerized deployment support  
- 📉 Integration with Grafana for real-time monitoring

---

## How to Contribute

1. **Fork** this repository and create a new branch (`git checkout -b feat/my-feature`).  
2. **Commit** your changes with meaningful messages.  
3. **Push** your branch (`git push origin feat/my-feature`).  
4. **Submit a PR** explaining your changes and purpose.

Run `pytest` locally to validate changes.  
Refer to [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding and CI rules.

---

## License

This project is licensed under the **MIT License**.  
See [`LICENSE`](LICENSE) for details.
