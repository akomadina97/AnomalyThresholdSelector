# AnomalyThresholdSelector

**AnomalyThresholdSelector** is a comprehensive toolkit designed for evaluating and comparing various threshold selection methods for network anomaly detection. This repository accompanies the paper "Comparing Threshold Selection Methods for Network Anomaly Detection" by Adrian Komadina, Mislav Martinić, Stjepan Groš, and Željka Mihajlović.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Generating Datasets](#generating-datasets)
  - [Running Evaluations](#running-evaluations)
- [Threshold Methods](#threshold-methods)
  - [Static Methods](#static-methods)
  - [Dynamic Methods](#dynamic-methods)
  - [Supervised Methods](#supervised-methods)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Overview
**AnomalyThresholdSelector** implements multiple supervised and unsupervised threshold selection techniques. The methods are categorized into Statistics-based, Distribution-based, Clustering-based, Density-based, and Graphical-based approaches. This toolkit supports testing scenarios using real firewall log data, processed and combined with injected anomalies to evaluate method performance in terms of metrics like MCC, F1 Score, and execution time.

## Features
- **Static Methods**: MAX, STD, PERCENTIL, IQR, and more.
- **Dynamic Methods**: SPOT, SAX, LOF, and rolling and windowing methods.
- **Supervised Methods**: PR, EER, Youden, zero-miss, and alpha-FPR.
- **Multiprocessing Support**: Handles timeout for each threshold generation method.
- **Comprehensive Evaluation**: Compares methods using metrics like MCC, F1 Score, and execution time.
- **Plotting and Analysis**: Utilities for plotting anomaly scores, predicting labels, and evaluating performance metrics.

## Installation
To install the required dependencies, run:

```bash
pip install -r requirements.txt
