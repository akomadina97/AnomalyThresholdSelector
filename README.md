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
```

## Usage
### Generating Datasets
To generate testing datasets with anomaly scores and labels, uncomment and run the following line in the `main` section of your script:

```python
generate_datasets()
```
### Running Evaluations
To run the evaluation of threshold selection methods on generated test data, configure and execute the `run` function with your desired settings. Below is an example configuration:

```python
run(
    threshold_selection_methods=['MODEL_NAME'],
    data_path='../Data/FirewallTestingData/',
    generation_method=0,
    plot=False,
    timeout_seconds=20000
)
```
#### Parameters
1. `threshold_selection_methods`: List of threshold methods to evaluate.
2. `data_path`: Path to the directory containing test data files.
3. `generation_method`: Method of data generation (0 for both, 1 for method 1, 2 for method 2).
4. `plot`: Whether to plot anomaly scores and labels.
5. `timeout_seconds`: Timeout in seconds for each threshold generation method.

## Threshold Methods
### Static Methods
1. MAX
2. STD_3
3. STD_4
4. STD_5
5. STD_6
6. PERCENTIL_97
7. PERCENTIL_98
8. PERCENTIL_99
9. IQR
10. DBScan
11. KMeans
12. POT_1
13. POT_2
14. OTSU
15. Density_distance
16. ECDF
17. LOF
18. AUCP
19. CHAU
20. CPD
21. DECOMP
22. DSN
23. EB
24. FGD
25. FILTER
26. FWFM
27. GESD
28. HIST
29. KARCH
30. MAD
31. MOLL
32. MTT
33. REGR
34. VAE
35. WIND
36. YJ

### Dynamic Methods
1. SPOT
2. SAX
3. LOF
4. Rolling_3
5. Rolling_5
6. EWM_3
7. EWM_5
8. Windowing_3
9. Windowing_5
10. Standard_deviation

### Supervised Methods
1. PR
2. EER
3. Youden
4. zero-miss
5. alpha-FPR

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any questions or issues, please contact the corresponding author Adrian Komadina at [adrian.komadina@fer.hr](mailto:adrian.komadina@fer.hr).
