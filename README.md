# Weighted Smooth Q-learning (WSQL)

This repository contains the implementation of the **Weighted Smooth Q-learning (WSQL)** algorithm, along with its applications to address two specific problems:

1. **Maximization Bias Example**
2. **Multi-Armed Bandit Example**

## Repository Structure

The repository is organized into the following two main folders:

1. **`maximization_bias/`**
    - Contains the implementation of the maximization bias example.
    - Demonstrates how WSQL mitigates maximization bias in reinforcement learning.

2. **`multi_armed_bandit/`**
    - Contains the implementation of the multi-armed bandit example.
    - Highlights how WSQL performs in decision-making scenarios involving bandits.

## Prerequisites

Before running the code, ensure you have the following installed:
- Python 3.8+
- NumPy 1.22+
- Matplotlib (for visualizations)

## Usage

Navigate to the folder corresponding to the example you want to run and execute the scripts. For instance:

### Running the Maximization Bias Example
```bash
cd maximization_bias
python bias.py
```

### Running the Multi-Armed Bandit Example
```bash
cd multi_armed_bandit
python roulette.py
```

Refer to the comments in each script for further customization options.

## Motivation and Acknowledgments

This implementation is heavily inspired by the excellent work available at the GitHub repository: [The Mean Squared Error of Double Q-Learning](https://github.com/wentaoweng/The-Mean-Squared-Error-of-Double-Q-Learning). Special thanks to the authors for making the code available online.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

For questions or suggestions, please feel free to open an issue or contact the repository owner.

