# AGi-cooking

This project implements the Sparse Evolutionary Training (SET) algorithm on the MNIST dataset using PyTorch. It evolves a sparse neural network by pruning and regrowing weights at each epoch — mimicking biological rewiring.

## 🌱 Highlights
- SparseLinear layers initialized with Erdős–Rényi graphs
- Evolution mechanism: prune + regrow every epoch
- Achieves >98% test accuracy on MNIST
- Clean, minimal PyTorch implementation

## 📦 Files
- `set_mnist.ipynb`: Complete working notebook (Colab-friendly)
- `requirements.txt`: Dependencies for local setup

## 🧠 Inspired By
- Mocanu et al., 2018 – [SET paper](https://arxiv.org/abs/1706.04303)
- AGI research into sparse, evolving architectures

## 🔧 To Run Locally

```bash
pip install -r requirements.txt
jupyter notebook

## 📈 Sample Output

You’ll see accuracy improving over epochs:

Epoch 20/20 → Loss: 0.0142 | Train Acc: 99.94% | Test Acc: 98.26%
