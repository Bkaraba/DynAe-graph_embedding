# DynAe GRAPH EMBEDDING METHOD
## Introduction
DynAe is an autoencoder-based model for dynamic graph embedding, which is a method for learning low-dimensional representations of nodes in a graph that can change over time.

Autoencoder models are neural networks that are trained to reconstruct their inputs, and they have been applied to a variety of tasks in computer vision, natural language processing, and other domains. In the context of dynamic graph embedding, DynAe uses an autoencoder architecture to learn embeddings that capture the evolving relationships between nodes in a dynamic graph.

## implementation
In this project we will implement DynAe graph embedding method using ogbg-molhiv dataset to train a model to learn node features and embeddin them on low dimension vector

## Requirements
- [Python3](https://www.python.org/downloads/)
- tensor flow
- ogb

good compute power.....on this am not kidding

## getting started
clone this repository

```
git clone git@github.com:Bkaraba/DynAe-graph_embedding.git
```
install packages
```
pip install tensorflow
pip install torch torch-scatter torch-sparse torch-geometric
pip install ogb
```
## run
```
python3 dynae.py
```
