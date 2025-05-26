
## 🚀 What is this?

A from-scratch implementation of:

* **Automatic differentiation** (backpropagation)
* **Computational graph construction**
* **Manual neural networks using scalar operations**
* Mini-Machine Learning framework with forward and backward passes


---

## 📦 Features

* Custom scalar `Value` class
* Tracks operations and builds computation graph
* Supports: `+`, `*`, `tanh`, `pow`, and gradients
* Backpropagation via `.backward()`
* Mini MLP framework using the engine
* Dead simple but **foundational for PyTorch-style autograd**

---


## 🔍 Core Concepts

### 1. `Value` Object

```python
v = Value(3.0)
```

* Holds a scalar float (`data`)
* Tracks gradients (`grad`)
* Stores backward function and parents for autograd

### 2. Forward Pass

```python
x = Value(2.0)
y = Value(3.0)
z = x * y + x
```

Each operation builds a node in the **computation graph**.

### 3. Backward Pass

```python
z.backward()
```

This triggers **reverse-mode autodiff**, walking the graph in topological order and applying chain rule.

---

## 🔧 Sample Neural Net

```python
from micrograd.engine import Value
from micrograd.nn import Neuron, Layer, MLP

mlp = MLP(nin=3, nouts=[4, 4, 1])  # 3-layer MLP

x = [2.0, 3.0, -1.0]
y = mlp(x)  # forward pass

y.backward()  # backprop
```

Train it like a mini PyTorch model. Fully connected layers. Tanh activations. No bullshit.

---

## 🧪 How to Run

```bash
git clone https://github.com/karpathy/micrograd.git
cd micrograd
python demo.py
```

Or try the [Jupyter notebook](https://github.com/karpathy/micrograd/blob/master/demo.ipynb) for visual exploration.

---

## 🧱 Code Structure

```
micrograd/
│
├── engine.py      # Core autograd engine
├── nn.py          # Neural network components (Neuron, MLP, etc.)
└── demo.py        # Sample MLP training on a toy dataset
```

---

## 🔥 Bonus Challenges

If you're serious, don't just clone. **Master it. Modify it. Rewrite it.**

* Add more activation functions (ReLU, Sigmoid)
* Vectorize for mini-batching
* Add visualization of the computation graph
* Implement gradients checks
* Extend to matrix-valued tensors
* Rebuild it in PyTorch for comparison

---

## 🧠 Learn This First Before GPT

If you're building GPT-style LLMs or transformer models:

> You *must* understand backprop like this. This is the atomic layer behind all of it.

---




