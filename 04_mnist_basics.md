# Terminology

**independent variables**: x, inputs
**dependent variables**: y, targets
**activation function**: The purpose of the activation function is to introduce non-linearity into the network, enabling it to learn and approximate complex, non-linear relationships between input features and output targets.
**loss function**: is a mathematical expression that quantifies the difference between the predicted outputs and the actual target values in deep learning models. The goal of training a neural network is to minimize the value of the loss function, which indicates that the neural network is making accurate predictions.

**Sigmoid**: The sigmoid function always outputs a number between 0 and 1. It's defined as follows:

```python
def sigmoid(x): return 1/(1+torch.exp(-x))
```

**Softmax**: we use the softmax activation function in the final layer to ensure that the activations are all between 0 and 1, and that they sum to 1.

```python
def softmax(x): return exp(x) / exp(x).sum(dim=1, keepdim=True)
```

# POE: how do you convert rank-3 tensor to rank-2 tensor?

To convert a rank-3 tensor to a rank-2 tensor, you can reshape or flatten the tensor along one or more dimensions. There are several ways to do this, depending on the desired structure of the resulting rank-2 tensor. Here are a few common approaches using PyTorch:

Flatten along one dimension: You can combine two dimensions of the rank-3 tensor into a single dimension, effectively flattening the tensor along one dimension. Here's an example using PyTorch's reshape() function:
import torch

```python
# Create a rank-3 tensor with dimensions 2x3x4
tensor_3d = torch.tensor([
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ],
    [
        [13, 14, 15, 16],
        [17, 18, 19, 20],
        [21, 22, 23, 24]
    ]
])

# Reshape the tensor to have dimensions 6x4 (flattening along the first dimension)
tensor_2d = tensor_3d.reshape(6, 4)

# Print the reshaped tensor
print(tensor_2d)
In this example, we have converted the rank-3 tensor with shape (2, 3, 4) into a rank-2 tensor with shape (6, 4) by combining the first two dimensions (2 and 3) into a single dimension (6).
```

Flatten the entire tensor: You can also flatten the entire rank-3 tensor into a one-dimensional vector and then reshape it into a rank-2 tensor. This can be done using the `view()` function in PyTorch:

```python
# Flatten the entire tensor into a 1D vector
tensor_flattened = tensor_3d.view(-1)

# Reshape the flattened tensor into a rank-2 tensor with dimensions 3x8
tensor_2d = tensor_flattened.view(3, 8)

# Print the reshaped tensor
print(tensor_2d)
```

In this example, we have flattened the entire rank-3 tensor with shape (2, 3, 4) into a one-dimensional vector and then reshaped it into a rank-2 tensor with shape (3, 8).

Note that when reshaping a tensor, the total number of elements must remain the same. For example, in both cases above, the rank-3 tensor has 2x3x4 = 24 elements, and the resulting rank-2 tensors also have 6x4 = 24 and 3x8 = 24 elements, respectively.

### What is `unsqueeze` method?

In PyTorch, the `unsqueeze()` method is used to add a new dimension (axis) to a tensor at the specified position. The new dimension will have a size of 1. This can be useful in cases where you need to match the shape of a tensor to the expected input of a function, or when you want to perform operations that require specific tensor shapes.

Here's an example of how to use unsqueeze():

```python
import torch

# Create a rank-1 tensor (vector) with 5 elements
tensor_1d = torch.tensor([1, 2, 3, 4, 5])

# Add a new dimension at index 0 (convert the rank-1 tensor to a rank-2 tensor)
tensor_2d = tensor_1d.unsqueeze(0)
print(tensor_2d)
print(f'Shape: {tensor_2d.shape}')  # Output: torch.Size([1, 5])

# Add a new dimension at index 1 (convert the rank-1 tensor to a rank-2 tensor)
tensor_2d = tensor_1d.unsqueeze(1)
print(tensor_2d)
print(f'Shape: {tensor_2d.shape}')  # Output: torch.Size([5, 1])
```

In this example, we create a rank-1 tensor with 5 elements and then use unsqueeze() to add a new dimension at different positions. When we add a new dimension at index 0, we obtain a rank-2 tensor with shape (1, 5). When we add a new dimension at index 1, we obtain a rank-2 tensor with shape (5, 1).

Keep in mind that unsqueeze() does not modify the original tensor. Instead, it returns a new tensor with the added dimension. If you want to modify the original tensor in-place, you can use the unsqueeze\_() method:

```python
tensor_1d.unsqueeze_(0)
print(tensor_1d)
print(f'Shape: {tensor_1d.shape}')  # Output: torch.Size([1, 5])
```

In this case, the `unsqueeze_()` method adds a new dimension to the original tensor at index 0, and the resulting tensor has shape (1, 5).

### How `Dataset` works

In PyTorch, a dataset is an object that provides access to a collection of data samples. Each sample typically consists of a feature tensor x and a label tensor y. The dataset should be indexable, meaning you can access a specific sample by its index. The zip function in Python is a convenient way to combine features and labels in the desired (x, y) format.

Here's a step-by-step explanation with an example:

Define your features (x) and labels (y):
Let's say you have a dataset with the following features and labels:

```python
features = [0.5, 1.2, 2.3, 3.1, 4.6]
labels = [1, 2, 3, 4, 5]
```

Use the zip function to combine features and labels:
zip takes two or more iterables and returns an iterator that generates tuples containing paired elements from the input iterables:

```python
zipped_data = zip(features, labels)
```

Convert the zipped data to a list:
As zip returns an iterator, you can convert it to a list for easier manipulation:

```python
data = list(zipped_data)
```

Now, data will be a list containing tuples of the form (x, y):

```python
[(0.5, 1), (1.2, 2), (2.3, 3), (3.1, 4), (4.6, 5)]
```

In this example, the dataset contains 5 samples, and each sample is represented as a tuple (feature, label).
