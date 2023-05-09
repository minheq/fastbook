import torch
import torch.nn as nn

categories = ["red", "green", "blue"]
category_to_index = {category: index for index, category in enumerate(categories)}
index = category_to_index["green"]  # Example input: 'green'
one_hot_encoded = torch.zeros(len(categories)).scatter_(0, torch.tensor(index), 1)
print(one_hot_encoded)


class ExampleModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ExampleModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x


# Define model parameters
input_dim = len(categories)  # Number of unique categories
# Dimension of the dense vector representation (embedding size)
output_dim = 4

# Create the model
model = ExampleModel(input_dim, output_dim)

# Pass the one-hot encoded data through the model
one_hot_encoded_input = torch.tensor(index).unsqueeze(
    0
)  # Convert index to tensor with batch dimension
dense_vector_output = model(one_hot_encoded_input)
print(dense_vector_output)
