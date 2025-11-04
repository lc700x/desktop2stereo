import torch

# Example with a function
@torch.compile
def my_function(x):
    return x * 2 + 1

# Example with a module
class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
model.to('cuda')

compiled_model = torch.compile(model)