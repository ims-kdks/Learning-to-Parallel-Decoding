from torch import nn

# # 1 layer
# class LogisticRegression(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.linear1 = nn.Linear(input_dim, input_dim)
    
#     def forward(self, x):
#         original_dim = x.shape[-1]
#         if original_dim != self.input_dim:
#             x = nn.functional.pad(x, pad=(0, self.input_dim - original_dim))

#         x = self.linear1(x)
        
#         if original_dim != self.input_dim:
#             x = x[:, :original_dim]
#         return x

# 2 layer
class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(input_dim, input_dim)
    
    def forward(self, x):
        original_dim = x.shape[-1]
        if original_dim != self.input_dim:
            x = nn.functional.pad(x, pad=(0, self.input_dim - original_dim))
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        
        if original_dim != self.input_dim:
            x = x[:, :original_dim]
        return x

# # 4 layer
# class LogisticRegression(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
#         self.linear1 = nn.Linear(input_dim, input_dim)
#         self.relu = nn.ReLU()
#         self.linear2 = nn.Linear(input_dim, input_dim)
#         self.linear3 = nn.Linear(input_dim, input_dim)
#         self.linear4 = nn.Linear(input_dim, input_dim)
    
#     def forward(self, x):
#         original_dim = x.shape[-1]
#         if original_dim != self.input_dim:
#             x = nn.functional.pad(x, pad=(0, self.input_dim - original_dim))
        
#         x = self.linear1(x)
#         x = self.relu(x)
#         x = self.linear2(x)
#         x = self.relu(x)
#         x = self.linear3(x)
#         x = self.relu(x)
#         x = self.linear4(x)

#         if original_dim != self.input_dim:
#             x = x[:, :original_dim]
#         return x