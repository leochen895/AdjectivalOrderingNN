import torch
from torch import nn
from torch import optim

class AdjOrderModel(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2):
        super(AdjOrderModel, self).__init__()
        
        self.hidden_dim1 = hidden_dim1;
        self.hidden_dim2 = hidden_dim2;
        self._linear1 = nn.Linear(100, self.hidden_dim1)
        self._linear2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self._linear3 = nn.Linear(self.hidden_dim2, 1)

        self._sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out1 = self._sigmoid(self._linear1(x))
        out2 = self._sigmoid(self._linear2(out1))
        y_pred = self._sigmoid(self._linear3(out2))
        
        return y_pred
    
# def tokenizer(text):
#     return [tok.text for tok in space_en.tokenizer(text)]

# def prepare_xy():
#     x_field = data.Field(sequential=True, tokenize=tokenizer, lower=True)
#     y_field = data.Field(sequential=False, use_vocab=False)
#     fields = [("x_field", x_field), ("y_field", y_field)]
#     train_data, val_data, test_data = UDPOS.splits(fields, root="data")

#     x_field.build_vocab(train, vectors="glove.6B.100d")
