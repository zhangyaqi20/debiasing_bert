import pytorch_lightning as pl
import torch
import torch.nn as nn

class DebiasingNN(pl.LightningModule):
    def __init__(self, 
                 learning_rate,
                 weight_decay,
                 dim_ultradense = 1,
                 dim = 768,
                 ):
        super().__init__()

        self.lr = learning_rate
        self.weight_decay = weight_decay
        
        self.dim = dim
        self.dim_ultradense = dim_ultradense
        self.P = torch.Tensor([1]*self.dim_ultradense + [0]*(self.dim - self.dim_ultradense)).unsqueeze(1)
        self.Q = nn.Parameter(self._initialize_orthogonal(dim))

    def _initialize_orthogonal(self, dim):
        Q = torch.empty(dim, dim)
        nn.init.orthogonal_(Q)
        return Q
    
    def reorthogonalize(self):
        U, _, Vh = torch.linalg.svd(self.Q)
        self.Q = U @ Vh

    def forward(self, E_x, E_y):
        N = E_x.shape[0]
        assert E_x.shape[1] == self.dim
        # print(f"E_x = {E_x}")
        QE_x = E_x @ self.Q
        # print(f"QE_x = {QE_x}")
        PQE_x = torch.squeeze(QE_x @ self.P, dim=1)
        assert PQE_x.shape == (N, )

        assert E_y.shape[1] == self.dim
        QE_y = E_y @ self.Q
        PQE_y = torch.squeeze(QE_y @ self.P, dim=1)
        assert PQE_y.shape == (N, )

        return (PQE_x, PQE_y)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

# nn = DebiasingNN(learning_rate=5e-5, weight_decay=0.01)
# nn.P = torch.Tensor([1,0,0]).unsqueeze(1)
# nn.Q = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# nn.dim = 3
# x = torch.Tensor([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])
# y = torch.Tensor([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7]])
# print(nn(x, y))