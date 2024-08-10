import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from loss.WasstesteinDist import WassersteinDist

class Debiasing(pl.LightningModule):
    def __init__(self, 
                 learning_rate,
                 weight_decay,
                 dim_ultradense = 1,
                 dim = 768,
                 ):
        super().__init__()

        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = WassersteinDist()

        self.dim = dim
        self.dim_ultradense = dim_ultradense
        self.P = torch.Tensor([1]*self.dim_ultradense + [0]*(self.dim - self.dim_ultradense))
        self.Q = torch.Tensor(self.dim, self.dim)
        nn.init.xavier_uniform_(self.Q)

        

    def forward(self, E_x, E_y):
        N = E_x.shape[1]
        assert E_x.shape[1] == self.dim
        # print(f"E_x = {E_x}")
        QE_x = torch.matmul(self.Q, E_x.T)
        # print(f"QE_x = {QE_x}")
        PQE_x = torch.matmul(self.P, QE_x)
        # print(f"PQE_x = {PQE_x}")
        assert PQE_x.shape == (N,)

        
        assert E_y.shape[1] == self.dim
        # print(f"E_y = {E_y}")
        QE_y = torch.matmul(self.Q, E_y.T)
        # print(f"QE_y = {QE_y}")
        PQE_y = torch.matmul(self.P, QE_y)
        # print(f"PQE_y = {PQE_y}")
        assert PQE_y.shape == (N,)

        return (PQE_x, PQE_y)

    def training_step(self, batch, batch_idx):
        E_star_X, E_star_Y = batch
        loss = self.loss_fn(E_star_X, E_star_Y)
        return loss
        
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"train_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        return avg_loss
        
    def validation_step(self, batch, batch_idx):
        E_star_X, E_star_Y = batch
        loss = self.loss_fn(E_star_X, E_star_Y)
        return loss
    
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"val_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        return avg_loss

    def test_step(self, batch, batch_idx):
        E_star_X, E_star_Y = batch
        loss = self.loss_fn(E_star_X, E_star_Y)
        return loss
    
    def on_test_epoch_end(self):
        avg_loss = torch.stack(self.train_losses).mean()
        self.log_dict({"test_loss": avg_loss}, on_step=False, on_epoch=True, prog_bar=True)
        self.train_losses = []
        return avg_loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


# nn = Debiasing(learning_rate=5e-5, weight_decay=0.01)
# nn.P = torch.Tensor([1,0,0])
# nn.Q = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])
# nn.dim = 3
# x = torch.Tensor([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])
# y = torch.Tensor([[0.1, 0.3, 0.5], [0.3, 0.5, 0.7]])
# # set: embeddings_x = x.T; embeddings_y = y.T
# print(nn(x, y))