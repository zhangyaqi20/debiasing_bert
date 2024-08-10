import pytorch_lightning as pl
from base import BaseTrainer
from tqdm import tqdm
from loss.WasstesteinDist import WassersteinDist

class DebiasingTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = WassersteinDist()

    def fit(self, model, embeddings):
        pbar = tqdm(total=1, desc='Training')
        E_x, E_y = embeddings
        for epoch in range(self.max_epochs):
            model.train()

            E_star_x, E_star_y = model(E_x, E_y)
            loss = self.loss_fn(E_star_x, E_star_y)

            model.zero_grad()
            loss.backward()
            model.optimizers[0].step()

            model.reorthogonalize()

            print(f'Epoch {epoch + 1}/{self.max_epochs}, Loss: {loss.item}')
        
            pbar.set_postfix({'Loss': loss.item})
            pbar.close()
