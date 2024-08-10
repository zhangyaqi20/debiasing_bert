import logging
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from loss.WasstesteinDist import WassersteinDist

logger = logging.getLogger(__name__)

class DebiasingTrainer(pl.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = WassersteinDist()
        self.best_ckpt_path = "./checkpoints/ckpt.pt"
        
    def fit(self, model, embeddings):
        self.optimizer = model.configure_optimizers()
        minLoss = float('inf')
        E_x, E_y = embeddings
        for epoch in range(self.max_epochs):
            E_star_x, E_star_y = model(E_x, E_y)
            loss = self.loss_fn(E_star_x, E_star_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            model.reorthogonalize()
            logger.info(f'Epoch {epoch + 1}/{self.max_epochs}, Loss: {loss.item()}')
            if loss.item() < minLoss:
                minLoss = loss.item()
                torch.save(model, self.best_ckpt_path)
    
    def test(self, datamodule):
        with torch.no_grad():
            # Compute bias before transformation
            sim_woman = F.cosine_similarity(datamodule.eval_E, datamodule.woman_info.embedding.unsqueeze(0), dim=1)
            sim_man = F.cosine_similarity(datamodule.eval_E, datamodule.man_info.embedding.unsqueeze(0), dim=1)
            sim_diff = sim_woman - sim_man
            sum_bias_sq = torch.sum(sim_diff ** 2)

            nn = torch.load(self.best_ckpt_path)
            QE = datamodule.eval_E @ nn.Q
            eval_E_comp = QE[:, 1:]

            # After transformation
            sim_woman_comp = F.cosine_similarity(eval_E_comp, datamodule.woman_info.embedding[1:].unsqueeze(0), dim=1)
            sim_man_comp = F.cosine_similarity(eval_E_comp, datamodule.man_info.embedding[1:].unsqueeze(0), dim=1)
            sim_diff_comp = sim_woman_comp - sim_man_comp
            sum_bias_sq_comp = torch.sum(sim_diff_comp ** 2)

        # Save similarities in csv:
        df_similarities = pd.DataFrame({
            "word": [prof.word for prof in datamodule.professions], 
            "Sim. to woman BEFORE": sim_woman.tolist(), 
            "Sim. to man BEFORE": sim_man.tolist(), 
            "diff BEFORE": sim_diff.tolist(), 
            "Sim. to woman AFTER": sim_woman_comp.tolist(), 
            "Sim. to man AFTER": sim_man_comp.tolist(), 
            "diff AFTER": sim_diff_comp.tolist()})
        df_similarities = df_similarities.round(4)
        df_similarities.to_csv("./results/simialrities.csv", index=False)
        
        logger.info(f"Before: {sum_bias_sq}\tAfter: {sum_bias_sq_comp}\tDiff: {sum_bias_sq_comp - sum_bias_sq}")