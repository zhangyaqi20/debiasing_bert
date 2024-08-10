import logging
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from DebiasingDataModule import DebiasingDataModule
from DebiasingNN import DebiasingNN
from DebiasingTrainer import DebiasingTrainer

log_file = "run.log"
logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
logger = logging.getLogger(__name__)

# DataModule
TOKENIZER_URL = "bert-base-uncased"
MAX_TOKEN_LEN = 128
BATCH_SIZE = 64
# Model
MODEL_URL = "bert-base-uncased"
LEARNING_RATE = 1e-2 # baseline: 5e-5
WEIGHT_DECAY = 0.01 # baseline: 0.01
# Trainer
NUM_EPOCH = 4
NUM_SANITY_VAL_STEPS = 0
ACCELERATOR = "gpu"
DEVICES = [2]

def main():
    dm = DebiasingDataModule()
    dm.prepare_data()
    nn = DebiasingNN(
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    trainer = DebiasingTrainer(
         max_epochs=NUM_EPOCH,
         num_sanity_val_steps=NUM_SANITY_VAL_STEPS,
         accelerator=ACCELERATOR,
         devices=DEVICES,
         deterministic=True,
         )

    dm.setup(stage="fit")
    # trainer.fit(nn, (dm.train_E_female, dm.train_E_male))
    
    dm.setup(stage="test")
    trainer.test(dm)

if __name__ == "__main__":
    main()