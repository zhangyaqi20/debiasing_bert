import json
import logging
import os
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class DebiasingDataset(Dataset):
    def __init__(self, data, data_entry, tokenizer, max_token_len=128) -> None:
        self.data = data
        self.data_entry = data_entry
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        text = self.data[index][self.data_entry]
        if text is None:
            return {"input_ids": torch.tensor([101, 102]+[0]*(self.max_token_len-2)),
                    "attention_mask": torch.tensor([1,1]+[0]*(self.max_token_len-2))}
        
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            return_tensors="pt"
        )
        return {"input_ids": encoded_text["input_ids"].flatten(),
                "attention_mask": encoded_text["attention_mask"].flatten()}

class DebiasingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, model_url = "bert-base-uncased",):
        super().__init__()
        self.batch_size = batch_size
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.bert = AutoModel.from_pretrained(model_url)
        self.def_pairs = None
        self.woman_info = None
        self.man_info = None
        self.professions = None

    def prepare_data(self):
        data_name = "bbcnews"
        data_subset = "2024-05"
        bbcnews_data = load_dataset("RealTimeData/bbc_news_alltime", data_subset, split="train")
        data_entry = "content"
        self.bbcnews_dataset = DebiasingDataset(bbcnews_data, data_entry, self.tokenizer)

        # Get BERT embeddings of the dataset
        # torch.cuda.empty_cache()
        bbcnews_E_filename = f'./data/context_embeddings/{data_name}-{data_subset}-{data_entry}_embeddings.pt'
        device = torch.device("cuda:2")
        if os.path.isfile(bbcnews_E_filename):
            self.bbcnews_E = torch.load(bbcnews_E_filename, map_location=device)
        else:
            bbcnews_dataloader = DataLoader(self.bbcnews_dataset, batch_size=self.batch_size, shuffle=False)
            dl_iter = iter(bbcnews_dataloader)
            self.bert = self.bert.to(device)
            with torch.no_grad():
                self.bbcnews_E = torch.Tensor().to(device) # size: N x max_token_len x 768
                for batch in tqdm(dl_iter):
                    batch = {key: val.to(device) for key, val in batch.items()}
                    E_batch = self.bert(**batch)["last_hidden_state"]
                    self.bbcnews_E = torch.cat((self.bbcnews_E, E_batch), 0)
            torch.save(self.bbcnews_E, bbcnews_E_filename)
        self.bbcnews_E = self.bbcnews_E.to("cpu")
        assert self.bbcnews_E.shape == (len(bbcnews_data), 128, 768)
        logger.info(f"Get embeddings of context data: {self.bbcnews_E.shape}")

        logger.info("Data Preparation Finished.")

    def setup(self, stage=None):
        if stage == "fit":
            with open('./data/definitional_pairs.json') as f:
                def_pairs = json.load(f)
            pairs = []
            for i, (f_word, m_word) in enumerate(def_pairs):
                f_token_id = self.tokenizer.convert_tokens_to_ids(f_word.lower())
                m_token_id = self.tokenizer.convert_tokens_to_ids(m_word.lower())
                p = WordPair(pair_index=i, 
                        f_word=f_word, f_token_id=f_token_id,
                        m_word=m_word, m_token_id=m_token_id)
                for context_idx in range(len(self.bbcnews_dataset)):
                    p.add_context(self.bbcnews_dataset[context_idx], context_idx)
                if p.isValid(): # filter [UNK] word and words without contexts
                    pairs.append(p)
                    p.construct_embedding(self.bbcnews_E)
                    if f_word == "woman" and m_word == "man":
                        self.woman_info = p.female_word
                        self.man_info = p.male_word
            self.def_pairs = pairs
            self.train_E_female = torch.stack([p.female_word.embedding for p in pairs])
            self.train_E_male = torch.stack([p.male_word.embedding for p in pairs])
            assert self.train_E_female.shape == (len(pairs), self.bbcnews_E.shape[2])
            assert self.train_E_male.shape == (len(pairs), self.bbcnews_E.shape[2])
            logger.info("Train Data Setup Finished.")
        else:
            with open('./data/professions.json') as f:
                professions = json.load(f)
            eval_words = []
            for prof in professions:
                token = prof[0]
                token_id = self.tokenizer.convert_tokens_to_ids(token.lower())
                if token_id == 100: # UNK word
                    continue
                word = Word(token, token_id)
                for context_idx in range(len(self.bbcnews_dataset)):
                    word.add_context(self.bbcnews_dataset[context_idx], context_idx)
                if word.isValid(): # filter [UNK] word and words without contexts
                    eval_words.append(word)
                    word.construct_embedding(self.bbcnews_E)
            self.professions = eval_words
            self.eval_E = torch.stack([word.embedding for word in eval_words])
            assert self.eval_E.shape == (len(eval_words), self.bbcnews_E.shape[2])
            logger.info("Eval Data Setup Finished.")

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

class Word():
    def __init__(self, word, token_id):
        self.word = word
        self.token_id = token_id
        self.context = {} # map from context index in the dataset to the index of word in the context sentence
        self.context_len = None
        self.embedding = None
    
    def add_context(self, context, context_idx):
        """context: tokenized encodings, {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        """
        try:
            token_idx = list(context["input_ids"]).index(self.token_id)
            self.context[context_idx] = token_idx
        except ValueError:
            pass
    
    def construct_embedding(self, data_E):
        self.embedding = torch.mean(data_E[torch.tensor(list(self.context.keys())), torch.tensor(list(self.context.values())), :], dim=0)
    
    def isValid(self):
        self._count_context()
        return self.token_id != 100 and self.context_len > 0
    
    def _count_context(self):
        self.context_len = len(self.context)
    
    def __str__(self) -> str:
        self._count_context()
        return self.word + " (id=" + str(self.token_id) + " | context_size = " + str(self.context_len) + ") "

class WordPair():
    def __init__(self, pair_index, f_word, f_token_id, m_word, m_token_id):
        self.pair_index = pair_index
        self.female_word = Word(f_word, f_token_id)
        self.male_word = Word(m_word, m_token_id)
    
    def add_context(self, context, context_idx):
        """context: tokenized encodings, {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        """
        self.female_word.add_context(context, context_idx)
        self.male_word.add_context(context, context_idx)
    
    def construct_embedding(self, data_E):
        self.female_word.construct_embedding(data_E)
        self.male_word.construct_embedding(data_E)

    def isValid(self):
        return self.female_word.isValid() and self.male_word.isValid()
    
    def __str__(self) -> str:
        return str(self.female_word) + " - " + str(self.male_word)


dm = DebiasingDataModule()
dm.prepare_data()