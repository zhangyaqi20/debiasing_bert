from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
import json
from collections import defaultdict
import torch

class DebiasingDataset(Dataset):
    def __init__(self, data, tokenizer, max_token_len=128) -> None:
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        text = self.data[index]["description"]
        if text is None:
            return {"input_ids": torch.tensor([101, 102]+[0]*(self.max_token_len-2)),
                    "attention_mask": torch.tensor([1,1]+[0]*(self.max_token_len-2)),
                    "token_type_ids": torch.tensor([0,0]+[0]*(self.max_token_len-2))}
        encoded_text = self.tokenizer(
            text,
            padding="max_length",
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_token_len,
            return_tensors="pt"
        )
        return {"input_ids": encoded_text["input_ids"].flatten(),
                "attention_mask": encoded_text["attention_mask"].flatten(),
                "token_type_ids": encoded_text["token_type_ids"].flatten()}

class DebiasingDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, model_url = "bert-base-uncased", ):
        super().__init__()
        self.batch_size = batch_size
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_url)
        self.bert = AutoModel.from_pretrained(model_url)

    def prepare_data(self):
        bbcnews_data = load_dataset("RealTimeData/bbc_news_alltime", "2024-03", split="train")
        bbcnews_dataset = DebiasingDataset(bbcnews_data, self.tokenizer)
        bbcnews_dataloader = DataLoader(bbcnews_dataset, batch_size=self.batch_size, shuffle=False)
        dl_iter = iter(bbcnews_dataloader)
        with torch.no_grad():
            bbcnews_E = torch.cat([self.bert(**batch)["last_hidden_state"] for batch in dl_iter], 0)
        print(bbcnews_E.shape)
        torch.save(bbcnews_E, 'bbcnews-2024-03_embeddings.pt')
        
        # with open('../data/definitional_pairs.json') as f:
        #     def_pairs = json.load(f)
        # pairs = []
        # for i, (f_word, m_word) in enumerate(def_pairs):
        #     f_token_id = self.tokenizer.convert_tokens_to_ids(f_word.lower())
        #     m_token_id = self.tokenizer.convert_tokens_to_ids(m_word.lower())
        #     pairs.append(
        #         Pair(pair_index=i, 
        #              f_word=f_word, f_token_id=f_token_id,
        #              m_word=m_word, m_token_id=m_token_id))
        # for p in pairs:
        #     for context_idx in range(len(bbcnews_dataset)):
        #         p.add_context(bbcnews_dataset[context_idx], context_idx)
        #     print(p)
        


        # E_x = self.bert(**x).T # dim x N
        # E_y = self.bert(**y).T # dim x N

    def setup(self, stage=None):
        # 在这里可以进行数据集的准备
        # train, val, test datasets 可以在这里按需初始化
        pass

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, batch_size=self.batch_size)

class Pair():
    def __init__(self, pair_index, f_word, f_token_id, m_word, m_token_id):
        self.pair_index = pair_index
        self.f_word = f_word
        self.f_token_id = f_token_id
        self.m_word = m_word
        self.m_token_id = m_token_id
        self.f_context = {} # map from context index in the dataset to the index of word in the context sentence
        self.m_context = {}
        self.f_context_len = None
        self.m_context_len = None
    
    def contains(self, token_id):
        if token_id == self.f_token_id or token_id == self.m_token_id:
            return self.pair_index
        else:
            return -1
    
    def add_context(self, context, context_idx):
        """context: tokenized encodings, {"input_ids": [], "attention_mask": [], "token_type_ids": []}
        """
        try:
            token_idx = list(context["input_ids"]).index(self.f_token_id)
            self.f_context[context_idx] = token_idx
        except ValueError:
            pass
            
        try:
            token_idx = list(context["input_ids"]).index(self.m_token_id)
            self.m_context[context_idx] = token_idx
        except ValueError:
            pass

    def _count_context(self):
        self.f_context_len = len(self.f_context)
        self.m_context_len = len(self.m_context)
    
    def __str__(self) -> str:
        self._count_context()
        return self.f_word + "(id=" + str(self.f_token_id) + " | context_size = " + str(self.f_context_len) + ") - " + self.m_word +  "(id=" + str(self.m_token_id) + " | context_size = " + str(self.m_context_len) + ")"


dm = DebiasingDataModule()
dm.prepare_data()