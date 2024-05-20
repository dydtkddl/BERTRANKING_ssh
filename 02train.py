import pandas as pd
from transformers import BertTokenizerFast
import torch
from torch.utils.data import DataLoader, Dataset
import datetime

# BERT 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained('/matbert-base-cased', do_lower_case=False)
print(1)

# 데이터 로드 및 전처리
DATAPATH = "./drive/MyDrive/고급딥러닝 프로젝트/"
FILENAME = "CIDfoundedDF_des_syn_complete.xlsx"
df = pd.read_excel(DATAPATH + FILENAME)[["name", "desall", 'synall']]
df.columns = ["concept", "description", "synonyms"]

mentions = []
synonyms = {}
for i in range(len(df)):
    row = df.iloc[i]
    concept = row["concept"]
    synonymsList = row["synonyms"]
    mentions.extend(synonymsList)
    synonyms[concept] = synonymsList

concepts = df['concept'].tolist()
descriptions = df['description'].tolist()

# 데이터셋 클래스 정의
class ConceptDataset(Dataset):
    def __init__(self, mentions, concepts, descriptions, synonyms, tokenizer, max_len):
        self.mentions = mentions
        self.concepts = concepts
        self.descriptions = descriptions
        self.synonyms = synonyms
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.mentions) * len(self.concepts)

    def __getitem__(self, item):
        mention_idx = item // len(self.concepts)
        concept_idx = item % len(self.concepts)

        mention = str(self.mentions[mention_idx])
        description = str(self.descriptions[concept_idx])

        label = 1 if mention in self.synonyms[self.concepts[concept_idx]] else 0

        encoding = self.tokenizer.encode_plus(
            mention,
            description,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,  # token_type_ids 반환하도록 설정
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),  # token_type_ids 반환
            'label': torch.tensor(label, dtype=torch.long)
        }

# 데이터셋 및 데이터로더 생성
dataset = ConceptDataset(mentions, concepts, descriptions, synonyms, tokenizer, max_len=512)
print(1)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
print(1)

# 데이터 로드 및 출력
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    token_type_ids = batch['token_type_ids']
    labels = batch['label']

    print("Input IDs:", input_ids)
    print("Attention Mask:", attention_mask)
    print("Token Type IDs:", token_type_ids)
    print("Labels:", labels)
    break  # 첫 번째 배치만 출력하고 종료
