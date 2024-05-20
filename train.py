import pandas as pd
from transformers import BertTokenizerFast
import torch
from torch.utils.data import DataLoader, Dataset
import datetime

# BERT 토크나이저 로드
tokenizer = BertTokenizerFast.from_pretrained('./matbert-base-cased', do_lower_case=False)

# 데이터 로드 및 전처리
DATAPATH = "./data/"
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
# 배치 크기와 최대 토큰 길이를 줄임
dataset = ConceptDataset(mentions, concepts, descriptions, synonyms, tokenizer, max_len=256)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

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

from transformers import BertForSequenceClassification, AdamW

# BERT 모델 로드
model = BertForSequenceClassification.from_pretrained('./matbert-base-cased', num_labels=2)

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 학습 함수 정의
def train(model, dataloader, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 캐시 비우기
            torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")

# 모델 학습
train(model, dataloader, epochs=3, learning_rate=2e-5)

# 모델 가중치 저장
timestamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M_%S")
model_save_path = f"./model_{timestamp}.bin"
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# # 개념 정규화 함수 정의
# def normalize_concept(mention, concepts, descriptions, model, tokenizer, max_len=64):
#     best_score = float('-inf')
#     best_concept = None

#     for concept, description in zip(concepts, descriptions):
#         encoding = tokenizer.encode_plus(
#             mention,
#             description,
#             add_special_tokens=True,
#             max_length=max_len,
#             return_token_type_ids=True,  # token_type_ids 반환하도록 설정
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt',
#         )

#         input_ids = encoding['input_ids'].to(device)
#         attention_mask = encoding['attention_mask'].to(device)
#         token_type_ids = encoding['token_type_ids'].to(device)

#         with torch.no_grad():
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

#         score = outputs.logits[0][1].item()  # Positive class score
#         if score > best_score:
#             best_score = score
#             best_concept = concept

#     return best_concept

# # 예제 실행
# mention = "Titanium Dioxide"
# concepts = df['concept'].tolist()
# descriptions = df['description'].tolist()
# best_concept = normalize_concept(mention, concepts, descriptions, model, tokenizer)
# print(f"The best concept for mention '{mention}' is '{best_concept}'")
