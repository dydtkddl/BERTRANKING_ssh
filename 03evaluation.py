def normalize_concept(mention, concepts, descriptions, model, tokenizer, max_len=128):
    best_score = float('-inf')
    best_concept = None

    for concept, description in zip(concepts, descriptions):
        encoding = tokenizer.encode_plus(
            mention,
            description,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        score = outputs.logits[0][1].item()  # Positive class score
        if score > best_score:
            best_score = score
            best_concept = concept

    return best_concept

# 예제 실행
mention = "CNT"
concepts = df['concept'].tolist()
descriptions = df['description'].tolist()
best_concept = normalize_concept(mention, concepts, descriptions, model, tokenizer)
print(f"The best concept for mention '{mention}' is '{best_concept}'")
