import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import PretrainDataset
from bert import BertModel
from tqdm import tqdm
from optimizer import AdamW


# import pandas as pd

# df_1 = pd.read_csv('data/ids-cfimdb-train.csv', sep='\t')
# df_2 = pd.read_csv('data/ids-sst-train.csv', sep='\t')
# df_3 = pd.read_csv('data/quora-train.csv', sep='\t')
# df_4 = pd.read_csv('data/sts-train.csv', sep='\t')


# s1 = df_1['sentence'].astype(str)
# s2 = df_2['sentence'].astype(str)
# s3 = df_3['sentence1'].astype(str)
# s4 = df_3['sentence2'].astype(str)
# s5 = df_4['sentence1'].astype(str)
# s6 = df_4['sentence2'].astype(str)

# all_sentences = pd.concat([s1, s2, s3, s4, s5, s6], ignore_index=True)
# all_sentences = all_sentences.drop_duplicates()

# all_sentences.to_csv('data/pretrain_texts.txt', index=False, header=False)

# print(f"提取完成！总共获得 {len(all_sentences)} 条独立句子。")

scaler = GradScaler('cuda')


device = torch.device('cuda')

mlm_head = torch.nn.Linear(768, 30522).to(device)
epoch_num = 2
lr = 1e-5


data = None
with open('data/pretrain_texts.txt', 'r') as fp:
    data = fp.readlines()

pretrain_data = PretrainDataset(data)

pretrain_dataloader = DataLoader(pretrain_data, batch_size = 32, shuffle = True, collate_fn=pretrain_data.collate_fn)
model = BertModel.from_pretrained('bert-base-uncased')
model.to(device)
optimizer = AdamW(list(model.parameters()) + list(mlm_head.parameters()), lr=lr)
mlm_head.weight = model.word_embedding.weight


for epoch in range(epoch_num):
    model.train()
    mlm_head.train()
    pbar = tqdm(pretrain_dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        optimizer.zero_grad()

        b_ids = batch['input_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        with autocast('cuda'):
            out = model(b_ids, b_mask)
            sequence_output = out['last_hidden_state']
            pred = mlm_head(sequence_output)
            loss = F.cross_entropy(pred.view(-1, 30522), b_labels.view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix(loss=f"{loss.item():.4f}")

torch.save(model.state_dict(), 'pretrain.pt')