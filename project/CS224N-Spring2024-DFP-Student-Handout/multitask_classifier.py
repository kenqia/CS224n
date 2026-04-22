'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace
import os
from itertools import cycle

from torch.amp import autocast, GradScaler
from torch.utils.data import Subset
# from pcgrad import PCGrad
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    纯 PyTorch 实现的带预热的线性衰减学习率调度器。
    完美平替 Hugging Face 的同名函数。
    """
    def lr_lambda(current_step: int):
        # 阶段 1：预热期 (Warmup) - 学习率从 0 线性爬升到 1.0 (也就是你设置的 1e-5)
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 阶段 2：衰减期 (Decay) - 学习率从 1.0 线性下降到 0
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        pretrain_path = 'pretrain.pt' 
        if os.path.exists(pretrain_path):
            # 加载 state_dict
            saved_state_dict = torch.load(pretrain_path, map_location='cpu')
            # 把权重塞进你的 self.bert 模块里
            self.bert.load_state_dict(saved_state_dict)
            print(f"Loaded domain-specific pre-trained weights from {pretrain_path}")

        self.sentiment_classes_num = 5
        self.sentiment_output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.sentiment_output_project = nn.Linear(config.hidden_size, self.sentiment_classes_num)

        self.paraphrase_output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.paraphrase_output_project = nn.Linear(config.hidden_size * 3, 1)

        # self.similarity_output_dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.similarity_output_project = nn.Linear(config.hidden_size * 3, 1)
        self.sim_scale = nn.Parameter(torch.tensor(2.5))
        self.sim_shift = nn.Parameter(torch.tensor(1.0))


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        out = self.bert(input_ids, attention_mask)
        return out

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        embed = self.forward(input_ids, attention_mask)
        pooler_output = embed['pooler_output']
        out = self.sentiment_output_project(self.sentiment_output_dropout(pooler_output))
        return out


    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        embed_1 = self.forward(input_ids_1, attention_mask_1)
        embed_2 = self.forward(input_ids_2, attention_mask_2)
        pooler_output_1 = embed_1['pooler_output']
        pooler_output_2 = embed_2['pooler_output']
        
        diff = torch.abs(pooler_output_1 - pooler_output_2)

        out = torch.cat([pooler_output_1, pooler_output_2, diff], dim=-1)
        out = self.paraphrase_output_project(self.paraphrase_output_dropout(out))
        return out

    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        embed_1 = self.forward(input_ids_1, attention_mask_1)['pooler_output']
        embed_2 = self.forward(input_ids_2, attention_mask_2)['pooler_output']

        sim = F.cosine_similarity(embed_1, embed_2, dim= -1)
        out = self.sim_scale * (sim + self.sim_shift)
        return out.view(-1, 1)

        # diff = torch.abs(embed_1 - embed_2)

        # out = torch.cat([embed_1, embed_2, diff], dim=-1)
        # out = self.similarity_output_project(self.similarity_output_dropout(out))
        # return out



def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    DEBUG_MODE = False # 开启你的急速测试模式


    if DEBUG_MODE:
        sst_train_data = Subset(sst_train_data, range(min(1000, len(sst_train_data))))
        para_train_data = Subset(para_train_data, range(min(1000, len(para_train_data))))
        sts_train_data = Subset(sts_train_data, range(min(1000, len(sts_train_data))))
        
        sst_dev_data = Subset(sst_dev_data, range(min(500, len(sst_dev_data))))
        para_dev_data = Subset(para_dev_data, range(min(500, len(para_dev_data))))
        sts_dev_data = Subset(sts_dev_data, range(min(500, len(sts_dev_data))))
        
        sst_train_collate_fn = sst_train_data.dataset.collate_fn
        para_train_collate_fn = para_train_data.dataset.collate_fn
        sts_train_collate_fn = sts_train_data.dataset.collate_fn

        sst_dev_collate_fn = sst_dev_data.dataset.collate_fn
        para_dev_collate_fn = para_dev_data.dataset.collate_fn
        sts_dev_collate_fn = sts_dev_data.dataset.collate_fn
    else:
        sst_train_collate_fn = sst_train_data.collate_fn
        para_train_collate_fn = para_train_data.collate_fn
        sts_train_collate_fn = sts_train_data.collate_fn

        sst_dev_collate_fn = sst_dev_data.collate_fn
        para_dev_collate_fn = para_dev_data.collate_fn
        sts_dev_collate_fn = sts_dev_data.collate_fn

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_collate_fn,
                                      num_workers=4, pin_memory=True)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_collate_fn,
                                    num_workers=4, pin_memory=True)



    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_collate_fn,
                                      num_workers=4, pin_memory=True)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_collate_fn,
                                    num_workers=4, pin_memory=True)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=para_train_collate_fn,
                                      num_workers=4, pin_memory=True)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=para_dev_collate_fn,
                                    num_workers=4, pin_memory=True)
    sst_iter = cycle(sst_train_dataloader)
    sts_iter = cycle(sts_train_dataloader)
    para_iter = cycle(para_train_dataloader)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': 768,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    # optimizer = PCGrad(optimizer)
    best_dev_acc = 0
    min_loss = float('inf')

    temperature = 2.0
    sst_p = len(sst_train_data)**(1.0/temperature)
    sts_p = len(sts_train_data)**(1.0/temperature)
    para_p = len(para_train_data)**(1.0/temperature)
    sum_p = sst_p + para_p + sts_p
    probs = [sst_p / sum_p, para_p / sum_p, sts_p / sum_p]
    scaler = GradScaler('cuda')
    loss_num = 3
    
    batch_num = len(sst_train_dataloader) + len(para_train_dataloader) + len(sts_train_dataloader) 
    total_steps = batch_num * args.epochs // 3  # 你真实的总更新步数
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)
    
    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for step in tqdm(range(batch_num // loss_num), desc=f'train-{epoch}', disable=TQDM_DISABLE):
        # for batch_sst in tqdm(sst_train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            optimizer.zero_grad()
            

            task_idx = [np.random.choice([0, 1, 2], p=probs) for i in range(loss_num)]
            
            losses = [None, None, None]
            with autocast('cuda'):
                for idx in task_idx:
                    loss = None
                    if idx == 0:
                        batch_sst = next(sst_iter)
                        b_sst_ids, b_sst_mask, b_sst_labels = (batch_sst['token_ids'],
                           batch_sst['attention_mask'], batch_sst['labels'])
                        b_sst_ids = b_sst_ids.to(device)
                        b_sst_mask = b_sst_mask.to(device)
                        b_sst_labels = b_sst_labels.to(device)
                        logits_sst = model.predict_sentiment(b_sst_ids, b_sst_mask)
                        loss = F.cross_entropy(logits_sst, b_sst_labels.view(-1), reduction='sum') / args.batch_size
                    elif idx == 1:
                        batch_para = next(para_iter)
                        b_para_ids_1, b_para_mask_1, b_para_ids_2, b_para_mask_2, b_para_labels = \
                                                    (batch_para['token_ids_1'], batch_para['attention_mask_1'],
                                                    batch_para['token_ids_2'], batch_para['attention_mask_2'],
                                                    batch_para['labels'])
                        b_para_ids_1 = b_para_ids_1.to(device)
                        b_para_ids_2 = b_para_ids_2.to(device)
                        b_para_mask_1 = b_para_mask_1.to(device)
                        b_para_mask_2 = b_para_mask_2.to(device)
                        b_para_labels = b_para_labels.to(device)
                        logits_para = model.predict_paraphrase(b_para_ids_1, b_para_mask_1, b_para_ids_2, b_para_mask_2)
                        loss = F.binary_cross_entropy_with_logits(logits_para.view(-1), b_para_labels.view(-1).float(), reduction='sum') / args.batch_size
                    else:
                        batch_sts = next(sts_iter)
            
                        b_sts_ids_1, b_sts_mask_1, b_sts_ids_2, b_sts_mask_2, b_sts_labels = \
                                                    (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                                                    batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                                                    batch_sts['labels'])
                        b_sts_ids_1 = b_sts_ids_1.to(device)
                        b_sts_ids_2 = b_sts_ids_2.to(device)
                        b_sts_mask_1 = b_sts_mask_1.to(device)
                        b_sts_mask_2 = b_sts_mask_2.to(device)
                        b_sts_labels = b_sts_labels.to(device)
                        logits_sts = model.predict_similarity(b_sts_ids_1, b_sts_mask_1, b_sts_ids_2, b_sts_mask_2) 
                        loss = F.mse_loss(logits_sts.view(-1), b_sts_labels.view(-1).float(), reduction='sum') / args.batch_size
                if losses[idx] is None:
                    losses[idx] = loss
                else:
                    losses[idx] += loss

            valid_losses = [l for l in losses if l is not None]
            total_loss = sum(valid_losses)

            asymmetric_pcgrad_step(model, optimizer, losses, scaler, scheduler)

            train_loss += total_loss.item()
            num_batches += loss_num

        train_loss = train_loss / (num_batches)

        # train_sentiment_accuracy,train_sst_y_pred, train_sst_sent_ids, \
            # train_paraphrase_accuracy, train_para_y_pred, train_para_sent_ids, \
            # train_sts_corr, train_sts_y_pred, train_sts_sent_ids = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, device)
        # dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            # dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            # dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device)
# 
        # train_acc = (train_sentiment_accuracy + train_paraphrase_accuracy + train_sts_corr) / 3.0
        # dev_acc = (dev_sentiment_accuracy + dev_paraphrase_accuracy + dev_sts_corr) / 3.0

        # if dev_acc > best_dev_acc:
        #     best_dev_acc = dev_acc
        #     save_model(model, optimizer, args, config, args.filepath)
        if train_loss < min_loss:
            min_loss = train_loss
            save_model(model, optimizer, args, config, args.filepath)


        # print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc:.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}")


def pcgrad_step(model, optimizer, losses, scaler):
    '''
    losses : [loss_sst, loss_para, loss_sts] 的列表
    '''

    random.shuffle(losses)

    task_gradients = []
    for idx, loss in enumerate(losses):
        optimizer.zero_grad()

        is_last = (idx == len(losses) - 1)
        scaler.scale(loss).backward(retain_graph=not is_last)

        grads = []
        for param in model.parameters():
            if not param.requires_grad:
                continue

            if param.grad is not None:
                grads.append(param.grad.detach().clone().flatten())
            else:
                grads.append(torch.zeros_like(param).flatten())
        task_gradients.append(torch.cat(grads))
    
    num_tasks = len(task_gradients)
    final_gradients = [tg.clone() for tg in task_gradients]

    for i in range(num_tasks):
        task_indices = list(range(num_tasks))
        random.shuffle(task_indices)
        
        for j in task_indices:
            if i == j:
                continue
            
            dot_product = torch.dot(final_gradients[i], task_gradients[j])
            if dot_product < 0:
                gj_norm_sq = torch.norm(task_gradients[j])**2 + 1e-8
                alpha_val = -(dot_product / gj_norm_sq).item()
                final_gradients[i].add_(task_gradients[j], alpha=alpha_val)
    
    merged_gradient = torch.stack(final_gradients).sum(dim=0)

    pointer = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue

        num_param = param.numel()
        param.grad = merged_gradient[pointer:pointer+num_param].view_as(param).clone()
        pointer += num_param
    
    scaler.step(optimizer)
    scaler.update()
    
def asymmetric_pcgrad_step(model, optimizer, losses, scaler, scheduler):
    '''
    定向保护 SST 的 PCGrad 算法
    ⚠️ 极其重要：传入的 losses 列表必须严格遵守顺序 [loss_sst, loss_para, loss_sts] 
    '''
    # 解包获取对应的 loss (顺序绝对不能错)
    loss_sst, loss_para, loss_sts = losses

    task_gradients = []

    last_valid_idx = -1
    for i in range(len(losses) - 1, -1, -1):
        if losses[i] is not None:
            last_valid_idx = i
            break
    
    # 依次求导，并收集 1D 梯度向量
    for idx, loss in enumerate(losses):
        if loss is None:
            task_gradients.append(None)
            continue
        optimizer.zero_grad()
        is_last = (idx == last_valid_idx)
        scaler.scale(loss).backward(retain_graph=not is_last)

        grads = []
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is not None:
                grads.append(param.grad.detach().clone().flatten())
            else:
                grads.append(torch.zeros_like(param).flatten())
        task_gradients.append(torch.cat(grads))

    # 提取各个任务的超级向量
    g_sst = task_gradients[0]
    g_para = task_gradients[1]
    g_sts = task_gradients[2]

    # ==========================================
    # 🔪 定向手术室 (Asymmetric Gradient Surgery)
    # ==========================================

    if g_sst is not None:

        # 1. 检查 Para 是否欺负了 SST
        if g_para is not None:
            dot_para_sst = torch.dot(g_para, g_sst)
            if dot_para_sst < 0:
                # 如果冲突，削弱 Para 中反向于 SST 的分量
                sst_norm_sq = torch.norm(g_sst)**2 + 1e-8
                alpha_para = -(dot_para_sst / sst_norm_sq).item()
                # 等价于 g_para -= (dot / norm) * g_sst，但 add_ 更省显存
                g_para.add_(g_sst, alpha=alpha_para)

        if g_sts is not None:
            # 2. 检查 STS 是否欺负了 SST
            dot_sts_sst = torch.dot(g_sts, g_sst)
            if dot_sts_sst < 0:
                # 如果冲突，削弱 STS 中反向于 SST 的分量
                sst_norm_sq = torch.norm(g_sst)**2 + 1e-8
                alpha_sts = -(dot_sts_sst / sst_norm_sq).item()
                g_sts.add_(g_sst, alpha=alpha_sts)

    # 注意：SST 受到绝对保护，无论它是否和别人冲突，我们都不修改 g_sst！
    # 也不对 g_para 和 g_sts 进行互相干预。

    # ==========================================
    # 3. 最终梯度组合与还原
    # ==========================================
    valid_grads = [g for g in [g_sst, g_para, g_sts] if g is not None]
    merged_gradient = torch.stack(valid_grads).sum(dim=0)

    pointer = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        num_param = param.numel()
        param.grad = merged_gradient[pointer:pointer+num_param].view_as(param).clone()
        pointer += num_param

    # 统一更新参数
    scaler.step(optimizer)
    scheduler.step()
    scaler.update()

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)
