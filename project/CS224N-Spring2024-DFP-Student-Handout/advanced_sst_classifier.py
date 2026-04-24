#!/usr/bin/env python3

import argparse
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from bert import BertModel
from classifier import (
    SentimentDataset,
    SentimentTestDataset,
    load_data,
    model_eval,
    model_test_eval,
)
from optimizer import AdamW


def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class AdvancedSSTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        for param in self.bert.parameters():
            param.requires_grad = config.fine_tune_mode == "full-model"

        if config.use_pretrain:
            state_dict = torch.load(config.pretrain_path, map_location="cpu")
            self.bert.load_state_dict(state_dict)
            print(f"Loaded domain-specific pre-trained weights from {config.pretrain_path}")

        sent_dim = config.hidden_size * 2
        self.norm = nn.LayerNorm(sent_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(sent_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, config.num_labels),
        )

    def encode(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask)
        cls_embedding = bert_out["pooler_output"]
        last_hidden = bert_out["last_hidden_state"]

        mask = attention_mask.unsqueeze(-1).float()
        mean_embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.norm(torch.cat([cls_embedding, mean_embedding], dim=-1))

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.encode(input_ids, attention_mask))


def save_model(model, optimizer, scheduler, args, config):
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": args,
            "model_config": config,
        },
        args.filepath,
    )
    print(f"Saved model to {args.filepath}")


def train(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    train_data, num_labels = load_data(args.train, "train")
    dev_data = load_data(args.dev, "valid")

    train_dataset = SentimentDataset(train_data, args)
    dev_dataset = SentimentDataset(dev_data, args)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        collate_fn=dev_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_labels=num_labels,
        hidden_size=768,
        fine_tune_mode=args.fine_tune_mode,
        use_pretrain=args.use_pretrain,
        pretrain_path=args.pretrain_path,
    )
    model = AdvancedSSTClassifier(config).to(device)

    head_params = []
    bert_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("bert."):
            bert_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if bert_params:
        param_groups.append({"params": bert_params, "lr": args.lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": args.head_lr})

    optimizer = AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    best_dev_acc = -float("inf")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_dataloader, desc=f"train-{epoch}", disable=args.disable_tqdm)
        for batch in progress:
            ids = batch["token_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=device.type == "cuda"):
                logits = model(ids, mask)
                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    label_smoothing=args.label_smoothing,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{float(loss):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        dev_acc, dev_f1, *_ = model_eval(dev_dataloader, model, device)
        print(
            f"Epoch {epoch}: train_loss={total_loss / len(train_dataloader):.4f}, "
            f"dev_acc={dev_acc:.4f}, dev_f1={dev_f1:.4f}"
        )

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, scheduler, args, config)


def test(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    saved = torch.load(args.filepath, map_location=device)
    config = saved["model_config"]
    model = AdvancedSSTClassifier(config)
    model.load_state_dict(saved["model"])
    model = model.to(device)
    print(f"Loaded model from {args.filepath}")

    dev_data = load_data(args.dev, "valid")
    test_data = load_data(args.test, "test")
    dev_dataset = SentimentDataset(dev_data, args)
    test_dataset = SentimentTestDataset(test_data, args)

    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)

    dev_acc, dev_f1, dev_pred, _, _, dev_sent_ids = model_eval(dev_dataloader, model, device)
    test_pred, _, test_sent_ids = model_test_eval(test_dataloader, model, device)
    print(f"dev acc :: {dev_acc:.3f}")
    print(f"dev f1 :: {dev_f1:.3f}")

    with open(args.dev_out, "w+") as f:
        f.write("id \t Predicted_Sentiment \n")
        for sent_id, pred in zip(dev_sent_ids, dev_pred):
            f.write(f"{sent_id} , {pred} \n")

    with open(args.test_out, "w+") as f:
        f.write("id \t Predicted_Sentiment \n")
        for sent_id, pred in zip(test_sent_ids, test_pred):
            f.write(f"{sent_id} , {pred} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fine_tune_mode", choices=("last-linear-layer", "full-model"), default="full-model")
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default="pretrain.pt")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--train", default="data/ids-sst-train.csv")
    parser.add_argument("--dev", default="data/ids-sst-dev.csv")
    parser.add_argument("--test", default="data/ids-sst-test-student.csv")
    parser.add_argument("--filepath", default="adv-sst.pt")
    parser.add_argument("--dev_out", default="predictions/adv-sst-only-dev-output.csv")
    parser.add_argument("--test_out", default="predictions/adv-sst-only-test-output.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)
    train(args)
    test(args)
