#!/usr/bin/env python3

import argparse
import csv
import math
import os
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
from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data,
)
import evaluation as evaluation_module
from evaluation import model_eval_multitask, model_eval_test_multitask
from optimizer import AdamW


TQDM_DISABLE = False


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


class MultitaskBERT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            param.requires_grad = config.fine_tune_mode == "full-model"

        if config.use_pretrain and os.path.exists(config.pretrain_path):
            saved_state_dict = torch.load(config.pretrain_path, map_location="cpu")
            self.bert.load_state_dict(saved_state_dict)
            print(f"Loaded domain-specific pre-trained weights from {config.pretrain_path}")

        sent_dim = config.hidden_size * 2
        pair_dim = sent_dim * 4 + 1

        self.sent_norm = nn.LayerNorm(sent_dim)
        self.sentiment_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(sent_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 5),
        )
        self.paraphrase_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(pair_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )
        self.similarity_head = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(pair_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1),
        )

    def encode(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask)
        cls_embedding = bert_out["pooler_output"]
        last_hidden = bert_out["last_hidden_state"]
        mask = attention_mask.unsqueeze(-1).float()
        mean_embedding = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        sentence_embedding = self.sent_norm(torch.cat([cls_embedding, mean_embedding], dim=-1))

        return {
            "cls": cls_embedding,
            "mean": mean_embedding,
            "sentence_embedding": sentence_embedding,
        }

    def _pair_features(self, emb1, emb2):
        cosine = F.cosine_similarity(emb1, emb2, dim=-1).unsqueeze(-1)
        return torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2, cosine], dim=-1)

    def predict_sentiment(self, input_ids, attention_mask):
        embedding = self.encode(input_ids, attention_mask)["sentence_embedding"]
        return self.sentiment_head(embedding)

    def predict_paraphrase(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.encode(input_ids_1, attention_mask_1)["sentence_embedding"]
        emb2 = self.encode(input_ids_2, attention_mask_2)["sentence_embedding"]
        features = self._pair_features(emb1, emb2)
        return self.paraphrase_head(features)

    def predict_similarity(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        emb1 = self.encode(input_ids_1, attention_mask_1)["sentence_embedding"]
        emb2 = self.encode(input_ids_2, attention_mask_2)["sentence_embedding"]
        features = self._pair_features(emb1, emb2)
        return 5.0 * torch.sigmoid(self.similarity_head(features))


def save_model(model, optimizer, scheduler, args, config, filepath):
    save_info = {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": args,
        "model_config": config,
        "system_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
    }
    torch.save(save_info, filepath)
    print(f"Saved model to {filepath}")


def build_dataloaders(args):
    sst_train_data, num_labels, para_train_data, sts_train_data = load_multitask_data(
        args.sst_train, args.para_train, args.sts_train, split="train"
    )
    sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
        args.sst_dev, args.para_dev, args.sts_dev, split="train"
    )

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    train_loaders = {
        "sst": DataLoader(
            sst_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sst_train_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "para": DataLoader(
            para_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=para_train_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "sts": DataLoader(
            sts_train_data,
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=sts_train_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }

    dev_loaders = {
        "sst": DataLoader(
            sst_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sst_dev_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "para": DataLoader(
            para_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=para_dev_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
        "sts": DataLoader(
            sts_dev_data,
            shuffle=False,
            batch_size=args.batch_size,
            collate_fn=sts_dev_data.collate_fn,
            num_workers=args.num_workers,
            pin_memory=True,
        ),
    }

    return num_labels, train_loaders, dev_loaders


def _next_batch(loader_iters, loaders, task_name):
    iterator = loader_iters[task_name]
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loaders[task_name])
        loader_iters[task_name] = iterator
        batch = next(iterator)
    return batch


def _autocast_context(device):
    if device.type == "cuda":
        return autocast("cuda")
    return autocast("cpu", enabled=False)


def _compute_loss(model, batch, task_name, device, args):
    if task_name == "sst":
        ids = batch["token_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        logits = model.predict_sentiment(ids, mask)
        loss = F.cross_entropy(
            logits,
            labels.view(-1),
            label_smoothing=args.sst_label_smoothing,
        )
        return loss

    ids_1 = batch["token_ids_1"].to(device, non_blocking=True)
    mask_1 = batch["attention_mask_1"].to(device, non_blocking=True)
    ids_2 = batch["token_ids_2"].to(device, non_blocking=True)
    mask_2 = batch["attention_mask_2"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)

    if task_name == "para":
        logits = model.predict_paraphrase(ids_1, mask_1, ids_2, mask_2).view(-1)
        return F.binary_cross_entropy_with_logits(logits, labels.view(-1).float())

    preds = model.predict_similarity(ids_1, mask_1, ids_2, mask_2).view(-1)
    return F.smooth_l1_loss(preds, labels.view(-1).float(), beta=args.sts_beta)


def train_multitask(args):
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    _, train_loaders, dev_loaders = build_dataloaders(args)

    config = SimpleNamespace(
        hidden_dropout_prob=args.hidden_dropout_prob,
        hidden_size=768,
        fine_tune_mode=args.fine_tune_mode,
        use_pretrain=args.use_pretrain,
        pretrain_path=args.pretrain_path,
    )
    model = MultitaskBERT(config).to(device)

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
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = GradScaler("cuda", enabled=device.type == "cuda")

    task_names = ["sst", "para", "sts"]
    task_probs = np.array([args.sst_prob, args.para_prob, args.sts_prob], dtype=np.float64)
    task_probs = task_probs / task_probs.sum()
    task_weights = {"sst": args.sst_weight, "para": args.para_weight, "sts": args.sts_weight}

    best_dev_score = -float("inf")
    loader_iters = {name: iter(loader) for name, loader in train_loaders.items()}

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        task_loss_sum = {"sst": 0.0, "para": 0.0, "sts": 0.0}
        task_count = {"sst": 0, "para": 0, "sts": 0}

        progress = tqdm(
            range(args.steps_per_epoch),
            desc=f"train-{epoch}",
            disable=args.disable_tqdm,
        )
        for _ in progress:
            task_name = np.random.choice(task_names, p=task_probs)
            batch = _next_batch(loader_iters, train_loaders, task_name)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device):
                loss = _compute_loss(model, batch, task_name, device, args)
                weighted_loss = task_weights[task_name] * loss

            scaler.scale(weighted_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = float(weighted_loss.detach().cpu())
            epoch_loss += loss_value
            task_loss_sum[task_name] += loss_value
            task_count[task_name] += 1
            progress.set_postfix(
                task=task_name,
                loss=f"{loss_value:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        dev_sentiment_accuracy, _, _, dev_paraphrase_accuracy, _, _, dev_sts_corr, _, _ = model_eval_multitask(
            dev_loaders["sst"], dev_loaders["para"], dev_loaders["sts"], model, device
        )
        dev_score = (dev_sentiment_accuracy + dev_paraphrase_accuracy + dev_sts_corr) / 3.0

        avg_task_loss = {
            name: task_loss_sum[name] / max(1, task_count[name]) for name in task_names
        }
        print(
            f"Epoch {epoch}: avg loss {epoch_loss / args.steps_per_epoch:.4f}, "
            f"sst loss {avg_task_loss['sst']:.4f}, para loss {avg_task_loss['para']:.4f}, "
            f"sts loss {avg_task_loss['sts']:.4f}, dev score {dev_score:.4f}"
        )

        if dev_score > best_dev_score:
            best_dev_score = dev_score
            save_model(model, optimizer, scheduler, args, config, args.filepath)


def test_multitask(args):
    with torch.no_grad():
        device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
        saved = torch.load(args.filepath, map_location=device)
        config = saved["model_config"]

        model = MultitaskBERT(config)
        model.load_state_dict(saved["model"])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, _, para_test_data, sts_test_data = load_multitask_data(
            args.sst_test, args.para_test, args.sts_test, split="test"
        )
        sst_dev_data, _, para_dev_data, sts_dev_data = load_multitask_data(
            args.sst_dev, args.para_dev, args.sts_dev, split="dev"
        )

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)
        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)
        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sst_test_dataloader = DataLoader(
            sst_test_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_test_data.collate_fn
        )
        sst_dev_dataloader = DataLoader(
            sst_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sst_dev_data.collate_fn
        )
        para_test_dataloader = DataLoader(
            para_test_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_test_data.collate_fn
        )
        para_dev_dataloader = DataLoader(
            para_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=para_dev_data.collate_fn
        )
        sts_test_dataloader = DataLoader(
            sts_test_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_test_data.collate_fn
        )
        sts_dev_dataloader = DataLoader(
            sts_dev_data, shuffle=False, batch_size=args.batch_size, collate_fn=sts_dev_data.collate_fn
        )

        (
            dev_sentiment_accuracy,
            dev_sst_y_pred,
            dev_sst_sent_ids,
            dev_paraphrase_accuracy,
            dev_para_y_pred,
            dev_para_sent_ids,
            dev_sts_corr,
            dev_sts_y_pred,
            dev_sts_sent_ids,
        ) = model_eval_multitask(
            sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, device
        )

        (
            test_sst_y_pred,
            test_sst_sent_ids,
            test_para_y_pred,
            test_para_sent_ids,
            test_sts_y_pred,
            test_sts_sent_ids,
        ) = model_eval_test_multitask(
            sst_test_dataloader, para_test_dataloader, sts_test_dataloader, model, device
        )

        os.makedirs(os.path.dirname(args.sst_dev_out), exist_ok=True)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy:.3f}")
            f.write("id \t Predicted_Sentiment \n")
            for sent_id, pred in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{sent_id} , {pred} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write("id \t Predicted_Sentiment \n")
            for sent_id, pred in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{sent_id} , {pred} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy:.3f}")
            f.write("id \t Predicted_Is_Paraphrase \n")
            for sent_id, pred in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{sent_id} , {pred} \n")

        with open(args.para_test_out, "w+") as f:
            f.write("id \t Predicted_Is_Paraphrase \n")
            for sent_id, pred in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{sent_id} , {pred} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr:.3f}")
            f.write("id \t Predicted_Similarity \n")
            for sent_id, pred in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{sent_id} , {pred} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write("id \t Predicted_Similarity \n")
            for sent_id, pred in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{sent_id} , {pred} \n")


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
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fine_tune_mode", type=str, choices=("last-linear-layer", "full-model"), default="full-model")
    parser.add_argument("--use_gpu", action="store_true")

    parser.add_argument("--hidden_dropout_prob", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--head_lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--sst_label_smoothing", type=float, default=0.05)
    parser.add_argument("--sts_beta", type=float, default=0.5)
    parser.add_argument("--steps_per_epoch", type=int, default=3000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--disable_tqdm", action="store_true")

    parser.add_argument("--sst_prob", type=float, default=0.40)
    parser.add_argument("--para_prob", type=float, default=0.35)
    parser.add_argument("--sts_prob", type=float, default=0.25)
    parser.add_argument("--sst_weight", type=float, default=1.35)
    parser.add_argument("--para_weight", type=float, default=1.0)
    parser.add_argument("--sts_weight", type=float, default=1.15)

    parser.add_argument("--use_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default="pretrain.pt")
    parser.add_argument("--run_name", type=str, default="adv-multitask")

    parser.add_argument("--sst_dev_out", type=str, default="predictions/adv-sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/adv-sst-test-output.csv")
    parser.add_argument("--para_dev_out", type=str, default="predictions/adv-para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/adv-para-test-output.csv")
    parser.add_argument("--sts_dev_out", type=str, default="predictions/adv-sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/adv-sts-test-output.csv")

    return parser.parse_args()


def summarize_predictions(args):
    scores = {}
    label_paths = {
        "sst": (args.sst_dev_out, args.sst_dev, "sentiment"),
        "para": (args.para_dev_out, args.para_dev, "is_duplicate"),
        "sts": (args.sts_dev_out, args.sts_dev, "similarity"),
    }

    for task, (pred_path, gold_path, field) in label_paths.items():
        gold = {}
        with open(gold_path, "r") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                gold[row["id"].strip().lower()] = float(row[field]) if task == "sts" else int(float(row[field]))

        pred = {}
        with open(pred_path, "r") as f:
            next(f)
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) < 2:
                    parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
                if len(parts) >= 2:
                    pred[parts[0].lower()] = float(parts[1]) if task == "sts" else int(float(parts[1]))

        ids = sorted(set(gold) & set(pred))
        if task == "sts":
            y_true = np.array([gold[i] for i in ids])
            y_pred = np.array([pred[i] for i in ids])
            scores[task] = float(np.corrcoef(y_true, y_pred)[0, 1])
        else:
            y_true = np.array([gold[i] for i in ids])
            y_pred = np.array([pred[i] for i in ids])
            scores[task] = float(np.mean(y_true == y_pred))

    print(
        "Summary: "
        f"sst={scores['sst']:.4f}, para={scores['para']:.4f}, sts={scores['sts']:.4f}, "
        f"avg={(scores['sst'] + scores['para'] + scores['sts']) / 3.0:.4f}"
    )


if __name__ == "__main__":
    args = get_args()
    args.filepath = f"{args.run_name}.pt"
    seed_everything(args.seed)
    evaluation_module.TQDM_DISABLE = args.disable_tqdm
    train_multitask(args)
    test_multitask(args)
    summarize_predictions(args)
