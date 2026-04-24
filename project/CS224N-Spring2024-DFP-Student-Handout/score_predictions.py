#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


def load_gold(path, task):
    gold = {}
    with open(path, "r") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            row_id = row["id"].strip().lower()
            if task == "sst":
                gold[row_id] = int(row["sentiment"])
            elif task == "para":
                gold[row_id] = int(float(row["is_duplicate"]))
            else:
                gold[row_id] = float(row["similarity"])
    return gold


def load_pred(path, task):
    pred = {}
    with open(path, "r") as f:
        next(f)
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                parts = [p.strip() for p in line.strip().split("\t") if p.strip()]
            if len(parts) >= 2:
                row_id = parts[0].lower()
                pred[row_id] = float(parts[1]) if task == "sts" else int(float(parts[1]))
    return pred


def score(task, gold_path, pred_path):
    gold = load_gold(gold_path, task)
    pred = load_pred(pred_path, task)
    ids = sorted(set(gold) & set(pred))
    y_true = np.array([gold[i] for i in ids])
    y_pred = np.array([pred[i] for i in ids])
    if task == "sts":
        return float(np.corrcoef(y_true, y_pred)[0, 1])
    return float(np.mean(y_true == y_pred))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_gold", default="data/ids-sst-dev.csv")
    parser.add_argument("--para_gold", default="data/quora-dev.csv")
    parser.add_argument("--sts_gold", default="data/sts-dev.csv")
    parser.add_argument("--sst_pred", required=True)
    parser.add_argument("--para_pred", required=True)
    parser.add_argument("--sts_pred", required=True)
    args = parser.parse_args()

    scores = {
        "sst": score("sst", args.sst_gold, args.sst_pred),
        "para": score("para", args.para_gold, args.para_pred),
        "sts": score("sts", args.sts_gold, args.sts_pred),
    }
    avg = (scores["sst"] + scores["para"] + scores["sts"]) / 3.0

    print(f"sst={scores['sst']:.6f}")
    print(f"para={scores['para']:.6f}")
    print(f"sts={scores['sts']:.6f}")
    print(f"avg={avg:.6f}")


if __name__ == "__main__":
    main()
