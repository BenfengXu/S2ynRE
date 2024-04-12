import argparse
import statistics
import pandas as pd
import csv
import shutil
import os


# ===== Set Your Timestamp =====
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--best_k", type=int, default=None, required=True)
parser.add_argument("--result_file", type=str, default=None, required=True)
parser.add_argument("--teacher_model_dir", type=str, default=None, required=True)
args = parser.parse_args()

def main():
    """main"""
    "DataFrame is formated as: [ckpt, bs, ep, lr, seed, dev, test]"
    df = pd.read_csv(args.result_file, header=None)
    df = df.sort_values(6, ascending=False)  # sort i.e., 6-th column
    df = df[args.best_k:]  # keep best 5 teachers
    for teacher in df[0].tolist():
        shutil.rmtree(os.path.join(args.teacher_model_dir, teacher))

if __name__ == "__main__":
    main()