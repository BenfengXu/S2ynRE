import argparse
import statistics
import pandas as pd
import csv
import shutil
import os


# ===== Set Your Timestamp =====
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--result_file", type=str, default=None, required=True)
parser.add_argument("--iteration", type=int, default=None, required=True)
parser.add_argument("--teacher_model_dir", type=str, default=None, required=True)
args = parser.parse_args()

def main():
    """main"""
    "DataFrame is formated as: [ckpt, bs, ep, lr, seed, dev, test]"
    df = pd.read_csv(args.result_file)
    for metric in ['dev_f1', 'dev_acc']:
        if metric in df.keys():
            break
    df = df[df['iteration'] == args.iteration]
    seeds = set(df['seed'])
    for seed in seeds:
        df_seed_tmp = df[df['seed'] == seed]
        df_seed_tmp = df_seed_tmp.sort_values(metric, ascending=False)  # sort i.e., 6-th column
        df_seed_tmp = df_seed_tmp[1:]  # keep best teacher per seed
        for teacher in df_seed_tmp['ckpt'].tolist():
            shutil.rmtree(os.path.join(args.teacher_model_dir, teacher))

if __name__ == "__main__":
    main()