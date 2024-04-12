import argparse
import statistics
import pandas as pd
import csv
import re

# ===== Set Your Timestamp =====
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--input_file", type=str, default=None, required=True)
args = parser.parse_args()

def main():
    """main"""
    processed_samples = []
    with open(args.input_file, 'r') as f:
        for line in f.readlines():
            raw_sample = line.strip()
            raw_sample = raw_sample.replace('<|endoftext|>', '')
            token = None
            h = None
            t = None
            h_start = [m.start() for m in re.finditer('\[unused0\]', raw_sample)]
            h_end = [m.start() for m in re.finditer('\[unused1\]', raw_sample)]
            t_start = [m.start() for m in re.finditer('\[unused2\]', raw_sample)]
            t_end = [m.start() for m in re.finditer('\[unused3\]', raw_sample)]
            len_marker = len('[unused0]')
            if len(h_start) == len(h_end) == len(t_start) == len(t_end) == 1:
                h_start = h_start[0]
                h_end = h_end[0]
                t_start = t_start[0]
                t_end = t_end[0]
                if h_start < t_start:
                    tokens = raw_sample[:h_start].split(' ')
                    h_start_tok = len(tokens)
                    tokens += [raw_sample[h_start:h_end]]
                    h_end_tok = len(tokens)
                    h_name = raw_sample[h_start:h_end]
                    tokens += raw_sample[h_end:t_start].split(' ')
                    t_start_tok = len(tokens)
                    tokens += [raw_sample[t_start:t_end]]
                    t_end_tok = len(tokens)
                    tokens += raw_sample[t_end:].split(' ')
                else:
                    tokens = raw_sample[:h_start].split(' ')
                    h_start_tok = len(tokens)
                    tokens += [raw_sample[h_start:h_end]]
                    h_end_tok = len(tokens)
                    tokens += raw_sample[h_end:t_start].split(' ')
                    t_start_tok = len(tokens)
                    tokens += [raw_sample[t_start:t_end]]
                    t_end_tok = len(tokens)
                    tokens += raw_sample[t_end:].split(' ')


if __name__ == "__main__":
    main()