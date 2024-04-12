import os
from pathlib import Path
import json
import re
import numpy as np

def Acc_eval(predictions, labels):
    return round(float(100 * (labels == predictions).sum() / len(labels)), 2)


def Semeval_eval(predictions, labels, dataset_file_dir):
    id2rel = json.load(open(Path(dataset_file_dir).joinpath("id2rel.json")))

    proposed_answer = os.path.join(dataset_file_dir, 'proposed_answer_tmp.txt')
    with open(proposed_answer, 'w') as f:
        for idx, pred in enumerate(predictions):
            f.write("{}\t{}".format(idx, id2rel[pred]))
            f.write('\n')

    answer_key = os.path.join(dataset_file_dir, 'answer_key_tmp.txt')
    with open(answer_key, 'w') as f:
        for idx, ins in enumerate(labels):
            f.write("{}\t{}".format(idx, id2rel[labels[idx]]))
            f.write('\n')

    official_eval_script = os.path.join(dataset_file_dir, 'semeval2010_task8_scorer-v1.2.pl')
    r = os.popen('perl {} {} {}'.format(official_eval_script, proposed_answer, answer_key))
    result = r.read()
    r.close()
    # example of result
    # 'P =  339/ 860 =  39.42%     R =  339/1229 =  27.58%     F1 =  32.46'
    result = re.search("Micro-averaged result \(excluding Other\):\n(.*?)\n\n", result).group(0).lstrip("Micro-averaged result (excluding Other):\n").rstrip("\n\n").split()
    result = [span for span in result if span.find('%') != -1]
    micro_precision = float(result[0][:-1])
    micro_recall = float(result[1][:-1])
    micro_f1 = float(result[2][:-1])
    return {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1}


def TACRED_eval(predictions, labels, num_labels):
    predictions = predictions.numpy()
    labels = labels.numpy()
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0
    for i in range(num_labels):
        if i == 0:
            continue
        tp += np.sum((labels == i) & (predictions == i))
        tn += np.sum((labels != i) & (predictions != i))
        fn += np.sum((labels != i) & (predictions == i))
        fp += np.sum((labels == i) & (predictions != i))
        total += 1
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = (2 * p * r) / (p + r + 1e-8)
    return {'precision': round(float(p * 100), 2), 'recall': round(float(r * 100), 2), 'f1': round(float(f1 * 100), 2)}
