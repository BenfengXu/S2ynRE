# S2ynRE

Released code for our ACL23 paper: [S2ynRE: Two-stage Self-training with Synthetic data for Low-resource Relation Extraction](https://aclanthology.org/2023.acl-long.455/)

# Data

Training data can be found at `./data/`, while generated synthetic data (either by ChatGPT or by finetuned GPT2) can be downloaded here:

https://drive.google.com/file/d/1-oPgKi3DhnpuFNAxAgOhqe0eec-8DR-r/view?usp=sharing

Following the paper setting, you should sample 10,000 for 1% setting, and 100,000 for 10% or 100% setting.

# Run

```
bash ./script/run_gen_iterative.sh
```

Note that this is the end-to-end script that integrates multiple process, including 1) training GPT2; 2) using GPT2 to generate synthetic samples; 3) the iterative process of distilling and two-stage trainig.

Cite us as:

```
@inproceedings{xu-etal-2023-s2ynre,
    title = "{S}2yn{RE}: Two-stage Self-training with Synthetic data for Low-resource Relation Extraction",
    author = "Xu, Benfeng  and
      Wang, Quan  and
      Lyu, Yajuan  and
      Dai, Dai  and
      Zhang, Yongdong  and
      Mao, Zhendong",
    editor = "Rogers, Anna  and
      Boyd-Graber, Jordan  and
      Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.455",
    doi = "10.18653/v1/2023.acl-long.455",
    pages = "8186--8207",
}
```

```

```
