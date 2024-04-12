# S2ynRE
Released code for our ACL23 paper, please stay tuned!

All synthetic data used in the paper can be downloaded here:


# Data

Training data can be found at `./data/`, while generated synthetic data (either by ChatGPT or by finetuned GPT2) can be downloaded here:

https://drive.google.com/file/d/1-oPgKi3DhnpuFNAxAgOhqe0eec-8DR-r/view?usp=sharing

Following the paper setting, you should use 10,000 for 1% setting, and 100,000 for 10% or 100% setting.

# Run

```
bash ./script/run_gen_iterative.sh
```

Note that this is the end-to-end script that integrated multiple process, including 1) training GPT2; 2) using GPT2 to generate synthetic samples; 3) the iterative process of distilling and two-stage trainig.