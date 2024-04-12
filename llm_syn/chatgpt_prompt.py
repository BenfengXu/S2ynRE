# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import os
import openai
import json
import random

# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key='xxx'


def prompt_format(ins):
    import json

    sub_start, sub_end = ins['h']['pos']
    obj_start, obj_end = ins['t']['pos']

    printable_tokens = []
    if sub_start <= obj_start:
        printable_tokens = printable_tokens + ins['token'][:sub_start]
        sub_start_marker_pos = len(printable_tokens)
        printable_tokens = printable_tokens + ['[Sub]'] + ins['token'][sub_start:sub_end] + ['[\Sub]'] + ins['token'][sub_end:obj_start]
        obj_start_marker_pos = len(printable_tokens)
        printable_tokens = printable_tokens + ['[Obj]'] + ins['token'][obj_start:obj_end] + ['[\Obj]'] + ins['token'][obj_end:]
    else:
        printable_tokens = printable_tokens + ins['token'][:obj_start]
        obj_start_marker_pos = len(printable_tokens)
        printable_tokens = printable_tokens + ['[Obj]'] + ins['token'][obj_start:obj_end] + ['[\Obj]'] + ins['token'][obj_end:sub_start]
        sub_start_marker_pos = len(printable_tokens)
        printable_tokens = printable_tokens + ['[Sub]'] + ins['token'][sub_start:sub_end] + ['[\Sub]'] + ins['token'][sub_end:]

    return " ".join(printable_tokens)


prompt_suffix = "The above is an illustrations of several structured relational sentences, the [Sub] and [Obj] mark the start position of entities, and [\Sub] and [\Obj] mark the end. Between entity pairs abstractive relations are entailed. Now please help me write more sentences, with similar topic, domain and the same sub-obj format, but diversified in entities, relations and semantics."

data = []
with open('./train_0.01.txt', 'r') as f:
    for line in f.readlines():
        data.append(json.loads(line.strip()))

print("===== Start querying =====")
for seed in range(1, 101):
    random.seed(seed)
    prompt = ""
    sampled_ins = random.sample(data, 5)
    for ins in sampled_ins:
        prompt += prompt_format(ins)
        prompt += '\n'
    prompt += prompt_suffix

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": "You are an intelligent writing assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    response["seed"] = seed
    response["prompt"] = prompt

    print("===== completed query seed {} prompt =====".format(seed))

    with open("./chatgpt_completion.json", 'a+') as f:
        json_str = json.dumps(response)
        f.write(json_str)
        f.write("\n")