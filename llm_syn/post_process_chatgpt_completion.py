import re
import json

def process_synsetic_samples(raw_sample):
    # clean
    raw_sample = raw_sample.strip()
    if raw_sample.startswith(("1. ", "2. ", "3. ", "4. ", "5. ", "6. ", "7. ", "8. ", "9. ")):
        raw_sample = raw_sample[3:]
    elif raw_sample.startswith(("10. ", "11. ", "12. ", "13. ", "14. ", "15. ", "16. ", "17. ", "18. ", "19. ", "20. ")):
        raw_sample = raw_sample[4:]

    # check
    token = None
    h = None
    t = None
    # Once you know how your string is being encoded, you can then think about what the re module will do with it. For instance, if you want to escape \ in a string you pass to the re module, you will need to pass \\ to re, which means you will need to use \\\\ in your quoted Python string. The Python string will end up with \\ and the re module will treat this as a single literal \ character.
    h_start = [m.start() for m in re.finditer('\[Sub\]', raw_sample)]
    h_end = [m.start() for m in re.finditer('\[\\\\Sub\]', raw_sample)]
    t_start = [m.start() for m in re.finditer('\[Obj\]', raw_sample)]
    t_end = [m.start() for m in re.finditer('\[\\\\Obj\]', raw_sample)]
    if not len(h_start) == len(h_end) == len(t_start) == len(t_end) == 1:
        return None
    len_s_marker = len('[Sub]')
    len_e_marker = len('[\Sub]')
    h_start = h_start[0]
    h_end = h_end[0]
    t_start = t_start[0]
    t_end = t_end[0]
    if not (h_start < h_end < t_start < t_end or t_start < t_end < h_start < h_end):
        return None
    
    # format
    if h_start < t_start:
        tokens = raw_sample[:h_start].strip().split(' ')
        h_start_tok = len(tokens)
        h_name = raw_sample[h_start + len_s_marker:h_end].strip().split(' ')
        tokens += h_name
        h_end_tok = len(tokens)
        tokens += raw_sample[h_end + len_e_marker:t_start].strip().split(' ')
        t_start_tok = len(tokens)
        t_name = raw_sample[t_start + len_s_marker:t_end].strip().split(' ')
        tokens += t_name
        t_end_tok = len(tokens)
        tokens += raw_sample[t_end + len_e_marker:].strip().split(' ')
    else:
        tokens = raw_sample[:t_start].strip().split(' ')
        t_start_tok = len(tokens)
        t_name = raw_sample[t_start + len_s_marker:t_end].strip().split(' ')
        tokens += t_name
        t_end_tok = len(tokens)
        tokens += raw_sample[t_end + len_e_marker:h_start].strip().split(' ')
        h_start_tok = len(tokens)
        h_name = raw_sample[h_start + len_s_marker:h_end].strip().split(' ')
        tokens += h_name
        h_end_tok = len(tokens)
        tokens += raw_sample[h_end + len_e_marker:].strip().split(' ')
    # edge cases, dirty work to rectify them contemporarily
    if tokens[0] == '':
        tokens = tokens[1:]
        h_start_tok -= 1
        h_end_tok -= 1
        t_start_tok -= 1
        t_end_tok -= 1
    instance = {"token": tokens,
                "h": {"name": " ".join(h_name), "pos": [h_start_tok, h_end_tok]},
                "t": {"name": " ".join(t_name), "pos": [t_start_tok, t_end_tok]},
                "relation": "unknown"}
    return instance


with open("./chatgpt_completion.json", 'r') as f:
    for line in f.readlines():
        responses = json.loads(line.strip())
        responses = responses["choices"][0]["message"]["content"].split("\n")
        for response in responses:
            response_formatted = process_synsetic_samples(response)
            if response_formatted is not None:
                with open("./chatgpt_completion_formatted.json", 'a+') as f:
                    json_str = json.dumps(response_formatted)
                    f.write(json_str)
                    f.write("\n")