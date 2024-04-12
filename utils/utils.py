import re


def process_synsetic_samples(raw_sample):
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
        if not (h_start < h_end < t_start < t_end or t_start < t_end < h_start < h_end):
            return None
        if h_start < t_start:
            tokens = raw_sample[:h_start].split(' ')
            h_start_tok = len(tokens)
            h_name = raw_sample[h_start + len_marker:h_end]
            tokens += [h_name]
            h_end_tok = len(tokens)
            tokens += raw_sample[h_end + len_marker:t_start].split(' ')
            t_start_tok = len(tokens)
            t_name = raw_sample[t_start + len_marker:t_end]
            tokens += [t_name]
            t_end_tok = len(tokens)
            tokens += raw_sample[t_end + len_marker:].split(' ')
        else:
            tokens = raw_sample[:t_start].split(' ')
            t_start_tok = len(tokens)
            t_name = raw_sample[t_start + len_marker:t_end]
            tokens += [t_name]
            t_end_tok = len(tokens)
            tokens += raw_sample[t_end + len_marker:h_start].split(' ')
            h_start_tok = len(tokens)
            h_name = raw_sample[h_start + len_marker:h_end]
            tokens += [h_name]
            h_end_tok = len(tokens)
            tokens += raw_sample[h_end + len_marker:].split(' ')
        # edge cases, dirty work to rectify them contemporarily
        if tokens[0] == '':
            tokens = tokens[1:]
            h_start_tok -= 1
            h_end_tok -= 1
            t_start_tok -= 1
            t_end_tok -= 1
        instance = {"token": tokens,
                    "h": {"name": h_name, "pos": [h_start_tok, h_end_tok]},
                    "t": {"name": t_name, "pos": [t_start_tok, t_end_tok]},
                    "relation": "unknown"}
        return instance
    return None
