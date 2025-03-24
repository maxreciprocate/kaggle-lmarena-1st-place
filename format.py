try:
    from itertools import batched
except:
    def batched(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

from functools import partial

debug = False

def fmt_v0(p, a, b, winner, max_length, tokenizer, submit):
    chat_template = "user:\n{p}<eot>\nbot:\n{a}<eot>\nassistant:\n{b}<eot>"
    chat_str = chat_template.format(p=p, a=a, b=b)
    chat_tok = tokenizer(chat_str, max_length=max_length, truncation=True)

    if submit:
        return chat_tok

    label = 0 if winner == "model_a" else 1
    return {**chat_tok, "chat_str": chat_str, "labels": label}

def fmt_v3(truncate_side, p, a, b, max_length, tokenizer):
    start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
    start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
    start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
    eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
    cut = tokenizer("[...]", add_special_tokens=False).input_ids
    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    # 21 is token formatting overhead
    max_length = max_length - 21
    min_length_of_section = 100

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    for pt, at, bt in batched(tok, 3):
        total_len = len(pt) + len(at) + len(bt)
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        ratio = max_length / total_len
        new_len_a = max(min_length_of_section, int(len(at) * ratio))
        new_len_b = max(min_length_of_section, int(len(bt) * ratio))
        new_len_p = max(min_length_of_section, int(len(pt) * ratio))
        if truncate_side == "left":
            pt_trunc = pt[-new_len_p:]
            at_trunc = at[-new_len_a:]
            bt_trunc = bt[-new_len_b:]
            if len(pt_trunc) < len(pt):
                pt_trunc = cut + pt_trunc
            if len(at_trunc) < len(at):
                at_trunc = cut + at_trunc
            if len(bt_trunc) < len(bt):
                bt_trunc = cut + bt_trunc
        elif truncate_side == "right":
            pt_trunc = pt[:new_len_p]
            at_trunc = at[:new_len_a]
            bt_trunc = bt[:new_len_b]
            if len(pt_trunc) < len(pt):
                pt_trunc = pt_trunc + cut
            if len(at_trunc) < len(at):
                at_trunc = at_trunc + cut
            if len(bt_trunc) < len(bt):
                bt_trunc = bt_trunc + cut
        elif truncate_side == "both":
            half_p = new_len_p // 2
            half_a = new_len_a // 2
            half_b = new_len_b // 2

            if len(pt) > new_len_p:
                pt_trunc = pt[:half_p] + cut + pt[-half_p:]
            else:
                pt_trunc = pt
            if len(at) > new_len_a:
                at_trunc = at[:half_a] + cut + at[-half_a:]
            else:
                at_trunc = at
            if len(bt) > new_len_b:
                bt_trunc = bt[:half_b] + cut + bt[-half_b:]
            else:
                bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos

        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

def fmt_v3_incl_rev(truncate_side, p, a, b, max_length, tokenizer):
     ab = fmt_v3(truncate_side, p, a, b, max_length, tokenizer)
     ba = fmt_v3(truncate_side, p, b, a, max_length, tokenizer)
     return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

def distribute_lengths(lengths, max_length):
    # Make a copy to avoid modifying the original list
    remaining = lengths.copy()
    result = [0] * len(lengths)
    n = len(lengths)

    # Rule 1: Take full length for small elements first
    thresholds = [max_length // n] * n
    for i, length in enumerate(lengths):
        if length <= thresholds[i]:
            result[i] = length
            remaining[i] = 0

    # Rule 2: Distribute remaining space proportionally
    total_remaining = sum(remaining)
    remaining_space = max_length - sum(result)

    if total_remaining > 0:
        for i, length in enumerate(remaining):
            if length > 0:
                # Calculate proportional share of remaining space
                share = int((length / total_remaining) * remaining_space)
                result[i] = min(lengths[i], share)

    return result

def distribute_lengths_adaptive(lengths, max_length):
    thresholds_variants = [
        # truncated only the longest
        [sorted(lengths)[1]] * 3,
        # don't truncated the smallest
        [min(lengths)] * 3,
        # evenly truncate
        [max_length / 3] * 3,
    ]

    for thresholds in thresholds_variants:
        result = [0] * len(lengths)
        # Make a copy to avoid modifying the original list
        remaining = lengths.copy()
        for i, length in enumerate(lengths):
            if length <= thresholds[i]:
                result[i] = length
                remaining[i] = 0

        # Rule 2: Distribute remaining space proportionally
        total_remaining = sum(remaining)
        remaining_space = max_length - sum(result)

        if remaining_space < 0:
            continue

        if total_remaining > 0:
            for i, length in enumerate(remaining):
                if length > 0:
                    # Calculate proportional share of remaining space
                    share = int((length / total_remaining) * remaining_space)
                    result[i] = min(lengths[i], share)

        return result


def fmt_v5(truncate_side, p, a, b, max_length, tokenizer):
    start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
    start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
    start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
    eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
    cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + eos)

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2]) for i in range(0,len(tok),3)]
    for pt, at, bt in tok:
        total_len = len(pt) + len(at) + len(bt) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths([len(pt), len(at), len(bt)], max_length - token_overhead)

        if truncate_side == "left":
            pt_trunc = pt[-new_len_p:]
            at_trunc = at[-new_len_a:]
            bt_trunc = bt[-new_len_b:]
            if len(pt_trunc) < len(pt):
                pt_trunc = cut + pt_trunc
            if len(at_trunc) < len(at):
                at_trunc = cut + at_trunc
            if len(bt_trunc) < len(bt):
                bt_trunc = cut + bt_trunc
        elif truncate_side == "right":
            pt_trunc = pt[:new_len_p]
            at_trunc = at[:new_len_a]
            bt_trunc = bt[:new_len_b]
            if len(pt_trunc) < len(pt):
                pt_trunc = pt_trunc + cut
            if len(at_trunc) < len(at):
                at_trunc = at_trunc + cut
            if len(bt_trunc) < len(bt):
                bt_trunc = bt_trunc + cut
        elif truncate_side == "both":
            half_p = new_len_p // 2
            half_a = new_len_a // 2
            half_b = new_len_b // 2

            if len(pt) > new_len_p:
                pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
            else:
                pt_trunc = pt
            if len(at) > new_len_a:
                at_trunc = at[:half_a-1] + cut + at[-half_a:]
            else:
                at_trunc = at
            if len(bt) > new_len_b:
                bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
            else:
                bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos
        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}


def fmt_v5_str(truncate_side, p, a, b, max_length, tokenizer):
    start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
    start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
    start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
    eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
    cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + eos)

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    out = []
    tok = [(tok[i], tok[i+1], tok[i+2]) for i in range(0,len(tok),3)]
    for pt, at, bt in tok:
        total_len = len(pt) + len(at) + len(bt) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            out.append({
                "prompt": tokenizer.decode(pt),
                "response_a": tokenizer.decode(at),
                "response_b": tokenizer.decode(bt),
            })
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths([len(pt), len(at), len(bt)], max_length - token_overhead)

        if truncate_side == "left":
            pt_trunc = pt[-new_len_p:]
            at_trunc = at[-new_len_a:]
            bt_trunc = bt[-new_len_b:]
            if len(pt_trunc) < len(pt):
                pt_trunc = cut + pt_trunc
            if len(at_trunc) < len(at):
                at_trunc = cut + at_trunc
            if len(bt_trunc) < len(bt):
                bt_trunc = cut + bt_trunc
        elif truncate_side == "right":
            pt_trunc = pt[:new_len_p]
            at_trunc = at[:new_len_a]
            bt_trunc = bt[:new_len_b]
            if len(pt_trunc) < len(pt):
                pt_trunc = pt_trunc + cut
            if len(at_trunc) < len(at):
                at_trunc = at_trunc + cut
            if len(bt_trunc) < len(bt):
                bt_trunc = bt_trunc + cut
        elif truncate_side == "both":
            half_p = new_len_p // 2
            half_a = new_len_a // 2
            half_b = new_len_b // 2

            if len(pt) > new_len_p:
                pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
            else:
                pt_trunc = pt
            if len(at) > new_len_a:
                at_trunc = at[:half_a-1] + cut + at[-half_a:]
            else:
                at_trunc = at
            if len(bt) > new_len_b:
                bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
            else:
                bt_trunc = bt

        out.append({
            "prompt": tokenizer.decode(pt_trunc),
            "response_a": tokenizer.decode(at_trunc),
            "response_b": tokenizer.decode(bt_trunc),
        })

    return {"prompt": [x["prompt"] for x in out], "response_a": [x["response_a"] for x in out], "response_b": [x["response_b"] for x in out]}


def fmt_v6(truncate_side, p, a, b, max_length, tokenizer):
    start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
    start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
    start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
    eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
    cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + eos)

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2]) for i in range(0,len(tok),3)]
    for pt, at, bt in tok:
        total_len = len(pt) + len(at) + len(bt) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths_adaptive([len(pt), len(at), len(bt)], max_length - token_overhead)

        if truncate_side == "left":
            pt_trunc = pt[-new_len_p:]
            at_trunc = at[-new_len_a:]
            bt_trunc = bt[-new_len_b:]
            if len(pt_trunc) < len(pt):
                pt_trunc = cut + pt_trunc
            if len(at_trunc) < len(at):
                at_trunc = cut + at_trunc
            if len(bt_trunc) < len(bt):
                bt_trunc = cut + bt_trunc
        elif truncate_side == "right":
            pt_trunc = pt[:new_len_p]
            at_trunc = at[:new_len_a]
            bt_trunc = bt[:new_len_b]
            if len(pt_trunc) < len(pt):
                pt_trunc = pt_trunc + cut
            if len(at_trunc) < len(at):
                at_trunc = at_trunc + cut
            if len(bt_trunc) < len(bt):
                bt_trunc = bt_trunc + cut
        elif truncate_side == "both":
            half_p = new_len_p // 2
            half_a = new_len_a // 2
            half_b = new_len_b // 2

            if len(pt) > new_len_p:
                pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
            else:
                pt_trunc = pt
            if len(at) > new_len_a:
                at_trunc = at[:half_a-1] + cut + at[-half_a:]
            else:
                at_trunc = at
            if len(bt) > new_len_b:
                bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
            else:
                bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos
        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

def fmt_v7_models(truncate_side, p, a, b, model_a, model_b, max_length, tokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        start_p = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        eot = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        cut = tokenizer("\n........\n", add_special_tokens=False).input_ids
    else:
        start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
        eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
        cut = tokenizer("[...]", add_special_tokens=False).input_ids

    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    token_overhead = len(bos + start_p + eot + eot + eot + eos)
    if debug:
        print(f'{token_overhead=}')

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2], ) for i in range(0,len(tok),3)]
    for (pt, at, bt), model_a, model_b in zip(tok, model_a, model_b):
        if model_a is None:
            model_a = "model"
        if model_b is None:
            model_b = "assistant"
        if "qwen" in tokenizer.name_or_path.lower():
            start_a = tokenizer(f"<|im_start|>{model_a}\n", add_special_tokens=False).input_ids
            start_b = tokenizer(f"<|im_start|>{model_b}\n", add_special_tokens=False).input_ids
        else:
            start_a = tokenizer(f"<start_of_turn>{model_a}\n", add_special_tokens=False).input_ids
            start_b = tokenizer(f"<start_of_turn>{model_b}\n", add_special_tokens=False).input_ids

        var_token_overhead = token_overhead + len(start_a) + len(start_b)

        total_len = len(pt) + len(at) + len(bt) + len(start_a) + len(start_b) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths([len(pt), len(at), len(bt)], max_length - var_token_overhead)

        if truncate_side == "left":
            pt_trunc = pt[-new_len_p:]
            at_trunc = at[-new_len_a:]
            bt_trunc = bt[-new_len_b:]
            if len(pt_trunc) < len(pt):
                pt_trunc = cut + pt_trunc
            if len(at_trunc) < len(at):
                at_trunc = cut + at_trunc
            if len(bt_trunc) < len(bt):
                bt_trunc = cut + bt_trunc
        elif truncate_side == "right":
            pt_trunc = pt[:new_len_p]
            at_trunc = at[:new_len_a]
            bt_trunc = bt[:new_len_b]
            if len(pt_trunc) < len(pt):
                pt_trunc = pt_trunc + cut
            if len(at_trunc) < len(at):
                at_trunc = at_trunc + cut
            if len(bt_trunc) < len(bt):
                bt_trunc = bt_trunc + cut
        elif truncate_side == "both":
            half_p = new_len_p // 2
            half_a = new_len_a // 2
            half_b = new_len_b // 2

            if len(pt) > new_len_p:
                pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
            else:
                pt_trunc = pt
            if len(at) > new_len_a:
                at_trunc = at[:half_a-1] + cut + at[-half_a:]
            else:
                at_trunc = at
            if len(bt) > new_len_b:
                bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
            else:
                bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos
        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

def fmt_crit(truncate_side, p, a, b, c, max_length, tokenizer):
    start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
    start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
    start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
    start_c = tokenizer("<start_of_turn>critic\n", add_special_tokens=False).input_ids
    eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
    cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + start_c + eot + eos)

    tok = tokenizer(sum(map(list, list(zip(p, a, b, c))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2], tok[i+3]) for i in range(0,len(tok),4)]
    for pt, at, bt, ct in tok:
        total_len = len(pt) + len(at) + len(bt) + len(ct) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + start_c + ct + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b, new_len_c = distribute_lengths([len(pt), len(at), len(bt), len(ct)], max_length - token_overhead)

        half_p = new_len_p // 2
        half_a = new_len_a // 2
        half_b = new_len_b // 2
        half_c = new_len_c // 2

        if len(pt) > new_len_p:
            pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
        else:
            pt_trunc = pt
        if len(at) > new_len_a:
            at_trunc = at[:half_a-1] + cut + at[-half_a:]
        else:
            at_trunc = at
        if len(bt) > new_len_b:
            bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
        else:
            bt_trunc = bt
        if len(ct) > new_len_c:
            ct_truct = ct[:half_c-1] + cut + ct[-half_c:]
        else:
            ct_truct = ct

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + start_c + ct_truct + eot + eos
        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

def fmt_v5_multi(truncate_side, p, a, b, max_length, tokenizer):
    name = tokenizer.name_or_path.lower()
    if "qwen" in name or "7b" in name or "14b" in name or "32b" in name or "72b" in name:
        start_p = tokenizer("<|im_start|>user\n", add_special_tokens=False).input_ids
        start_a = tokenizer("<|im_start|>model\n", add_special_tokens=False).input_ids
        start_b = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        eot = tokenizer("<|im_end|>\n", add_special_tokens=False).input_ids
        cut = tokenizer("\n........\n", add_special_tokens=False).input_ids
    else:
        start_p = tokenizer("<start_of_turn>user\n", add_special_tokens=False).input_ids
        start_a = tokenizer("<start_of_turn>model\n", add_special_tokens=False).input_ids
        start_b = tokenizer("<start_of_turn>assistant\n", add_special_tokens=False).input_ids
        eot = tokenizer("<end_of_turn>\n", add_special_tokens=False).input_ids
        cut = tokenizer("[...]", add_special_tokens=False).input_ids

    if tokenizer.bos_token_id is not None:
        bos = [tokenizer.bos_token_id]
    else:
        bos = []
    if tokenizer.eos_token_id is not None:
        eos = [tokenizer.eos_token_id]
    else:
        eos = []

    token_overhead = len(bos + start_p + eot + start_a + eot + start_b + eot + eos) + len(cut) * 3

    tok = tokenizer(sum(map(list, list(zip(p, a, b))), []), add_special_tokens=False).input_ids

    all_input_ids = []
    tok = [(tok[i], tok[i+1], tok[i+2]) for i in range(0,len(tok),3)]
    for pt, at, bt in tok:
        total_len = len(pt) + len(at) + len(bt) + token_overhead
        if total_len <= max_length:
            input_ids = bos + start_p + pt + eot + start_a + at + eot + start_b + bt + eot + eos
            all_input_ids.append(input_ids)
            continue

        new_len_p, new_len_a, new_len_b = distribute_lengths([len(pt), len(at), len(bt)], max_length - token_overhead)

        half_p = new_len_p // 2
        half_a = new_len_a // 2
        half_b = new_len_b // 2

        if len(pt) > new_len_p:
            pt_trunc = pt[:half_p-1] + cut + pt[-half_p:]
        else:
            pt_trunc = pt
        if len(at) > new_len_a:
            at_trunc = at[:half_a-1] + cut + at[-half_a:]
        else:
            at_trunc = at
        if len(bt) > new_len_b:
            bt_trunc = bt[:half_b-1] + cut + bt[-half_b:]
        else:
            bt_trunc = bt

        input_ids = bos + start_p + pt_trunc + eot + start_a + at_trunc + eot + start_b + bt_trunc + eot + eos
        if debug:
            print(f"len {total_len} -> {len(input_ids)}")
            print(f"len_p {len(pt)} -> {len(pt_trunc)}")
            print(f"len_a {len(at)} -> {len(at_trunc)}")
            print(f"len_b {len(bt)} -> {len(bt_trunc)}")
        all_input_ids.append(input_ids)

    return {"input_ids": all_input_ids}

def fmt_v5_incl_rev(truncate_side, p, a, b, max_length, tokenizer):
     ab = fmt_v5(truncate_side, p, a, b, max_length, tokenizer)
     ba = fmt_v5(truncate_side, p, b, a, max_length, tokenizer)
     return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

def fmt_v6_incl_rev(truncate_side, p, a, b, max_length, tokenizer):
     ab = fmt_v6(truncate_side, p, a, b, max_length, tokenizer)
     ba = fmt_v6(truncate_side, p, b, a, max_length, tokenizer)
     return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

def fmt_crit_incl_rev(truncate_side, p, a, b, c_ab, c_ba, max_length, tokenizer):
    ab = fmt_crit(truncate_side, p, a, b, c_ab, max_length, tokenizer)
    ba = fmt_crit(truncate_side, p, b, a, c_ba, max_length, tokenizer)
    return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

def fmt_v7_models_incl_rev(truncate_side, p, a, b, model_a, model_b, max_length, tokenizer):
    ab = fmt_v7_models(truncate_side, p, a, b, model_a, model_b, max_length, tokenizer)
    ba = fmt_v7_models(truncate_side, p, b, a, model_b, model_a, max_length, tokenizer)
    return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

def fmt_v5_multi_incl_rev(truncate_side, p, a, b, max_length, tokenizer):
    ab = fmt_v5_multi(truncate_side, p, a, b, max_length, tokenizer)
    ba = fmt_v5_multi(truncate_side, p, b, a, max_length, tokenizer)
    return {"input_ids": [[x, y] for x,y in zip(ab["input_ids"], ba["input_ids"])]}

formatting_map = {
    "fmt0": fmt_v0,
    "fmt3_left": partial(fmt_v3, "left"),
    "fmt3_right": partial(fmt_v3, "right"),
    "fmt3_both": partial(fmt_v3, "both"),
    "fmt3_both_incl_rev": partial(fmt_v3_incl_rev, "both"),
    "fmt5rev": partial(fmt_v5_incl_rev, "both"),
    "fmt6rev": partial(fmt_v6_incl_rev, "both"),
    "fmt_crit_incl_rev": partial(fmt_crit_incl_rev, "both"),
    "fmt7_models_rev": partial(fmt_v7_models_incl_rev, "both"),
    "fmt7_models": partial(fmt_v7_models, "both"),
    "fmt5multirev": partial(fmt_v5_multi_incl_rev, "both"),
}
