import json

import torch


class ProcessorPAB:
    PROMPT_PREFIX = """Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie."""

    PROMPT_SUFFIX = "verdict is: [["

    LABEL_COLS = ["winner_model_a", "winner_model_b", "winner_tie"]

    def __init__(self, tokenizer, max_length, support_system_role):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.support_system_role = support_system_role

    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Conversation between a User and two Assistants|>"
        tail = "<|The End of Conversation between a User and two Assistants|>\n"
        parts = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts.append(
                f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n\n### Assistant B:\n{response_b}\n"
            )
        text = "".join(parts)
        input_ids = self.tokenizer(
            text,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text = self.tokenizer.decode(input_ids)
        return head + truncated_text + tail

    def build_input(self, data):
        conversation = self.build_conversation(
            json.loads(data["prompt"]),
            json.loads(data["response_a"]),
            json.loads(data["response_b"]),
        )
        if self.support_system_role:
            messages = [
                {"role": "system", "content": self.PROMPT_PREFIX},
                {"role": "user", "content": conversation},
            ]
        else:
            messages = [
                {"role": "user", "content": f"{self.PROMPT_PREFIX}\n{conversation}"},
            ]
        input_text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            + self.PROMPT_SUFFIX
        )
        input_ids = self.tokenizer(
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]
        label = torch.tensor([data[col] for col in self.LABEL_COLS]).float()
        return dict(
            input_ids=input_ids,
            input_text=input_text,
            label=label,
        )


class ProcessorPAPB(ProcessorPAB):
    def build_conversation(self, prompts, responses_a, responses_b):
        head = "<|The Start of Assistant A’s Conversation with User|>"
        sep = "<|The End of Assistant A’s Conversation with User|>\n\n<|The Start of Assistant B’s Conversation with User|>"
        tail = "<|The End of Assistant B’s Conversation with User|>\n"
        parts_a = []
        parts_b = []
        for prompt, response_a, response_b in zip(prompts, responses_a, responses_b):
            if prompt is None:
                prompt = "null"
            if response_a is None:
                response_a = "null"
            if response_b is None:
                response_b = "null"
            parts_a.append(f"\n### User:\n{prompt}\n\n### Assistant A:\n{response_a}\n")
            parts_b.append(f"\n### User:\n{prompt}\n\n### Assistant B:\n{response_b}\n")
        text_a = "".join(parts_a)
        text_b = "".join(parts_b)
        input_ids_a = self.tokenizer(
            text_a,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids
        input_ids_b = self.tokenizer(
            text_b,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
        ).input_ids

        truncated_text_a = self.tokenizer.decode(input_ids_a)
        truncated_text_b = self.tokenizer.decode(input_ids_b)
        return head + truncated_text_a + sep + truncated_text_b + tail
