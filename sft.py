"""SFT data: the template and the loss mask. Part 2 edits `format_example`
and `collate`; the rest is given.

An E2E example is a meaning representation (prompt) and a reference
(target). For training it becomes one sequence,

    <prompt tokens> <target tokens> <|endoftext|>

with loss on the target span and its end-of-text token only (lecture 12,
"Templates and Masking"). Tokenizer: A4's GPT-2 BPE through tiktoken.
"""

import json
import os

import tiktoken
import torch

ENC = tiktoken.get_encoding("gpt2")
EOT = ENC.eot_token                 # 50256, <|endoftext|>
IGNORE = -100                       # F.cross_entropy ignores targets of -100

# The template. At generation time the prompt ends after the colon.
PROMPT = "{mr}\nDescription:"

HERE = os.path.dirname(os.path.abspath(__file__))
E2E_DIR = os.path.join(HERE, "data", "e2e")


def format_example(mr: str, ref: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, target_ids): PROMPT filled with `mr`, and a
    leading space plus `ref` followed by EOT."""
    prompt = PROMPT.format(mr=mr)
    target = " " + ref
    return ENC.encode(prompt), ENC.encode(target) + [EOT]


def collate(examples: list[tuple[list[int], list[int]]]):
    """Batch formatted examples into (x, y) int64 tensors of shape (B, T):
    y[t] is the token after x[t], as in A1's train.py. Right-pad with EOT
    to the longest row; y is IGNORE on prompt and padding positions and
    the target span (including EOT) is scored."""
    sequences = [prompt + target for prompt, target in examples]
    lengths = [len(seq) - 1 for seq in sequences]
    width = max(lengths)
    x = torch.full((len(examples), width), EOT, dtype=torch.long)
    y = torch.full((len(examples), width), IGNORE, dtype=torch.long)

    for row, ((prompt, target), sequence, length) in enumerate(
        zip(examples, sequences, lengths)
    ):
        x[row, :length] = torch.tensor(sequence[:-1], dtype=torch.long)
        target_start = len(prompt) - 1
        y[row, target_start:length] = torch.tensor(target, dtype=torch.long)

    return x, y


def load_split(split: str, e2e_dir: str = E2E_DIR) -> list[dict]:
    """Rows of data/e2e/<split>.jsonl, written by data/prepare_e2e.py."""
    path = os.path.join(e2e_dir, f"{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run python data/prepare_e2e.py")
    with open(path) as f:
        return [json.loads(line) for line in f]


def describe(prompt_ids: list[int], target_ids: list[int]) -> str:
    """One formatted example token by token, with the mask beside it."""
    x, y = collate([(prompt_ids, target_ids)])
    toks = [ENC.decode([t]) for t in x[0].tolist()]
    out = []
    for tok, tgt in zip(toks, y[0].tolist()):
        shown = repr(tok)[1:-1]
        out.append(f"{shown:>14s}  {'scored' if tgt != IGNORE else '-'}")
    return "\n".join(out)
