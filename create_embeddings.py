# Intended for
# https://huggingface.co/intfloat/e5-mistral-7b-instruct
#
# May work with other models.
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from lib.utils import load_instances, load_outputs

# Usage
# normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# - or -
# norms = np.linalg.norm(embeddings, axis=1)
# norms = np.repeat(norms[: , None], embeddings.shape[-1], axis=1)
# embeddings = embeddings / norms
# scores = (embeddings[:2] @ embeddings[2:].T) * 100
# print(scores.tolist())

PROMPT_TYPE = 'prompt'
RESPONSE_TYPE = 'response'

INPUT_TYPE_MAP = {
    'prompt_similar': {
        'type': PROMPT_TYPE,
        'instruct': 'Given a query, retrieve similar queries',
    },
    'response_similar': {
        'type': RESPONSE_TYPE,
        'instruct': 'Given an answer, retrieve similar answers',
    },
    'response_format': {
        'type': RESPONSE_TYPE,
        'instruct': 'Given an answer, retrieve answers with a similar format',
    },
}


# def __post_init__(self):
#     if self.task_description is None:
#         if self.input_type == 'prompt':
#             self.task_description = 'Given a query, retrieve similar queries'
#         elif self.input_type == 'response':
#             self.task_description = 'Given an answer, retrieve similar answers'
#         else:
#             raise ValueError(f'unknown input type {self.input_type}')


@dataclass
class Args:
    embedding_file: str
    input_file: str
    input_type: str
    model: str = 'intfloat/e5-mistral-7b-instruct'
    task_description: str | None = None
    batch_size: int = 4


def extract_prompt(obj: dict) -> str:
    m: dict[str, str] = obj['messages'][0]
    assert m['role'] == 'user'
    assert isinstance(m['content'], str)
    return m['content']


def get_instruct(task_description: str, string: str) -> str:
    assert isinstance(string, str)
    return f'Instruct: {task_description}\nQuery: {string}'


def last_token_pool(
        last_hidden_states: Tensor,
        attention_mask: Tensor,
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        idx = torch.arange(batch_size, device=last_hidden_states.device)
        return last_hidden_states[idx, sequence_lengths]


def embed(args: Args, model, tokenizer, inputs):
    embedding_chunks = []

    # model = model.to(0)
    with torch.no_grad():
        dl = DataLoader(inputs, batch_size=args.batch_size)
        for batch in tqdm(dl):
            batch_tensors = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=4096,
                return_tensors='pt',
                add_special_tokens=False,
            )
            batch_tensors = batch_tensors.to(model.device)
            outputs = model(**batch_tensors)
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_tensors['attention_mask'],
            )
            embedding_chunks.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(embedding_chunks)
    return embeddings


def run(args: Args, subsample: int | None = None) -> None:
    if Path(args.embedding_file).exists():
        raise ValueError(f'embedding_file already exists "{args.embedding_file}"')

    print('Creating embeddings')
    print(json.dumps(asdict(args), indent=2))

    input_type_info = INPUT_TYPE_MAP[args.input_type]
    if args.task_description is None:
        args.task_description = input_type_info['instruct']

    if input_type_info['type'] == PROMPT_TYPE:
        instances = load_instances(args.input_file)
        strings = [extract_prompt(obj) for obj in instances]
    elif input_type_info['type'] == RESPONSE_TYPE:
        response_samples = load_outputs(args.input_file)
        strings = [s for s, *_ in response_samples]
    else:
        raise ValueError(f'unknown input type {args.input_type}')

    if subsample:
        strings = strings[:subsample]

    instructions = [get_instruct(args.task_description, s) for s in strings]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, load_in_8bit=True)

    instructions = [s + tokenizer.pad_token for s in instructions]
    embeddings = embed(args, model, tokenizer, instructions)

    args_json = json.dumps(asdict(args))
    strings = np.array(strings)
    np.savez(args.embedding_file, args=args_json, embeddings=embeddings, strings=strings)
