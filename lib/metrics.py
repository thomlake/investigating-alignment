from itertools import combinations

import numpy as np


def jaccard_similarity(sample_1: list[str], sample_2: list[str]) -> float:
    sample_1 = set(sample_1)
    sample_2 = set(sample_2)
    return len(sample_1 & sample_2) / len(sample_1 | sample_2)


def pairwise_jaccard_similarity(
        samples: list[list[str]],
        skip_empty: bool = True,
        reduction='mean'
) -> float | None:
    pairs = list(combinations(samples, 2))
    values = []
    for s1, s2 in pairs:
        if any([s1, s2]):
            values.append(jaccard_similarity(s1, s2))
        elif not skip_empty:
            raise ValueError(f'pair with empty elements: {s1} | {s2}')

    if not len(values):
        return None

    if callable(reduction):
        return reduction(values)

    if reduction == 'mean':
        return sum(values) / len(values)
    elif reduction == 'max':
        return max(values)

    raise ValueError(f'unknown reduction: "{reduction}"')


def length_difference(samples: list[list[str]]):
    pairs = combinations(samples, 2)
    return np.mean([abs(len(s1) - len(s2)) for s1, s2 in pairs])


def mean_length(samples: list[list[str]]):
    return np.mean([len(s) for s in samples])


def fraction_prefix_matches(samples: list[str], n: int) -> float:
    prefixes = {s[:n] for s in samples}
    return len(prefixes) / len(samples)


ABSTAIN_PHRASES = [
    "ai language model",
    "cannot have personal",
    "can't have personal",
    "don't have personal",
    "do not have personal",
    "cannot provide personal",
    "can't provide personal",
    "cannot condone",
    "can't condone",
]

# "I'm sorry", "I apologize", "As an", "I'm just", "I'm an", "I cannot"


def contains_abstain_phrase(text: str):
    text = text.lower()
    return any(phrase in text for phrase in ABSTAIN_PHRASES)


def count_number_list_length(text: str):
    lines = text.lower().split('\n')
    count = 0

    def list_start(i):
        return f'{i}. '

    for line in lines:
        if line.startswith(list_start(count + 1)):
            count += 1

    return count


def contains_number_list(text: str, threshold: int = 2):
    return count_number_list_length(text) >= threshold


def count_bullet_list_length(text: str):
    lines = text.lower().split('\n')
    count = 0

    for line in lines:
        if line.startswith('- '):
            count += 1

    return count


def contains_bullet_list(text: str, threshold: int = 2):
    return count_bullet_list_length(text) >= threshold
