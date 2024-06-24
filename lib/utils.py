import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Generator

from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

from wonderwords import RandomWord

DEFAULT_OPENAI_SYSTEM_PROMPT = (
    'You are an AI assistant that helps people find information.'
)

DATASET_HANDLES = {
    'lima-test': 'data/inputs/lima-test.jsonl',
}

MODEL_HANDLES = {
    'llama-7b': 'meta-llama/Llama-2-7b-hf',
    'llama-chat-7b': 'meta-llama/Llama-2-7b-chat-hf',
}


def get_dataset_name(s: str):
    return DATASET_HANDLES.get(s, s)


def get_model_name(s: str):
    return MODEL_HANDLES.get(s, s)


def random_name():
    rand = RandomWord()
    adj = rand.word(include_categories=['adjectives'])
    noun = rand.word(include_categories=['noun'])
    return f'{adj}-{noun}'


# ------ #
# OpenAI #
# ------ #


def ensure_openai_format(
       messages: list[str] | list[dict[str, str]],
) -> list[dict[str, str]]:
    first_message = messages[0]
    if isinstance(first_message, dict):
        return messages

    new_messages = []
    for i, content in enumerate(messages):
        role = 'user' if i % 2 == 0 else 'assistant'
        new_messages.append({'role': role, 'content': content})

    return new_messages


def openai_configure(org: str = 'TAUR') -> OpenAI:
    return _openai_configure(OpenAI, org)


def openai_configure_async(org: str = 'TAUR') -> AsyncOpenAI:
    return _openai_configure(AsyncOpenAI, org)


def _openai_configure(cls, org):
    org = org.upper()
    if org == 'TAUR':
        return cls(
            api_key=os.environ['OPENAI_API_KEY_PERSONAL'],
            organization=os.environ['OPENAI_ORGANIZATION_TAUR'],
        )
    elif org == 'INDEED':
        return cls(
            base_url=os.environ['LLM_PROXY_QA_BASE'],
            api_key=os.environ['LLM_PROXY_QA_PENGUIN'],
            organization=os.environ['OPENAI_ORGANIZATION_INDEED'],
        )

    raise ValueError(f'Unknown org: "{org}" (Valid options are "INDEED" or "TAUR")')


def openai_chat_request(
        client: OpenAI,
        chat: list[dict[str, str]],
        max_retries: int = 10,
        default_sleep_time: int = 2,
        model: str = 'gpt-3.5-turbo',
        **openai_params: dict[str, Any],
) -> ChatCompletion:
    for i in range(max_retries):
        try:
            completion = client.chat.completions.create(
                messages=chat,
                model=model,
                **openai_params,
            )
            return completion
        except RateLimitError as e:
            if i == max_retries - 1:
                raise

            words = str(e).split(' ')
            try:
                sleep_time = int(words[words.index('after') + 1])
            except ValueError:
                sleep_time = default_sleep_time

            time.sleep(sleep_time)

    raise RuntimeError('Should not see me!')


def openai_async_requests(
        client: AsyncOpenAI,
        chats: list[list[dict[str, str]]],
        max_retries: int = 10,
        default_sleep_time: int = 2,
        model: str = 'gpt-3.5-turbo',
        **openai_params: dict[str, Any],
) -> list[ChatCompletion]:
    raise NotImplementedError()


def extract_prompt(obj: dict) -> None | str:
    return obj.get('prompt', None)


def extract_icl_examples(prompt: str, n_sections: int = 2) -> None | str:
    chunks = prompt.split('```')
    inputs = []
    outputs = []
    base = 2 * n_sections
    i_rem = 1
    o_rem = base - 1
    for i, chunk in enumerate(chunks):
        if i % base == i_rem:
            inputs.append(chunk.strip())
        elif i % base == o_rem:
            outputs.append(chunk.strip())

    inputs.pop()
    if not outputs[-1]:
        outputs.pop()

    assert len(inputs) == len(outputs)
    return inputs, outputs


def extract_outputs(obj: dict) -> list[str]:
    outputs = obj['outputs']
    if len(outputs) == 0:
        return []

    d = outputs[0]

    if 'message' in d:
        return openai_extract_outputs(obj)

    if 'generated_text' in d:
        return tgi_extract_outputs(obj)

    raise ValueError(f'unknown output format: {obj}')


def tgi_extract_outputs(obj: dict) -> list[str]:
    return [d['generated_text'].strip() for d in obj['outputs']]


def openai_extract_outputs(obj: dict) -> list[str]:
    return [d['message']['content'].strip() for d in obj['outputs']]


# ---- #
# Chat #
# ---- #


def truncate_chat_for_generation(
        chat: list[dict[str, Any]],
        max_chat_turns: int | None = None,
        assistant_role: str = 'assistant',
) -> tuple[list[dict[str, str]], dict[str, str] | None]:
    """Truncate a chat to a specific number of rounds.

    Each round ends when the "assistant" responds.
    For example::

        # Chat with 1 Turn
        [
            {'role': 'user', ...},
            {'role': 'assistant', ...},
        ]

        # Another chat with 1 Turn
        [
            {'role': 'system', ...},
            {'role': 'user', ...},
            {'role': 'assistant', ...},
        ]

        # Chat with 3 Turns
        [
            {'role': 'system', ...},
            {'role': 'user', ...},
            {'role': 'assistant', ...},
            {'role': 'user', ...},
            {'role': 'assistant', ...},
            {'role': 'user', ...},
            {'role': 'assistant', ...},
        ]

    Parameters
    ----------
    chat : list[dict[str, Any]]
        A list of messages in OpenAI Chat format.
    max_chat_turns : int | None
        The maximum number of chat turns to include in the
        output. Each assistant message counts as a turn.
        If ``None``, then all chat rounds are used.
    assistant_role : str
        The role that will be used to count assistant turns.
        By default, this is "assistant".

    Returns
    -------
    truncated_chat : list[dict[str, str]]
        The chat messages associated with the first max_chat_turns turns.
    next_assistant_message : dict[str, str] | None
        The assistant message following the first max_chat_turns turns.

    Raise
    -----
    ValueError
        If the chat is empty.
    """
    if len(chat) == 0:
        raise ValueError('empty chat')

    if max_chat_turns is not None:
        turn_count = 0
        truncated_chat: list[dict[str, str]] = []
        for message in chat:
            truncated_chat.append(message)
            if message['role'] == assistant_role:
                turn_count += 1
                if turn_count >= max_chat_turns:
                    break

        chat = truncated_chat

    if chat[-1]['role'] == assistant_role:
        return chat[:-1], chat[-1]
    else:
        return chat, None


# --------------- #
# Post Processing #
# --------------- #


def parse_number_list(content: str) -> list[str]:
    elements = []
    current = []
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line[0].isnumeric():
            if current:
                elements.append('\n'.join(current))

            _, line = line.split(maxsplit=1)
            current = [line]
        elif current:
            current.append(line)

    if current:
        elements.append('\n'.join(current))

    return elements


# ---- #
# Data #
# ---- #


def iter_jsonlines(file: str | Path) -> Generator[dict[str, Any], None, None]:
    with open(file) as fp:
        for line in fp:
            yield json.loads(line)


def load_instances(file: str | Path) -> list[dict[str, Any]]:
    ds = []
    for i, d in enumerate(iter_jsonlines(file)):
        md: dict[str, Any] = d.setdefault('metadata', {})
        i = md.setdefault('instance_id', i)
        ds.append(d)

    return ds


def load_outputs(file: str | Path) -> list[dict[str, Any]]:
    ds = []
    for i, d in enumerate(iter_jsonlines(file)):
        ok = d.get('ok', True)
        if not ok:
            ds.append(None)
        else:
            output = extract_outputs(d)
            ds.append(output)

    return ds


def load_evaluations(
        instance_file: str | Path,
        output_file: str | Path,
) -> list[dict[str, Any]]:
    instances = list(iter_jsonlines(instance_file))
    outputs = list(iter_jsonlines(output_file))
    assert len(instances) == len(outputs)

    evaluations = {}
    for instance, output in zip(instances, outputs):
        i = instance['metadata']['instance_id']
        evaluation = evaluations.setdefault(i, [])
        if not output['ok']:
            evaluation.append(None)
            continue

        output, = extract_outputs(output)
        try:
            d = json.loads(output)
        except Exception:
            evaluation.append(None)
            continue

        for v in d.values():
            if v['score'] == 'N/A':
                v['score'] = None
            else:
                v['score'] = float(v['score'])

        evaluation.append(d)

    return evaluations
