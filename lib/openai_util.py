import asyncio
import os
import time
from typing import Any

from openai import AsyncOpenAI, OpenAI, RateLimitError
from openai.types.chat import ChatCompletion

MAX_RETRIES = 5
DEFAULT_SLEEP_TIME = 2
DEFAULT_OPENAI_SYSTEM_PROMPT = (
    'You are an AI assistant that helps people find information.'
)


def ensure_chat_format(
       messages: list[str] | list[dict[str, str]],
) -> list[dict[str, str]]:
    first_message = messages[0]
    if isinstance(first_message, dict):
        return messages

    new_messages = []
    for i, content in enumerate(messages):
        assert isinstance(content, str)

        role = 'user' if i % 2 == 0 else 'assistant'
        new_messages.append({'role': role, 'content': content})

    return new_messages


def configure_client(org: str = 'TAUR') -> OpenAI:
    return _configure_client(OpenAI, org)


def configure_async_client(org: str = 'TAUR') -> AsyncOpenAI:
    return _configure_client(AsyncOpenAI, org)


def _configure_client(cls, org):
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


def get_rate_limit_sleep_time(error: RateLimitError) -> int:
    words = str(error).split(' ')
    try:
        return int(words[words.index('after') + 1])
    except ValueError:
        return DEFAULT_SLEEP_TIME


def send_chat_request(
        client: OpenAI,
        chat: list[dict[str, str]],
        model: str = 'gpt-3.5-turbo',
        **openai_params: dict[str, Any],
) -> ChatCompletion:
    for i in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                messages=chat,
                model=model,
                **openai_params,
            )
            outputs = [choice.model_dump() for choice in completion.choices]
            return {'ok': True, 'outputs': outputs}
        except RateLimitError as e:
            if i == MAX_RETRIES - 1:
                return {'ok': False, 'error': str(e)}

            sleep_time = get_rate_limit_sleep_time(e)
            time.sleep(sleep_time)

    raise RuntimeError('Should not see me!')


async def send_async_chat_request(
        client: AsyncOpenAI,
        chat: list[dict[str, str]],
        **openai_params: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    for i in range(MAX_RETRIES):
        try:
            completion = await client.chat.completions.create(
                messages=chat,
                **openai_params,
            )
            outputs = [choice.model_dump() for choice in completion.choices]
            return {'ok': True, 'outputs': outputs}
        except RateLimitError as e:
            if i == MAX_RETRIES - 1:
                return {'ok': False, 'error': str(e)}

            sleep_time = get_rate_limit_sleep_time(e)
            time.sleep(sleep_time)

    raise RuntimeError('Should not see me!')


def send_async_chat_request_batch(
        client: AsyncOpenAI,
        chats: list[list[dict[str, str]]],
        max_concurrency: int | None = 10,
        status_frequency: int | None = 20,
        model: str = 'gpt-3.5-turbo',
        **openai_params: dict[str, Any],
) -> list[ChatCompletion]:
    coroutine = _send_async_chat_request_batch(
        client=client,
        chats=chats,
        max_concurrency=max_concurrency,
        status_frequency=status_frequency,
        model=model,
        **openai_params,
    )
    result = asyncio.run(coroutine)
    return result


async def _send_async_chat_request_batch(
        client: AsyncOpenAI,
        chats: list[list[dict[str, str]]],
        max_concurrency: int | None = 10,
        status_frequency: int | None = 20,
        **openai_params: dict[str, Any],
) -> list[dict[str, Any]]:
    total = len(chats)
    coroutines = (
        send_async_chat_request(client, chat, **openai_params) for chat in chats
    )

    run_semaphore: asyncio.Semaphore | None = None
    if max_concurrency is not None:
        run_semaphore = asyncio.Semaphore(max_concurrency)

    async def wrapper(i: int, coroutine):
        if run_semaphore is not None:
            async with run_semaphore:
                result = await coroutine
        else:
            result = await coroutine

        if status_frequency is not None and (i + 1) % status_frequency == 0:
            print(f'[{(i+1)/total:.2%}] completed {i+1} of {total}')

        return result

    wrapped_coroutines = (wrapper(i, c) for i, c in enumerate(coroutines))
    outputs = await asyncio.gather(*wrapped_coroutines)
    return outputs
