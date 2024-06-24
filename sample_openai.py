from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import HfArgumentParser

from lib import openai_util
from lib.side_info import (
    EmbeddingArgs, EmbeddingSideInfo,
    RandomArgs, RandomSideInfo,
    SummaryArgs, SummarySideInfo,
    SideInfo,
)
from lib.templates import TemplateWrapper, load_template


@dataclass
class Run:
    instance_file: str
    output_dir: str
    user_template_file: str | None = None
    user_template: TemplateWrapper | None = None

    def __post_init__(self):
        if self.user_template is None and self.user_template_file:
            self.user_template = load_template(self.user_template_file)


@dataclass
class OpenAIParams:
    model: str = 'gpt-3.5-turbo-0613'  # Hint: gpt-4-1106-preview
    temperature: float = 0
    response_format: str | None = None  # Hint: json_object
    n: int = 1
    max_tokens: int | None = 1024

    def dict(self):
        params = {
            'model': self.model,
            'temperature': self.temperature,
            'n': self.n,
            'max_tokens': self.max_tokens,
        }
        if self.response_format:
            params['response_format'] = {'type': self.response_format}

        return params


@dataclass
class Args:
    run: Run
    openai_params: OpenAIParams
    side_info_args: EmbeddingArgs | RandomArgs | SummaryArgs | None = None

    @staticmethod
    def type_list():
        return [Run, OpenAIParams]


def load_previous_outputs(file: str | Path):
    if not Path(file).exists():
        return {}

    previous_outputs = {}
    with open(file) as fp:
        for i, line in enumerate(fp):
            previous_outputs[i] = json.loads(line)

    print(f'found {len(previous_outputs)} previous outputs')
    return previous_outputs


def run(args: Args) -> None:
    output_dir = Path(args.run.output_dir)
    output_file = output_dir / 'samples.jsonl'
    previous_outputs = load_previous_outputs(output_file)

    with open(args.run.instance_file) as fp:
        instances = [json.loads(line) for line in fp]

    side_info = None
    if args.side_info_args is None:
        pass
    elif isinstance(args.side_info_args, EmbeddingArgs):
        side_info = EmbeddingSideInfo.load(args.side_info_args)
    elif isinstance(args.side_info_args, RandomArgs):
        side_info = RandomSideInfo.load(args.side_info_args)
    elif isinstance(args.side_info_args, SummaryArgs):
        side_info = SummarySideInfo.load(args.side_info_args)
    else:
        raise ValueError(
            f'Unknown side info arg w/ type: {type(args.side_info_args)}'
            f' and value: {args.side_info_args}'
        )

    if side_info is not None:
        assert args.run.user_template is not None

    def reformat_messages(index: int, messages: list[dict[str, str]]):
        if side_info is None:
            return messages

        messages = list(messages)
        user_message_index = 0
        if messages[0]['role'] == 'system':
            user_message_index = 1

        user_message = messages[user_message_index]
        assert user_message['role'] == 'user'
        content = user_message['content']
        context = side_info.get(index)
        content = args.run.user_template.render(input=content, context=context)
        messages[user_message_index] = {**user_message, 'content': content}
        return messages

    ids = []
    chats = []
    for i, x in enumerate(instances):
        output = previous_outputs.get(i)
        if output is None or not output['ok']:
            messages = reformat_messages(i, x['messages'])
            ids.append(i)
            chats.append(messages)

    if not len(ids):
        print('exiting, nothing to do')
        return

    print('calling openai')
    client = openai_util.configure_async_client(org='TAUR')

    outputs = openai_util.send_async_chat_request_batch(
        client=client,
        chats=chats,
        **args.openai_params.dict(),
    )
    num_failed = sum(1 for output in outputs if not output['ok'])
    total = len(instances)
    rate = num_failed / total
    print(f'failures: {num_failed} of {total} ({rate:.3%})')

    # Add new outputs to previous outputs
    for i, output in zip(ids, outputs):
        previous_outputs[i] = output

    print('writing outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    arg_dict = asdict(args)
    with open(output_dir / 'args.json', 'w') as fp:
        json.dump(arg_dict, fp, indent=2)

    with open(output_file, 'w') as fp:
        for i, _ in enumerate(instances):
            output = previous_outputs.get(i)
            if not output:
                break

            print(json.dumps(output, ensure_ascii=False), file=fp)


if __name__ == '__main__':
    parser = HfArgumentParser(Args.type_list())
    *args, unknown = parser.parse_args_into_dataclasses(
        return_remaining_strings=True,
    )
    if unknown:
        raise ValueError(f'unknown arguments: {unknown}')

    args = Args(*args)
    run(args)
