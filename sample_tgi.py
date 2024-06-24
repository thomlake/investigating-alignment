"""
Command-line interface for tgi.py

Sample generations from multiple models given a file of prompts.
The models should be saved in the folder mounted by TGI as `/data`.

file: sample.py
author: tlake
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pprint

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lib.side_info import (
    EmbeddingArgs, EmbeddingSideInfo,
    RandomArgs, RandomSideInfo,
    SummaryArgs, SummarySideInfo,
    SideInfo,
)
from lib.templates import load_template
from lib.tgi import Parameters, Server


@dataclass
class Run:
    instance_file: str
    output_dir: str
    chat_template: str | None = None
    chat_template_file: str | None = None
    strip_chat_template: bool = True
    name: str | None = None

    def __post_init__(self):
        if self.chat_template is None and self.chat_template_file:
            wrapped_template = load_template(
                self.chat_template_file,
                strip=self.strip_chat_template,
            )
            self.chat_template = wrapped_template.template


@dataclass
class Args:
    run: Run
    server: Server
    params: Parameters
    side_info_args: EmbeddingArgs | RandomArgs | SummaryArgs | None = None

    def __post_init__(self):
        if not self.run.name:
            self.run.name = Path(self.server.model_id).stem

    @staticmethod
    def type_list():
        return [Run, Server, Parameters]


def generate(server: Server, params: Parameters, prompts: list[str]):
    print('1. starting TGI container', server)
    try:
        server.start_container()
        print('2. generating samples')
        outputs = server.generate(prompts, params=params)
        print('3. stopping container')
    finally:
        if server.container_name:
            server.stop_container()

    print('done!')
    return [{'prompt': p, 'outputs': o} for p, o in zip(prompts, outputs)]


def create_prompts(
        data: list[dict],
        tokz: PreTrainedTokenizerBase,
        side_info: SideInfo | None = None,
) -> list[str]:

    def format_prompt(
            index: int,
            chat: list[dict[str, str]],
    ) -> str:
        # Total hack!
        if side_info is not None:
            context = side_info.get(index)
            context_message = {'role': 'context', 'content': context}
            chat = [context_message, *chat]

        return tokz.apply_chat_template(
            chat,
            tokenize=False,
            add_generation_prompt=True,
        )

    prompts: list[str] = []
    for i, d in enumerate(data):
        prompt = format_prompt(i, d['messages'])
        prompts.append(prompt)

    return prompts


def run(args: Args):
    pprint(args)

    output_dir = Path(args.run.output_dir)
    if output_dir.exists():
        raise ValueError(f'output dir already exists "{output_dir}"')

    if not Path(args.server.volume).exists():
        raise ValueError('docker volume does not exist')

    print('loading inputs')
    with open(args.run.instance_file) as fp:
        instances = [json.loads(line) for line in fp]

    tokz = AutoTokenizer.from_pretrained(args.server.model_id)
    if args.run.chat_template is not None:
        tokz.chat_template = args.run.chat_template
    elif tokz.chat_template is not None:
        args.run.chat_template = tokz.chat_template
    else:
        raise ValueError('no chat template defined')

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

    prompts = create_prompts(instances, tokz, side_info)
    print('!!! Begin First Prompt !!!')
    print(prompts[0])
    print('!!! End First Prompt !!!')

    print('generating output')
    output_data = generate(args.server, args.params, prompts)

    output_dir.mkdir(parents=True)
    arg_dict = asdict(args)
    with open(output_dir / 'args.json', 'w') as fp:
        json.dump(arg_dict, fp, indent=2)

    with open(output_dir / 'samples.jsonl', 'w') as fp:
        for d in output_data:
            print(json.dumps(d, ensure_ascii=False), file=fp)
