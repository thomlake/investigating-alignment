from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from transformers import HfArgumentParser

from lib.openai_util import DEFAULT_OPENAI_SYSTEM_PROMPT
from lib.templates import TemplateWrapper, load_template
from lib.utils import load_instances, load_outputs


@dataclass
class Args:
    result_dir: str
    instance_file: str
    output_file: str
    template_file: str
    sample_index: int | None = None
    system_prompt: str = DEFAULT_OPENAI_SYSTEM_PROMPT
    _template_wrapper: TemplateWrapper | None = None

    def __post_init__(self):
        if not self._template_wrapper:
            self._template_wrapper = load_template(self.template_file)

    @property
    def template(self):
        return self._template_wrapper


def run(args: Args) -> None:
    result_dir = Path(args.result_dir)
    if result_dir.exists():
        raise ValueError(f'output already exists "{result_dir}"')

    instances = load_instances(args.instance_file)
    outputs = load_outputs(args.output_file)

    assert len(instances) == len(outputs)

    result_dir.mkdir(parents=True, exist_ok=False)
    with open(result_dir / 'args.json', 'w') as fp:
        json.dump(asdict(args), fp, indent=2)

    with open(result_dir / 'data.jsonl', 'w') as fp:
        for instance, samples in zip(instances, outputs):
            i = instance['metadata']['instance_id']
            input = instance['messages'][0]['content']
            if args.sample_index is not None:
                samples = [samples[args.sample_index]]

            for output in samples:
                user_content = args.template.render(
                    input=input,
                    output=output,
                )
                user_message = {'role': 'user', 'content': user_content}
                if args.system_prompt:
                    system_message = {'role': 'system', 'content': args.system_prompt}
                    messages = [system_message, user_message]
                else:
                    messages = [user_message]

                result = {'metadata': {'instance_id': i}, 'messages': messages}
                print(json.dumps(result, ensure_ascii=False), file=fp)


if __name__ == '__main__':
    parser = HfArgumentParser(Args)
    args, unknown = parser.parse_args_into_dataclasses(
        return_remaining_strings=True,
    )
    if unknown:
        raise ValueError(f'Unknown arguments: {unknown}')

    raise ValueError
    run(args)

    instance_file = './data/LIMA-test-manual-50/instances/{setting}/data.jsonl'
    model_dir = './data/LIMA-test-manual-50/outputs/{model}-{setting}'
    output_dir = './data/LIMA-test-manual-50/eval-quality/instances/{model}-{setting}'

    models = [
        # 'gpt-3.5-turbo',
        'llama-2-base',
        'llama-2-chat',
        'llama-2-icl',
        # 'mistral-base',
        # 'mistral-icl',
        # 'mistral-instruct',
        # 'mistral-sft',
        # 'zephyr',
    ]

    for model in models:
        for setting in settings:
            model_path = Path(model_dir.format(model=model, setting=setting))
            openai_model_file = model_path / 'openai_samples.jsonl'
            tgi_model_file = model_path / 'samples.jsonl'
            if openai_model_file.is_file():
                model_file = str(openai_model_file)
            elif tgi_model_file.is_file():
                model_file = str(tgi_model_file)
            else:
                raise ValueError(f'no samples found in model path: "{model_path}"')

            args = Args(
                output_dir=output_dir.format(model=model, setting=setting),
                instance_file=instance_file.format(setting=setting),
                model_file=model_file,
                template_file='./lib/template_library/just_eval_multiscore.jinja',
                sample_index=None,
            )
            print(args)
            run(args)

    # args = Args(
    #     instance_file='./data/lima-test-personas/instances-no-context/data.jsonl',
    #     model_file='./data/temperature-0/output/llama-2-icl-no-context/samples.jsonl',
    #     output_dir='./data/temperature-0/evaluation/instances/llama-2-icl-no-context',
    #     template_file='./lib/template_library/just_eval_multiscore.jinja',
    # )
    # print(args)
    # main(args)
