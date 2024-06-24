from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm.auto import tqdm
from transformers import HfArgumentParser

from lib.templates import TemplateWrapper, load_template
from lib.utils import (
    DEFAULT_OPENAI_SYSTEM_PROMPT,
    extract_outputs,
)


@dataclass
class Args:
    output_dir: str
    instance_file: str
    model_file_1: str
    model_file_2: str
    template_file: str
    sample_index_1: int | None = 0
    sample_index_2: int | None = 0
    system_prompt: str = DEFAULT_OPENAI_SYSTEM_PROMPT
    _template_wrapper: TemplateWrapper | None = None

    def __post_init__(self):
        if not self._template_wrapper:
            self._template_wrapper = load_template(self.template_file)

    @property
    def template(self):
        return self._template_wrapper


def main(args: Args) -> None:
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f'output already exists "{output_dir}"')

    with open(args.instance_file) as fp:
        instances = [json.loads(line) for line in fp]

    with open(args.model_file_1) as fp:
        outputs_1 = [json.loads(line) for line in fp]

    outputs_1 = [extract_outputs(d) for d in outputs_1]

    with open(args.model_file_2) as fp:
        outputs_2 = [json.loads(line) for line in fp]

    outputs_2 = [extract_outputs(output) for output in outputs_2]

    assert len(instances) == len(outputs_1) == len(outputs_2)

    output_dir.mkdir(parents=True, exist_ok=False)
    with open(output_dir / 'args.json', 'w') as fp:
        json.dump(asdict(args), fp, indent=2)

    with open(output_dir / 'data.jsonl', 'w') as fp:
        loop_iter = enumerate(zip(instances, outputs_1, outputs_2))
        loop_iter = tqdm(loop_iter, total=len(instances))
        for i, (instance, samples_1, samples_2) in loop_iter:
            input = instance['messages'][0]['content']
            if args.sample_index_1 is not None:
                samples_1 = [samples_1[args.sample_index_1]]

            if args.sample_index_2 is not None:
                samples_2 = [samples_2[args.sample_index_2]]

            for y1 in samples_1:
                for y2 in samples_2:
                    content = args.template.render(input=input, outputs=[y1, y2])
                    messages = []
                    if args.system_prompt:
                        m = {'role': 'system', 'content': args.system_prompt}
                        messages.append(m)

                    messages.append({'role': 'user', 'content': content})
                    output = {'metadata': {'instance_id': i}, 'messages': messages}
                    print(json.dumps(output, ensure_ascii=False), file=fp)


if __name__ == '__main__':
    # parser = HfArgumentParser(Args)
    # args, unknown = parser.parse_args_into_dataclasses(
    #     return_remaining_strings=True,
    # )
    # if unknown:
    #     raise ValueError(f'Unknown arguments: {unknown}')

    # main(args)

    # evaluate-missing
    # args = Args(
    #     output_dir='./data/lima-test-personas/evaluate-missing/2024-02-25/instances/llama-2-base--llama-2-chat',
    #     instance_file='./data/lima-test-personas/instances-no-context/data.jsonl',
    #     reference_response_file='./data/lima-test-personas/output-2024-02-04/llama-2-chat-no-context/samples.jsonl',
    #     output_response_file='./data/lima-test-personas/output-2024-02-04/llama-2-base-no-context/samples.jsonl',
    #     template_file='./lib/template_library/evaluate_missing.jinja',
    #     reference_sample_index=0,
    #     use_all_output_samples=False,
    # )

    # evaluate-missing-and-relevance
    # args = Args(
    #     output_dir='./data/lima-test-personas/evaluate-missing-and-relevance/2024-02-25/instances/llama-2-base--llama-2-chat',
    #     instance_file='./data/lima-test-personas/instances-no-context/data.jsonl',
    #     reference_response_file='./data/lima-test-personas/output-2024-02-04/llama-2-chat-no-context/samples.jsonl',
    #     output_response_file='./data/lima-test-personas/output-2024-02-04/llama-2-base-no-context/samples.jsonl',
    #     template_file='./lib/template_library/evaluate_missing_and_relevance.jinja',
    #     reference_sample_index=0,
    #     use_all_output_samples=False,
    # )

    # evaluate-items-missing-and-relevance
    args = Args(
        output_dir=   './data/LIMA-test-manual-50/extract-topics/instances/ref_llama-2-chat--alt_llama-2-icl',
        instance_file='./data/LIMA-test-manual-50/instances/no-context/data.jsonl',
        model_file_1= './data/LIMA-test-manual-50/outputs/llama-2-chat-no-context/samples.jsonl',
        model_file_2= './data/LIMA-test-manual-50/outputs/llama-2-icl-no-context/samples.jsonl',
        template_file='./lib/template_library/extract_topics_two.jinja',
        sample_index_1=0,
        sample_index_2=None,
    )
    print(args)
    main(args)
    exit(0)

    instance_file = './data/lima-test-personas/instances-{setting}/data.jsonl'
    model_dir = './data/lima-test-personas/output-2024-02-04/{model}-{setting}'
    output_dir = './data/lima-test-personas/evaluation-2024-02-09--all-gpt4/instances/{model}-{setting}'

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

    settings = [
        'no-context',
        # 'single-context',
        # 'multi-context',
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
                use_all_samples=True,
            )
            print(args)
            main(args)

    # args = Args(
    #     instance_file='./data/lima-test-personas/instances-no-context/data.jsonl',
    #     model_file='./data/temperature-0/output/llama-2-icl-no-context/samples.jsonl',
    #     output_dir='./data/temperature-0/evaluation/instances/llama-2-icl-no-context',
    #     template_file='./lib/template_library/just_eval_multiscore.jinja',
    # )
    # print(args)
    # main(args)
