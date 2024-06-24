import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from tqdm.auto import tqdm

import create_instances_evaluate_one
import sample_openai
import sample_tgi
from lib import utils


@dataclass
class Model:
    name: str
    id: str | None = None
    chat_template_file: str | None = None
    stop_tokens: str | None = None

    side_info_type: Literal['embedding', 'random', 'summary'] | None = None
    side_info_ref_model: str | None = None
    side_info_embedding_type: str | None = None
    side_info_num_examples: int = 3


# Model handles

GPT_35_TURBO = 'gpt-3.5-turbo'

LLAMA_2_BASE = 'llama-2-base'
LLAMA_2_CHAT = 'llama-2-chat'

LLAMA_2_ICL = 'llama-2-icl'
LLAMA_2_ICL_CHAT = 'llama-2-icl-chat'
LLAMA_2_ICL_CHAT_RAND = 'llama-2-icl-chat-rand'
LLAMA_2_ICL_CHAT_KNN_PROMPT = 'llama-2-icl-chat-knn-prompt_similar'
LLAMA_2_ICL_CHAT_KNN_RESPONSE = 'llama-2-icl-chat-knn-response_similar'
LLAMA_2_ICL_CHAT_KNN_FORMAT = 'llama-2-icl-chat-knn-response_format'

LLAMA_2_ICL_CHAT_SUMMARY_COT = 'llama-2-icl-chat-summary-cot'
LLAMA_2_ICL_CHAT_SUMMARY = 'llama-2-icl-chat-summary'
LLAMA_2_ICL_GPT35_SUMMARY = 'llama-2-icl-gpt-3.5-summary'
LLAMA_2_ICL_CHAT_5NN_PROMPT = 'llama-2-icl-chat-5nn-prompt_similar'

LLAMA_2_ICL_GPT35 = 'llama-2-icl-gpt-3.5'
LLAMA_2_ICL_GPT35_RAND = 'llama-2-icl-gpt-3.5-rand'
LLAMA_2_ICL_GPT35_KNN_PROMPT = 'llama-2-icl-gpt-3.5-knn-prompt_similar'
LLAMA_2_ICL_GPT35_KNN_RESPONSE = 'llama-2-icl-gpt-3.5-knn-response_similar'
LLAMA_2_ICL_GPT35_KNN_FORMAT = 'llama-2-icl-gpt-3.5-knn-response_format'

MISTRAL_BASE = 'mistral'
MISTRAL_INSTRUCT = 'mistral-instruct'
MISTRAL_ICL = 'mistral-icl'
MISTRAL_ICL_INSTRUCT = 'mistral-icl-instruct'
MISTRAL_ICL_INSTRUCT_RAND = 'mistral-icl-instruct-rand'
MISTRAL_ICL_INSTRUCT_KNN_PROMPT = 'mistral-icl-instruct-knn-prompt_similar'
MISTRAL_ICL_INSTRUCT_KNN_RESPONSE = 'mistral-icl-instruct-knn-response_similar'
MISTRAL_ICL_INSTRUCT_SUMMARY = 'mistral-icl-instruct-summary'

MISTRAL_ICL_CHAT = 'mistral-icl-chat'
MISTRAL_ICL_CHAT_RAND = 'mistral-icl-chat-rand'
MISTRAL_ICL_CHAT_SUMMARY = 'mistral-icl-chat-summary'

MISTRAL_ICL_GPT35 = 'mistral-icl-gpt-3.5'
MISTRAL_ICL_GPT35_RAND = 'mistral-icl-gpt-3.5-rand'
MISTRAL_ICL_GPT35_SUMMARY = 'mistral-icl-gpt-3.5-summary'

# HF Model IDs

LLAMA_2_ID = 'meta-llama/Llama-2-7b-hf'
LLAMA_2_CHAT_ID = 'meta-llama/Llama-2-7b-chat-hf'
MISTRAL_ID = 'mistralai/Mistral-7B-v0.1'
MISTRAL_INSTRUCT_ID = 'mistralai/Mistral-7B-Instruct-v0.2'

# Chat templates

URIAL_0_FILE = './lib/template_library/urial_0.jinja'
URIAL_1K_FILE = './lib/template_library/urial_1k.jinja'
URIAL_LLAMA_FILE = './lib/template_library/urial_llama.jinja'
URIAL_GPT35_FILE = './lib/template_library/urial_gpt35.jinja'
URIAL_DYNAMIC_FILE = './lib/template_library/urial_dynamic.jinja'
URIAL_LLAMA_SUMMARY_FILE = './lib/template_library/urial_llama_summary.jinja'
URIAL_LLAMA_SUMMARY_COT_FILE = './lib/template_library/urial_llama_summary_cot.jinja'
URIAL_GPT35_SUMMARY_FILE = './lib/template_library/urial_gpt35_summary.jinja'

URIAL_MISTRAL_FILE = './lib/template_library/urial_mistral.jinja'
URIAL_MISTRAL_SUMMARY_FILE = './lib/template_library/urial_mistral_summary.jinja'


VERBATIM_FILE = './lib/template_library/verbatmin.jinja'
URIAL_1K_INCOMPLETE_FILE = './lib/template_library/urial_1k_incomplete.jinja'
URIAL_LLAMA_INCOMPLETE_FILE = './lib/template_library/urial_llama_incomplete.jinja'
URIAL_MISTRAL_INCOMPLETE_FILE = './lib/template_library/urial_mistral_incomplete.jinja'


URIAL_STOP_TOKENS = ['```']
URIAL_COT_STOP_TOKENS = ['# Query']
LLAMA_STOP_TOKENS = ['</s>']

CONFLICTING_QA = 'ConflictingQA'
LIMA_OE = 'LIMA-OE'

EMBEDDING_SIDE_INFO = 'embedding'
RANDOM_SIDE_INFO = 'random'
SUMMARY_SIDE_INFO = 'summary'

PROMPT_SIMILAR = 'prompt_similar'
RESPONSE_SIMILAR = 'response_similar'
RESPONSE_FORMAT = 'response_format'

DEFAULT_EMBEDDING_K = 3
TEMPERATURE = 0.5
REPETITION_PENALTY = 1.1

MODELS_MAP: dict[str, Model] = {
    # Base models
    LLAMA_2_BASE: Model(
        name=LLAMA_2_BASE,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_0_FILE,
        stop_tokens=URIAL_STOP_TOKENS
    ),
    LLAMA_2_CHAT: Model(
        name=LLAMA_2_CHAT,
        id=LLAMA_2_CHAT_ID,
        stop_tokens=LLAMA_STOP_TOKENS,
    ),
    GPT_35_TURBO: Model(
        GPT_35_TURBO,
    ),

    # Human responses
    LLAMA_2_ICL: Model(
        name=LLAMA_2_ICL,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_1K_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),

    # Llama Chat Teacher
    LLAMA_2_ICL_CHAT: Model(
        name=LLAMA_2_ICL_CHAT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_LLAMA_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    LLAMA_2_ICL_CHAT_RAND: Model(
        name=LLAMA_2_ICL_CHAT_RAND,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=RANDOM_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
    ),
    LLAMA_2_ICL_CHAT_KNN_PROMPT: Model(
        name=LLAMA_2_ICL_CHAT_KNN_PROMPT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
        side_info_embedding_type=PROMPT_SIMILAR,
    ),
    LLAMA_2_ICL_CHAT_KNN_RESPONSE: Model(
        name=LLAMA_2_ICL_CHAT_KNN_RESPONSE,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
        side_info_embedding_type=RESPONSE_SIMILAR,
    ),
    LLAMA_2_ICL_CHAT_KNN_FORMAT: Model(
        name=LLAMA_2_ICL_CHAT_KNN_FORMAT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
        side_info_embedding_type=RESPONSE_FORMAT,
    ),
    LLAMA_2_ICL_CHAT_5NN_PROMPT: Model(
        name=LLAMA_2_ICL_CHAT_5NN_PROMPT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
        side_info_embedding_type=PROMPT_SIMILAR,
        side_info_num_examples=5,
    ),
    LLAMA_2_ICL_CHAT_SUMMARY: Model(
        name=LLAMA_2_ICL_CHAT_SUMMARY,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_LLAMA_SUMMARY_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=SUMMARY_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
    ),
    LLAMA_2_ICL_CHAT_SUMMARY_COT: Model(
        name=LLAMA_2_ICL_CHAT_SUMMARY_COT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_LLAMA_SUMMARY_COT_FILE,
        stop_tokens=URIAL_COT_STOP_TOKENS,
    ),

    # GPT-3.5 Teacher
    LLAMA_2_ICL_GPT35: Model(
        name=LLAMA_2_ICL_GPT35,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_GPT35_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    LLAMA_2_ICL_GPT35_RAND: Model(
        name=LLAMA_2_ICL_GPT35_RAND,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=RANDOM_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
    ),
    LLAMA_2_ICL_GPT35_KNN_PROMPT: Model(
        name=LLAMA_2_ICL_GPT35_KNN_PROMPT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
        side_info_embedding_type=PROMPT_SIMILAR,
    ),
    LLAMA_2_ICL_GPT35_KNN_RESPONSE: Model(
        name=LLAMA_2_ICL_GPT35_KNN_RESPONSE,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
        side_info_embedding_type=RESPONSE_SIMILAR,
    ),
    LLAMA_2_ICL_GPT35_KNN_FORMAT: Model(
        name=LLAMA_2_ICL_GPT35_KNN_FORMAT,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
        side_info_embedding_type=RESPONSE_FORMAT,
    ),
    LLAMA_2_ICL_GPT35_SUMMARY: Model(
        name=LLAMA_2_ICL_GPT35_SUMMARY,
        id=LLAMA_2_ID,
        chat_template_file=URIAL_GPT35_SUMMARY_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=SUMMARY_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
    ),
    # Mistral
    MISTRAL_ICL_CHAT: Model(
        name=MISTRAL_ICL_CHAT,
        id=MISTRAL_ID,
        chat_template_file=URIAL_LLAMA_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    MISTRAL_ICL_CHAT_RAND: Model(
        name=MISTRAL_ICL_CHAT_RAND,
        id=MISTRAL_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=RANDOM_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
    ),
    MISTRAL_ICL_CHAT_SUMMARY: Model(
        name=MISTRAL_ICL_CHAT_SUMMARY,
        id=MISTRAL_ID,
        chat_template_file=URIAL_LLAMA_SUMMARY_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=SUMMARY_SIDE_INFO,
        side_info_ref_model=LLAMA_2_CHAT,
    ),
    MISTRAL_ICL_GPT35: Model(
        name=MISTRAL_ICL_GPT35,
        id=MISTRAL_ID,
        chat_template_file=URIAL_GPT35_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    MISTRAL_ICL_GPT35_RAND: Model(
        name=MISTRAL_ICL_GPT35_RAND,
        id=MISTRAL_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=RANDOM_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
    ),
    MISTRAL_ICL_GPT35_SUMMARY: Model(
        name=MISTRAL_ICL_GPT35_SUMMARY,
        id=MISTRAL_ID,
        chat_template_file=URIAL_GPT35_SUMMARY_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=SUMMARY_SIDE_INFO,
        side_info_ref_model=GPT_35_TURBO,
    ),

    # Mistral Mistral
    MISTRAL_BASE: Model(
        name=MISTRAL_BASE,
        id=MISTRAL_ID,
        chat_template_file=URIAL_0_FILE,
        stop_tokens=URIAL_STOP_TOKENS
    ),

    MISTRAL_INSTRUCT: Model(
        name=MISTRAL_INSTRUCT,
        id=MISTRAL_INSTRUCT_ID,
        stop_tokens=LLAMA_STOP_TOKENS,
    ),
    MISTRAL_ICL: Model(
        name=MISTRAL_ICL,
        id=MISTRAL_ID,
        chat_template_file=URIAL_1K_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    MISTRAL_ICL_INSTRUCT: Model(
        name=MISTRAL_ICL_INSTRUCT,
        id=MISTRAL_ID,
        chat_template_file=URIAL_MISTRAL_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
    ),
    MISTRAL_ICL_INSTRUCT_RAND: Model(
        name=MISTRAL_ICL_INSTRUCT_RAND,
        id=MISTRAL_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=RANDOM_SIDE_INFO,
        side_info_ref_model=MISTRAL_INSTRUCT,
    ),

    MISTRAL_ICL_INSTRUCT_KNN_PROMPT: Model(
        name=MISTRAL_ICL_INSTRUCT_KNN_PROMPT,
        id=MISTRAL_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=MISTRAL_INSTRUCT,
        side_info_embedding_type=PROMPT_SIMILAR,
    ),
    MISTRAL_ICL_INSTRUCT_KNN_RESPONSE: Model(
        name=MISTRAL_ICL_INSTRUCT_KNN_RESPONSE,
        id=MISTRAL_ID,
        chat_template_file=URIAL_DYNAMIC_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=EMBEDDING_SIDE_INFO,
        side_info_ref_model=MISTRAL_INSTRUCT,
        side_info_embedding_type=RESPONSE_SIMILAR,
    ),
    MISTRAL_ICL_INSTRUCT_SUMMARY: Model(
        name=MISTRAL_ICL_INSTRUCT_SUMMARY,
        id=MISTRAL_ID,
        chat_template_file=URIAL_MISTRAL_SUMMARY_FILE,
        stop_tokens=URIAL_STOP_TOKENS,
        side_info_type=SUMMARY_SIDE_INFO,
        side_info_ref_model=MISTRAL_INSTRUCT,
    ),
}


def conflicting_qa_create_instances():
    output_file = './data/ConflictingQA/instances/data.jsonl'
    if Path(output_file).exists():
        print('nothing to do')
        return

    df = pd.read_pickle('/Users/tlake/ut/data/ConflictingQA/data.pkl')
    queries = df.search_query.unique()
    chats = []
    for q in queries:
        m = {'role': 'user', 'content': q}
        chat = {'messages': [m]}
        chats.append(chat)

    with open(output_file, 'w') as fp:
        for chat in chats:
            print(json.dumps(chat, ensure_ascii=False), file=fp)


def generate_outputs_tgi():
    data_model_pairs = [
        (CONFLICTING_QA, LLAMA_2_BASE),
        (CONFLICTING_QA, LLAMA_2_ICL),
        (CONFLICTING_QA, LLAMA_2_CHAT),

        (CONFLICTING_QA, LLAMA_2_ICL_CHAT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_SUMMARY),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_RAND),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_5NN_PROMPT),

        (CONFLICTING_QA, LLAMA_2_ICL_GPT35),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_SUMMARY),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_RAND),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_RESPONSE),

        (CONFLICTING_QA, MISTRAL_BASE),
        (CONFLICTING_QA, MISTRAL_INSTRUCT),
        (CONFLICTING_QA, MISTRAL_ICL),

        (CONFLICTING_QA, MISTRAL_ICL_INSTRUCT),
        (CONFLICTING_QA, MISTRAL_ICL_INSTRUCT_RAND),

        (CONFLICTING_QA, MISTRAL_ICL_INSTRUCT_KNN_PROMPT),
        (CONFLICTING_QA, MISTRAL_ICL_INSTRUCT_KNN_RESPONSE),
        (CONFLICTING_QA, MISTRAL_ICL_INSTRUCT_SUMMARY),

        (LIMA_OE, MISTRAL_BASE),
        (LIMA_OE, MISTRAL_INSTRUCT),
        (LIMA_OE, MISTRAL_ICL),

        (LIMA_OE, MISTRAL_ICL_INSTRUCT),
        (LIMA_OE, MISTRAL_ICL_INSTRUCT_RAND),

        (LIMA_OE, MISTRAL_ICL_INSTRUCT_KNN_PROMPT),
        (LIMA_OE, MISTRAL_ICL_INSTRUCT_KNN_RESPONSE),
        (LIMA_OE, MISTRAL_ICL_INSTRUCT_SUMMARY),

        (LIMA_OE, LLAMA_2_BASE),
        (LIMA_OE, LLAMA_2_ICL),
        (LIMA_OE, LLAMA_2_CHAT),

        (LIMA_OE, LLAMA_2_ICL_CHAT_SUMMARY_COT),

        (LIMA_OE, LLAMA_2_ICL_CHAT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_SUMMARY),
        (LIMA_OE, LLAMA_2_ICL_CHAT_RAND),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (LIMA_OE, LLAMA_2_ICL_CHAT_5NN_PROMPT),

        (LIMA_OE, LLAMA_2_ICL_GPT35),
        (LIMA_OE, LLAMA_2_ICL_GPT35_SUMMARY),
        (LIMA_OE, LLAMA_2_ICL_GPT35_RAND),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_RESPONSE),

        (LIMA_OE, MISTRAL_ICL_CHAT),
        (LIMA_OE, MISTRAL_ICL_CHAT_RAND),
        (LIMA_OE, MISTRAL_ICL_CHAT_SUMMARY),
        (LIMA_OE, MISTRAL_ICL_GPT35),
        (LIMA_OE, MISTRAL_ICL_GPT35_RAND),
        (LIMA_OE, MISTRAL_ICL_GPT35_SUMMARY),
    ]
    total = len(data_model_pairs)

    greedy = False
    output_handle = 'outputs-temp0' if greedy else 'outputs'

    for i, (data_name, model_name) in enumerate(data_model_pairs):
        model = MODELS_MAP[model_name]

        if greedy:
            params = sample_tgi.Parameters(
                temperature=0,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=700,
                num_samples=1,
                stop=model.stop_tokens,
            )
        else:
            params = sample_tgi.Parameters(
                temperature=TEMPERATURE,
                repetition_penalty=REPETITION_PENALTY,
                max_new_tokens=700,
                num_samples=5,
                stop=model.stop_tokens,
            )

        run = sample_tgi.Run(
            instance_file=f'./data/{data_name}/instances/data.jsonl',
            output_dir=f'./data/{data_name}/{output_handle}/{model_name}',
            chat_template_file=model.chat_template_file,
            name=model_name,
        )

        server = sample_tgi.Server(model_id=model.id)

        side_info_args = None
        if model.side_info_type is None:
            pass
        elif model.side_info_type == RANDOM_SIDE_INFO:
            ref_model = model.side_info_ref_model
            num_examples = model.side_info_num_examples

            side_info_args = sample_tgi.RandomArgs(
                instance_file=f'./data/{data_name}/instances/data.jsonl',
                output_file=f'./data/{data_name}/outputs/{ref_model}/samples.jsonl',
                k=num_examples,
            )
        elif model.side_info_type == SUMMARY_SIDE_INFO:
            ref_model = model.side_info_ref_model
            side_info_args = sample_tgi.SummaryArgs(
                summary_file=f'./data/{data_name}/summaries-{output_handle}/outputs/{ref_model}/samples.jsonl',
            )
        elif model.side_info_type == EMBEDDING_SIDE_INFO:
            ref_model = model.side_info_ref_model
            embedding_type = model.side_info_embedding_type
            num_examples = model.side_info_num_examples

            embedding_file = ''
            if embedding_type == PROMPT_SIMILAR:
                embedding_file = f'./data/{data_name}/instances/embeddings-{embedding_type}.npz'
            elif embedding_type in {RESPONSE_SIMILAR, RESPONSE_FORMAT}:
                embedding_file = f'./data/{data_name}/outputs/{ref_model}/embeddings-{embedding_type}.npz'
            else:
                raise ValueError(f'Unknown embedding_type: "{model.side_info_embedding_type}"')

            side_info_args = sample_tgi.EmbeddingArgs(
                embedding_file=embedding_file,
                instance_file=f'./data/{data_name}/instances/data.jsonl',
                output_file=f'./data/{data_name}/outputs/{ref_model}/samples.jsonl',
                k=num_examples,
                normalize=True,
            )
        else:
            raise ValueError(f'Unknown side_info_type: "{model.side_info_type}"')

        args = sample_tgi.Args(
            run=run,
            server=server,
            params=params,
            side_info_args=side_info_args,
        )
        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_tgi.run(args)


def generate_outputs_openai():
    gpt_35_turbo_version = 'gpt-3.5-turbo-0613'
    greedy = False
    output_handle = 'outputs-temp0' if greedy else 'outputs'

    data_model_pairs = [
        (CONFLICTING_QA, GPT_35_TURBO),
        (LIMA_OE, GPT_35_TURBO),
    ]
    total = len(data_model_pairs)
    for i, (data_name, model_name) in enumerate(data_model_pairs):
        if greedy:
            args = sample_openai.Args(
                run=sample_openai.Run(
                    instance_file=f'./data/{data_name}/instances/data.jsonl',
                    output_dir=f'./data/{data_name}/{output_handle}/{model_name}',
                ),
                openai_params=sample_openai.OpenAIParams(
                    model=gpt_35_turbo_version,
                    temperature=0,
                    max_tokens=700,
                    n=1,
                ),
            )
        else:
            args = sample_openai.Args(
                run=sample_openai.Run(
                    instance_file=f'./data/{data_name}/instances/data.jsonl',
                    output_dir=f'./data/{data_name}/{output_handle}/{model_name}',
                ),
                openai_params=sample_openai.OpenAIParams(
                    model=gpt_35_turbo_version,
                    temperature=TEMPERATURE,
                    max_tokens=700,
                    n=5,
                ),
            )

        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_openai.run(args)


def generate_outputs_openai_with_summary():
    gpt_35_turbo_version = 'gpt-3.5-turbo-0613'

    data_model_pairs = [
        (CONFLICTING_QA, GPT_35_TURBO),
        (LIMA_OE, GPT_35_TURBO, LLAMA_2_CHAT),
    ]
    total = len(data_model_pairs)
    for i, (data_name, model_name, summary_model) in enumerate(data_model_pairs):
        args = sample_openai.Args(
            # Sample
            run=sample_openai.Run(
                instance_file=f'./data/{data_name}/instances/data.jsonl',
                output_dir=f'./data/{data_name}/outputs/{model_name}-summary-{summary_model}',
                user_template_file='./lib/template_library/answer_with_summary.jinja',
            ),
            openai_params=sample_openai.OpenAIParams(
                model=gpt_35_turbo_version,
                temperature=TEMPERATURE,
                max_tokens=700,
                n=5,
            ),
            side_info_args=sample_openai.SummaryArgs(
                summary_file=f'./data/{data_name}/summaries-outputs/outputs/{summary_model}/samples.jsonl'
            ),

            # Greedy
            # run=sample_openai.Run(
            #     instance_file=f'./data/{data_name}/instances/data.jsonl',
            #     output_dir=f'./data/{data_name}/outputs-temp0/{model_name}-summary-{summary_model}',
            #     user_template_file='./lib/template_library/answer_with_summary.jinja',
            # ),
            # openai_params=sample_openai.OpenAIParams(
            #     model=gpt_35_turbo_version,
            #     temperature=0,
            #     max_tokens=700,
            #     n=1,
            # ),
            # side_info_args=sample_openai.SummaryArgs(
            #     summary_file=f'./data/{data_name}/summaries-outputs-temp0/outputs/{summary_model}/samples.jsonl'
            # ),
        )

        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_openai.run(args)


def generate_embeddings():
    import create_embeddings

    data_jobs = [
        (CONFLICTING_QA, PROMPT_SIMILAR),
        (CONFLICTING_QA, RESPONSE_SIMILAR),
        (LIMA_OE, PROMPT_SIMILAR),
        (LIMA_OE, RESPONSE_SIMILAR),
    ]
    for data_name, input_type in data_jobs:
        args = create_embeddings.Args(
            embedding_file=f'./data/{data_name}/instances/embeddings-{input_type}.npz',
            input_file=f'./data/{data_name}/instances/data.jsonl',
            input_type=input_type,
        )
        create_embeddings.run(args)

    model_jobs = [
        (CONFLICTING_QA, LLAMA_2_CHAT, PROMPT_SIMILAR),
        (CONFLICTING_QA, GPT_35_TURBO, PROMPT_SIMILAR),
        (CONFLICTING_QA, MISTRAL_INSTRUCT, PROMPT_SIMILAR),
        (CONFLICTING_QA, LLAMA_2_CHAT, PROMPT_SIMILAR),
        (CONFLICTING_QA, GPT_35_TURBO, PROMPT_SIMILAR),
        (CONFLICTING_QA, MISTRAL_INSTRUCT, PROMPT_SIMILAR),

        (LIMA_OE, LLAMA_2_CHAT, RESPONSE_SIMILAR),
        (LIMA_OE, GPT_35_TURBO, RESPONSE_SIMILAR),
        (LIMA_OE, MISTRAL_INSTRUCT, RESPONSE_SIMILAR),
        (LIMA_OE, LLAMA_2_CHAT, RESPONSE_SIMILAR),
        (LIMA_OE, GPT_35_TURBO, RESPONSE_SIMILAR),
        (LIMA_OE, MISTRAL_INSTRUCT, RESPONSE_SIMILAR),
    ]
    for data_name, model_name, input_type in model_jobs:
        args = create_embeddings.Args(
            embedding_file=f'./data/{data_name}/outputs/{model_name}/embeddings-{input_type}.npz',
            input_file=f'./data/{data_name}/outputs/{model_name}/samples.jsonl',
            input_type=input_type,
            batch_size=1,
        )
        create_embeddings.run(args)


def create_berstscore_pairs():
    ref_models = [
        LLAMA_2_CHAT,
        GPT_35_TURBO,
    ]

    alt_models = [
        LLAMA_2_CHAT,
        GPT_35_TURBO,
        LLAMA_2_BASE,
        LLAMA_2_ICL,
        LLAMA_2_ICL_CHAT,
        LLAMA_2_ICL_CHAT_KNN_PROMPT,
        LLAMA_2_ICL_GPT35,
        LLAMA_2_ICL_GPT35_RAND,
        LLAMA_2_ICL_GPT35_KNN_PROMPT,
    ]

    for data in [CONFLICTING_QA, LIMA_OE]:
        ref_model_output_map = {}
        for model in ref_models:
            with open(f'./data/{data}/outputs/{model}/samples.jsonl') as fp:
                objs = [json.loads(line) for line in fp]

            output_map = {}
            for instance_id, obj in enumerate(objs):
                try:
                    output, *_ = utils.extract_outputs(obj)
                except ValueError:
                    continue

                output_map[instance_id] = output.strip()

            ref_model_output_map[model] = output_map

        alt_model_output_map = {}
        for model in alt_models:
            with open(f'./data/{data}/outputs/{model}/samples.jsonl') as fp:
                objs = [json.loads(line) for line in fp]

            output_map = {}
            for instance_id, obj in enumerate(objs):
                try:
                    outputs = utils.extract_outputs(obj)
                except ValueError:
                    continue

                for sample_id, output in enumerate(outputs):
                    output_map[instance_id, sample_id] = output.strip()

            alt_model_output_map[model] = output_map

        for ref in ref_models:
            for alt in alt_models:
                path = Path(f'./data/{data}/bertscore/inputs')
                path.mkdir(parents=True, exist_ok=True)
                file = path / f'{ref}--{alt}.json'
                if file.exists():
                    print(f'exists: {file}')
                    continue

                instance_ids = []
                ref_sample_ids = []
                alt_sample_ids = []
                ref_texts = []
                alt_texts = []
                ref_output_map = ref_model_output_map[ref]
                alt_output_map = alt_model_output_map[alt]
                for (instance_id, sample_id), alt_text in alt_output_map.items():
                    ref_text = ref_output_map.get(instance_id)
                    if ref_text is not None:
                        instance_ids.append(instance_id)
                        ref_sample_ids.append(0)
                        ref_texts.append(ref_text)
                        alt_sample_ids.append(sample_id)
                        alt_texts.append(alt_text)

                out = {
                    'ref_model': ref,
                    'alt_model': alt,
                    'instance_ids': instance_ids,
                    'ref_sample_ids': ref_sample_ids,
                    'alt_sample_ids': alt_sample_ids,
                    'ref_texts': ref_texts,
                    'alt_texts': alt_texts,
                }
                with open(file, 'w') as fp:
                    json.dump(out, fp)


def generate_bertscores():
    import evaluate
    bertscore = evaluate.load('bertscore')

    for data in [CONFLICTING_QA, LIMA_OE]:
        print(f'running on data: {data}')
        input_dir = Path(f'./data/{data}/bertscore/inputs')
        input_files = list(input_dir.glob('*.json'))
        for input_file in tqdm(input_files):
            output_file = input_dir.parent / 'outputs' / input_file.name
            if output_file.exists():
                print(f'exists: {output_file}')
                continue

            with open(input_file) as fp:
                obj = json.load(fp)

            predictions = obj['alt_texts']
            references = obj['ref_texts']
            results = bertscore.compute(predictions=predictions, references=references, lang='en')
            obj['results'] = results

            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as fp:
                json.dump(obj, fp)


def create_summarization_data():
    data_model_pairs = [
        (CONFLICTING_QA, LLAMA_2_ICL),
        (CONFLICTING_QA, LLAMA_2_CHAT),
        (CONFLICTING_QA, GPT_35_TURBO),
        (CONFLICTING_QA, MISTRAL_INSTRUCT),

        (LIMA_OE, LLAMA_2_ICL),
        (LIMA_OE, LLAMA_2_CHAT),
        (LIMA_OE, GPT_35_TURBO),
        (LIMA_OE, MISTRAL_INSTRUCT),
    ]

    output_handle = 'outputs'
    # output_handle = 'outputs-temp0'

    for data_name, model_name in data_model_pairs:
        data_args = create_instances_evaluate_one.Args(
            result_dir=f'./data/{data_name}/summaries-{output_handle}/instances/{model_name}',
            instance_file=f'./data/{data_name}/instances/data.jsonl',
            output_file=f'./data/{data_name}/{output_handle}/{model_name}/samples.jsonl',
            template_file='./lib/template_library/summarize_response.jinja',
            sample_index=0,
        )
        create_instances_evaluate_one.run(args=data_args)


def summarize_outputs():
    data_model_pairs = [
        (CONFLICTING_QA, LLAMA_2_ICL),
        (CONFLICTING_QA, LLAMA_2_CHAT),
        (CONFLICTING_QA, GPT_35_TURBO),
        (CONFLICTING_QA, MISTRAL_INSTRUCT),
        (LIMA_OE, LLAMA_2_ICL),
        (LIMA_OE, LLAMA_2_CHAT),
        (LIMA_OE, GPT_35_TURBO),
        (LIMA_OE, MISTRAL_INSTRUCT),
    ]
    total = len(data_model_pairs)
    output_handle = 'outputs'
    # output_handle = 'outputs-temp0'

    for i, (data_name, model_name) in enumerate(data_model_pairs):
        sample_args = sample_openai.Args(
            run=sample_openai.Run(
                instance_file=f'./data/{data_name}/summaries-{output_handle}/instances/{model_name}/data.jsonl',
                output_dir=f'./data/{data_name}/summaries-{output_handle}/outputs/{model_name}',
            ),
            openai_params=sample_openai.OpenAIParams(
                model='gpt-4-1106-preview',
                temperature=0,
                n=1,
                max_tokens=256,
            )
        )
        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_openai.run(args=sample_args)


def create_eval_quality_data():
    data_model_pairs = [
        (CONFLICTING_QA, LLAMA_2_BASE),
        (CONFLICTING_QA, LLAMA_2_ICL),
        (CONFLICTING_QA, LLAMA_2_CHAT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_SUMMARY),

        (CONFLICTING_QA, GPT_35_TURBO),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_RESPONSE),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_SUMMARY),

        (LIMA_OE, LLAMA_2_BASE),
        (LIMA_OE, LLAMA_2_ICL),
        (LIMA_OE, LLAMA_2_CHAT),
        (LIMA_OE, LLAMA_2_ICL_CHAT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (LIMA_OE, LLAMA_2_ICL_CHAT_SUMMARY),

        (LIMA_OE, GPT_35_TURBO),
        (LIMA_OE, LLAMA_2_ICL_GPT35),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_RESPONSE),
        (LIMA_OE, LLAMA_2_ICL_GPT35_SUMMARY),
    ]

    for data_name, model_name in data_model_pairs:
        data_args = create_instances_evaluate_one.Args(
            result_dir=f'./data/{data_name}/eval-quality/instances/{model_name}',
            instance_file=f'./data/{data_name}/instances/data.jsonl',
            output_file=f'./data/{data_name}/outputs/{model_name}/samples.jsonl',
            template_file='./lib/template_library/just_eval_multiscore.jinja',
            sample_index=0,
        )
        create_instances_evaluate_one.run(args=data_args)


def eval_quality():
    data_model_pairs = [
        (CONFLICTING_QA, LLAMA_2_BASE),
        (CONFLICTING_QA, LLAMA_2_ICL),
        (CONFLICTING_QA, LLAMA_2_CHAT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (CONFLICTING_QA, LLAMA_2_ICL_CHAT_SUMMARY),

        (CONFLICTING_QA, GPT_35_TURBO),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_KNN_RESPONSE),
        (CONFLICTING_QA, LLAMA_2_ICL_GPT35_SUMMARY),

        (LIMA_OE, LLAMA_2_BASE),
        (LIMA_OE, LLAMA_2_ICL),
        (LIMA_OE, LLAMA_2_CHAT),
        (LIMA_OE, LLAMA_2_ICL_CHAT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_CHAT_KNN_RESPONSE),
        (LIMA_OE, LLAMA_2_ICL_CHAT_SUMMARY),

        (LIMA_OE, GPT_35_TURBO),
        (LIMA_OE, LLAMA_2_ICL_GPT35),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_PROMPT),
        (LIMA_OE, LLAMA_2_ICL_GPT35_KNN_RESPONSE),
        (LIMA_OE, LLAMA_2_ICL_GPT35_SUMMARY),
    ]
    total = len(data_model_pairs)

    for i, (data_name, model_name) in enumerate(data_model_pairs):
        sample_args = sample_openai.Args(
            run=sample_openai.Run(
                instance_file=f'./data/{data_name}/eval-quality/instances/{model_name}/data.jsonl',
                output_dir=f'./data/{data_name}/eval-quality/outputs/{model_name}',
            ),
            openai_params=sample_openai.OpenAIParams(
                model='gpt-4-1106-preview',
                temperature=0,
                response_format='json_object',
                n=1,
                max_tokens=1024,
            )
        )
        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_openai.run(args=sample_args)


def conflicting_qa_create_eval_stance_data():
    model_names = [
        LLAMA_2_BASE,
        LLAMA_2_ICL,
        LLAMA_2_CHAT,
        LLAMA_2_ICL_CHAT,
        LLAMA_2_ICL_CHAT_KNN_PROMPT,
        LLAMA_2_ICL_CHAT_KNN_RESPONSE,
        LLAMA_2_ICL_GPT35,
        LLAMA_2_ICL_GPT35_KNN_PROMPT,
        LLAMA_2_ICL_GPT35_KNN_RESPONSE,
        GPT_35_TURBO,
    ]

    for model_name in model_names:
        data_args = create_instances_evaluate_one.Args(
            result_dir=f'./data/ConflictingQA/eval-stance/instances/{model_name}',
            instance_file='./data/ConflictingQA/instances/data.jsonl',
            output_file=f'./data/ConflictingQA/outputs/{model_name}/samples.jsonl',
            template_file='./lib/template_library/evaluate_stance.jinja',
            sample_index=None,
        )
        create_instances_evaluate_one.run(args=data_args)


def conflicting_qa_eval_stance():
    model_names = [
        LLAMA_2_BASE,
        LLAMA_2_ICL,
        LLAMA_2_CHAT,
        LLAMA_2_ICL_CHAT,
        LLAMA_2_ICL_CHAT_KNN_PROMPT,
        LLAMA_2_ICL_CHAT_KNN_RESPONSE,
        LLAMA_2_ICL_GPT35,
        LLAMA_2_ICL_GPT35_KNN_PROMPT,
        LLAMA_2_ICL_GPT35_KNN_RESPONSE,
        GPT_35_TURBO,
    ]
    total = len(model_names)

    for i, model_name in enumerate(model_names):
        sample_args = sample_openai.Args(
            run=sample_openai.Run(
                instance_file=f'./data/ConflictingQA/eval-stance/instances/{model_name}/data.jsonl',
                output_dir=f'./data/ConflictingQA/eval-stance/outputs/{model_name}',
            ),
            openai_params=sample_openai.OpenAIParams(
                model='gpt-4-1106-preview',
                temperature=0,
                response_format='json_object',
                n=1,
                max_tokens=1024,
            )
        )
        print(f'running job {i+1} of {total} ({i/total:.3%} complete)')
        sample_openai.run(args=sample_args)
