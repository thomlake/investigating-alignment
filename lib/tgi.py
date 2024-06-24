"""
Python class for

1. Starting a text-generation-inference (TGI) docker container
2. Asynchronous querying of the generate endpoint

# Example

```python
import tgi

server = tgi.Server(model_id='meta-llama/Llama-2-7b-hf')
server.start_container()

inputs = ["inputs 1", "inputs 2", "..."]
responses = server.generate_batch(inputs)
more_responses = server.generate_batch(inputs, top_p=0.9, temperature=0.1)

server.stop_container()
```

file: tgi.py
author: tlake
"""
from __future__ import annotations

import asyncio
import os
import time
from aiohttp import ClientSession
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import docker
from docker.models.containers import Container
from docker.types import DeviceRequest

PORT = 8081


def get_generate_url(port: int = PORT):
    return f'http://127.0.0.1:{port}/generate'


@dataclass
class Parameters:
    """Parameters passed to TGI generate endpoint.

    See: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceClient.text_generation
    """
    details: bool = False
    do_sample: bool = False
    max_new_tokens: int = 128
    best_of: int | None = None
    repetition_penalty: float | None = None
    return_full_text: bool = False
    seed: int | None = None
    stop: list[str] | None = field(default_factory=list)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    truncate: int | None = None
    typical_p: float | None = None
    watermark: bool = False
    decoder_input_details: bool = False
    top_n_tokens: int | None = None
    # Not a TGI parameter
    num_samples: int = 1

    def __post_init__(self):
        if self.temperature is not None and self.temperature <= 0:
            self.temperature = None
        self.do_sample = self.temperature is not None

        if self.top_n_tokens is not None:
            self.details = True

        if self.num_samples > 1 and not self.do_sample:
            raise ValueError(
                'Requested multiple sequences but not sampling! '
                f'(temperature={self.temperature})'
            )


@dataclass
class Server:
    """
    Class-based representation of a TGI docker container.

    At the CLI, the run command used would look something like::

        docker run -d
            --gpus '"all"'
            --shm-size 1g
            -e HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
            -p 8081:80
            -v $VOLUME:/data ghcr.io/huggingface/text-generation-inference:1.0.3
            --model-id meta-llama/Llama-2-7b-hf
            --max-input-length 4095
            --max-total-tokens 4096
            ... (more params)
    """
    volume: str = os.path.abspath('tgi')
    # image: str = 'ghcr.io/huggingface/text-generation-inference:1.3'
    image: str = 'ghcr.io/huggingface/text-generation-inference:1.4'
    detach: bool = True
    shm_size: str = '1g'
    gpus: str = 'all'
    model_id: str = 'meta-llama/Llama-2-7b-hf'
    max_input_length: int = 4095
    max_total_tokens: int = 4096
    max_top_n_tokens: int = 100
    quantize: str = None
    ports: dict[str, int] = field(default_factory=lambda: {'80/tcp': PORT})
    container_name: str = None

    def __post_init__(self):
        self.volume = os.path.abspath(self.volume)
        if self.quantize and not isinstance(self.quantize, str):
            self.quantize = 'bitsandbytes'

    @property
    def generate_url(self):
        return get_generate_url(port=self.ports['80/tcp'])

    @property
    def volumes(self):
        return {os.path.abspath(self.volume): {'bind': '/data', 'mode': 'rw'}}

    @property
    def environment(self):
        env = {}
        token = os.environ.get('HF_TOKEN')
        if token:
            env['HF_TOKEN'] = token
            env['HUGGING_FACE_HUB_TOKEN'] = token

        return env

    @property
    def device_requests(self):
        req = DeviceRequest(device_ids=[self.gpus], capabilities=[['gpu']])
        return [req]

    @property
    def command(self):
        args = []
        args.extend(['--model-id', self.try_convert_path(self.model_id)])
        args.extend(['--max-input-length', str(self.max_input_length)])
        args.extend(['--max-total-tokens', str(self.max_total_tokens)])
        args.extend(['--max-top-n-tokens', str(self.max_top_n_tokens)])
        if self.quantize:
            args.extend(['--quantize', self.quantize])
        return args

    def run_kwargs(self):
        return {
            'image': self.image,
            'command': self.command,
            'volumes': self.volumes,
            'environment': self.environment,
            'device_requests': self.device_requests,
            'ports': self.ports,
            'detach': self.detach,
        }

    def is_valid_path(self, path: str):
        if not os.path.exists(path):
            return False
        path = os.path.abspath(path)
        return path.startswith(self.volume)

    def try_convert_path(self, path: str):
        if self.is_valid_path(path):
            path = self.convert_path(path)
        return path

    def convert_path(self, path: str):
        path = os.path.abspath(path)
        empty, rest = path.split(self.volume)
        assert empty == ''
        rest = rest.strip('/')
        container_root = self.volumes[self.volume]['bind']
        container_path = os.path.join(container_root, rest)
        return container_path

    def start_container(
            self,
            client: docker.client.DockerClient = None,
            max_steps: int = 100,
            sleep_time: int = 5,
            ready_line_ending: str = 'Connected',
            verbose: bool = True
    ) -> Container:
        inner_client = client or docker.from_env()
        run_kwargs = self.run_kwargs()
        container = inner_client.containers.run(**run_kwargs)

        def check_container_state():
            container.reload()
            if container.status == 'exited':
                msg = f'Container "{container.id}" failed to start (exited)'
                raise ValueError(msg)

            log_lines = container.logs().decode().split('\n')
            for line in log_lines:
                line = line.strip()
                if line.endswith(ready_line_ending):
                    return True

            return False

        if verbose:
            msg = f'Waiting for TGI container ({container.name})'
            print(msg, end=' ', flush=True)

        for _ in range(max_steps):
            if check_container_state():
                break  # Success!

            if verbose:
                print('.', end='', flush=True)

            time.sleep(sleep_time)
        else:
            # Failure.
            raise ValueError('Container failed to start (timeout)')

        if verbose:
            print(' done')

        if not client:
            inner_client.close()

        self.container_name = container.name
        return container

    def stop_container(self, client: docker.client.DockerClient = None):
        if self.container_name is None:
            raise ValueError('No container associated with server')

        inner_client = client or docker.from_env()
        container = inner_client.containers.get(self.container_name)
        container.stop()
        if not client:
            inner_client.close()

    def generate(
            self,
            inputs_batch: list[str],
            params: Parameters,
    ) -> list[list[dict[str, Any]]]:
        coroutine = _generate(self.generate_url, inputs_batch, params)
        result = asyncio.run(coroutine)

        groups = defaultdict(list)
        for _id, output in result:
            groups[_id].append(output)

        sorted_groups = sorted(groups.items(), key=lambda kv: kv[0])
        return [group for _, group in sorted_groups]

    def generate_one(
            self,
            inputs: str,
            params: Parameters,
    ) -> list[dict[str, Any]]:
        (result,) = self.generate([inputs], params)
        return result


async def _generate(
        url: str,
        inputs_batch: list[str],
        params: Parameters,
        max_concurrency: int | None = 10,
) -> list[dict[str, Any]]:
    param_dict = asdict(params)
    # Remove num_samples and handle it here.
    # It is not a supported TGI parameter (unfortunately).
    # The functionality is implemented here.
    num_samples = param_dict.pop('num_samples')

    async with ClientSession() as session:
        coros = (
            _generate_one(_id, url, inputs, param_dict, session)
            for _id, inputs in enumerate(inputs_batch)
            for _ in range(num_samples)
        )
        if max_concurrency is not None:
            semaphore = asyncio.Semaphore(max_concurrency)

            async def safe(coro):
                async with semaphore:
                    return await coro

            outputs = await asyncio.gather(*(safe(c) for c in coros))
        else:
            outputs = await asyncio.gather(*coros)

    return outputs


async def _generate_one(
        _id: int,
        url: str,
        inputs: str,
        param_dict: dict[str, Any],
        session: ClientSession,
) -> tuple[int, dict[str, Any]]:
    json = {'inputs': inputs, 'parameters': param_dict}
    try:
        async with session.post(url=url, json=json) as response:
            output = await response.json()
            return _id, output

    except Exception as e:
        print('Request failed!', e)
        raise
