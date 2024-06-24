import random
from dataclasses import dataclass

import numpy as np

from lib.utils import load_instances, load_outputs


class SideInfo:
    def get(self, index: int, k: int | None = None, **kwargs) -> dict:
        raise NotImplementedError()


@dataclass
class EmbeddingArgs:
    embedding_file: str
    instance_file: str
    output_file: str
    k: int
    normalize: bool = True


@dataclass
class EmbeddingSideInfo(SideInfo):
    embeddings: np.ndarray
    objects: list
    args: EmbeddingArgs = None

    def get(self, index: int, k: int | None = None) -> list:
        assert 0 <= index < len(self.objects)
        e = self.embeddings
        v = self.embeddings[index]

        k = k or self.args.k
        i0, *ix = np.argsort(np.dot(e, v))[-(k + 1):][::-1]
        assert i0 == index
        assert len(ix) == k
        return [self.objects[i] for i in ix]

    @staticmethod
    def load(args: EmbeddingArgs):
        npz = np.load(args.embedding_file)
        embeddings = npz['embeddings'].astype(float)
        if args.normalize:
            norms = np.linalg.norm(embeddings, axis=1)
            norms = np.repeat(norms[:, None], embeddings.shape[-1], axis=1)
            embeddings = embeddings / norms
            assert np.allclose(1, np.linalg.norm(embeddings, axis=1))

        input_docs = []
        instances = load_instances(args.instance_file)
        for obj in instances:
            m = obj['messages'][-1]
            assert m['role'] == 'user', m

            d = m['content']
            input_docs.append(d.strip())

        output_docs = []
        outputs = load_outputs(args.output_file)
        for s0, *_ in outputs:
            output_docs.append(s0.strip())

        assert len(input_docs) == len(embeddings)
        assert len(input_docs) == len(output_docs)
        objects = list(zip(input_docs, output_docs))
        return EmbeddingSideInfo(embeddings, objects, args=args)


@dataclass
class RandomArgs:
    instance_file: str
    output_file: str
    k: int


@dataclass
class RandomSideInfo(SideInfo):
    objects: list
    args: RandomArgs = None

    def get(self, index: int, k: int | None = None) -> list:
        assert 0 <= index < len(self.objects)
        k = k or self.args.k
        # Pick one extra so we can drop index if needed
        ix = random.sample(range(len(self.objects)), k=k + 1)
        ix = [i for i in ix if i != index][:k]
        assert len(ix) == k
        return [self.objects[i] for i in ix]

    @staticmethod
    def load(args: RandomArgs):
        input_docs = []
        instances = load_instances(args.instance_file)
        for obj in instances:
            m = obj['messages'][-1]
            assert m['role'] == 'user', m

            d = m['content']
            input_docs.append(d.strip())

        output_docs = []
        outputs = load_outputs(args.output_file)
        for s0, *_ in outputs:
            output_docs.append(s0.strip())

        assert len(input_docs) == len(output_docs)
        objects = list(zip(input_docs, output_docs))
        return RandomSideInfo(objects, args=args)


@dataclass
class SummaryArgs:
    summary_file: str


@dataclass
class SummarySideInfo(SideInfo):
    summaries: list[str]
    args: SummaryArgs = None

    def get(self, index: int) -> list:
        summary = self.summaries[index]
        return [summary]

    @staticmethod
    def load(args: SummaryArgs):
        summaries = [
            s0.strip() for s0, *_ in load_outputs(args.summary_file)
        ]
        return SummarySideInfo(summaries, args=args)
