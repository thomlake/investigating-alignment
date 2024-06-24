from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ID = '_record_id'


class JsonRecords:
    data: list[dict[str, Any]]
    file: str | None

    def __init__(self, data: list[dict[str, Any]], file: str = None) -> None:
        self.data = data or []
        self.file = file

    def __len__(self) -> int:
        return len(self.data)

    def get(self, i: int) -> dict[str, Any]:
        return self.data[i]

    def set(self, i: int, d: dict[str, Any]) -> None:
        self.data[i] = d

    def append(self, d: dict[str, Any]) -> None:
        _id = len(self._data)
        d[ID] = _id
        self.data.append(d)

    def save(self, file: str | None = None):
        file = file or self.file
        if file is None:
            raise ValueError('no file set for records')

        Path(file).parent.mkdir(parents=True, exist_ok=True)
        with open(file, 'w') as fp:
            for d in self.data:
                print(json.dumps(d, ensure_ascii=False), file=fp)

    @staticmethod
    def load(file: str):
        with open(file) as fp:
            data = [json.loads(line) for line in fp]

        return JsonRecords(data, file=file)

    @staticmethod
    def from_jsonlines(file: str):
        records = JsonRecords.load(file)
        _add_ids(records.data)
        return records

    @staticmethod
    def from_data(
            data: dict[str, Any],
            file: str | None = None,
            add_id: str = True,
    ):
        if add_id:
            _add_ids(data)

        return JsonRecords(data, file=file)


def _add_ids(data: list[dict[str, Any]]) -> None:
    for i, d in enumerate(data):
        d[ID] = i


def remap(d):
    d_new = {}
    d_new['_record_id'] = d['_record_id']
    d_new['instance'] = {'messages': d['chat']}
    d_new['personas'] = {
        'ok': d['personas']['ok'],
        'raw': d['personas']['output'],
        'descriptions': d['personas']['parsed_output'],
    }
