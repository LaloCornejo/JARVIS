from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class TrainingExample:
    id: str
    conversations: list[ConversationTurn]
    system_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TrainingDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path | None = None,
        examples: list[TrainingExample] | None = None,
        tokenizer: Any = None,
        max_length: int = 2048,
    ):
        self.examples: list[TrainingExample] = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        if data_path:
            self.load_from_file(Path(data_path))
        if examples:
            self.examples.extend(examples)

    def load_from_file(self, path: Path) -> None:
        if path.suffix == ".jsonl":
            self._load_jsonl(path)
        elif path.suffix == ".json":
            self._load_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_jsonl(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(self._parse_example(data))

    def _load_json(self, path: Path) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    self.examples.append(self._parse_example(item))
            else:
                self.examples.append(self._parse_example(data))

    def _parse_example(self, data: dict) -> TrainingExample:
        conversations = []
        for turn in data.get("conversations", data.get("messages", [])):
            conversations.append(
                ConversationTurn(
                    role=turn.get("role", turn.get("from", "user")),
                    content=turn.get("content", turn.get("value", "")),
                )
            )
        return TrainingExample(
            id=data.get("id", str(len(self.examples))),
            conversations=conversations,
            system_prompt=data.get("system", data.get("system_prompt")),
            metadata=data.get("metadata", {}),
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        return self._tokenize(example)

    def _tokenize(self, example: TrainingExample) -> dict[str, Any]:
        if self.tokenizer is None:
            return {"conversations": example.conversations, "id": example.id}

        messages = []
        if example.system_prompt:
            messages.append({"role": "system", "content": example.system_prompt})

        for turn in example.conversations:
            messages.append({"role": turn.role, "content": turn.content})

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    def add_example(self, example: TrainingExample) -> None:
        self.examples.append(example)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for example in self.examples:
                data = {
                    "id": example.id,
                    "conversations": [
                        {"role": t.role, "content": t.content} for t in example.conversations
                    ],
                }
                if example.system_prompt:
                    data["system"] = example.system_prompt
                if example.metadata:
                    data["metadata"] = example.metadata
                f.write(json.dumps(data) + "\n")

    def split(self, train_ratio: float = 0.9) -> tuple[TrainingDataset, TrainingDataset]:
        split_idx = int(len(self.examples) * train_ratio)
        train_examples = self.examples[:split_idx]
        val_examples = self.examples[split_idx:]
        return (
            TrainingDataset(examples=train_examples, tokenizer=self.tokenizer),
            TrainingDataset(examples=val_examples, tokenizer=self.tokenizer),
        )


class DataCollator:
    def __init__(self, tokenizer: Any, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        if "input_ids" not in features[0]:
            return {"features": features}

        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_length,
        )

        input_ids = []
        attention_mask = []
        labels = []

        pad_token_id = self.tokenizer.pad_token_id or 0

        for f in features:
            ids = f["input_ids"][:max_len]
            mask = f.get("attention_mask", [1] * len(ids))[:max_len]
            lbls = f.get("labels", ids)[:max_len]

            padding_len = max_len - len(ids)
            input_ids.append(ids + [pad_token_id] * padding_len)
            attention_mask.append(mask + [0] * padding_len)
            labels.append(lbls + [-100] * padding_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_dataloader(
    dataset: TrainingDataset,
    batch_size: int = 4,
    shuffle: bool = True,
    collator: DataCollator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )
