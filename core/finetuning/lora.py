from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .data import DataCollator, TrainingDataset, create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_steps: int = 100
    max_length: int = 2048
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    output_dir: str = "data/lora_output"
    save_steps: int = 500
    logging_steps: int = 10
    eval_steps: int = 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> LoRAConfig:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


class LoRATrainer:
    def __init__(
        self,
        model_name: str,
        config: LoRAConfig | None = None,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.config = config or LoRAConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.tokenizer = None
        self.peft_model = None
        self.optimizer = None
        self.scheduler = None

        self.global_step = 0
        self.best_loss = float("inf")
        self.training_history: list[dict] = []

    def setup(self) -> None:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Please install transformers and peft: pip install transformers peft"
            ) from e

        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        task_type_map = {
            "CAUSAL_LM": TaskType.CAUSAL_LM,
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
        }

        peft_config = LoraConfig(
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias=self.config.bias,
            task_type=task_type_map.get(self.config.task_type, TaskType.CAUSAL_LM),
        )

        self.peft_model = get_peft_model(self.model, peft_config)
        self.peft_model.print_trainable_parameters()

        self.optimizer = AdamW(
            self.peft_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def train(
        self,
        train_dataset: TrainingDataset,
        eval_dataset: TrainingDataset | None = None,
        callback: Callable[[dict], None] | None = None,
    ) -> dict[str, Any]:
        if self.peft_model is None:
            self.setup()

        train_dataset.tokenizer = self.tokenizer
        if eval_dataset:
            eval_dataset.tokenizer = self.tokenizer

        collator = DataCollator(self.tokenizer, self.config.max_length)
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collator=collator,
        )

        eval_loader = None
        if eval_dataset:
            eval_loader = create_dataloader(
                eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collator=collator,
            )

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.peft_model.train()
        total_loss = 0.0
        accumulated_loss = 0.0

        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for step, batch in enumerate(train_loader):
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }

                outputs = self.peft_model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                loss.backward()

                accumulated_loss += loss.item()
                total_loss += loss.item()

                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.peft_model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulated_loss / self.config.logging_steps
                        log_data = {
                            "step": self.global_step,
                            "epoch": epoch + 1,
                            "loss": avg_loss,
                        }
                        self.training_history.append(log_data)
                        logger.info(f"Step {self.global_step}: loss={avg_loss:.4f}")

                        if callback:
                            callback(log_data)

                        accumulated_loss = 0.0

                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint(output_dir / f"checkpoint-{self.global_step}")

                    if eval_loader and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self._evaluate(eval_loader)
                        logger.info(f"Eval loss: {eval_loss:.4f}")

                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self._save_checkpoint(output_dir / "best")

                        self.peft_model.train()

        self._save_checkpoint(output_dir / "final")

        return {
            "total_steps": self.global_step,
            "final_loss": total_loss / len(train_loader) / self.config.num_epochs,
            "best_eval_loss": self.best_loss,
            "output_dir": str(output_dir),
        }

    def _evaluate(self, eval_loader: DataLoader) -> float:
        self.peft_model.eval()
        total_loss = 0.0
        total_steps = 0

        with torch.no_grad():
            for batch in eval_loader:
                batch = {
                    k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)
                }
                outputs = self.peft_model(**batch)
                total_loss += outputs.loss.item()
                total_steps += 1

        return total_loss / total_steps if total_steps > 0 else float("inf")

    def _save_checkpoint(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.peft_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.config.save(path / "training_config.json")

        with open(path / "training_state.json", "w") as f:
            json.dump(
                {
                    "global_step": self.global_step,
                    "best_loss": self.best_loss,
                },
                f,
            )

        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path) -> None:
        try:
            from peft import PeftModel
        except ImportError as e:
            raise ImportError("Please install peft: pip install peft") from e

        if self.model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            base_model_name = self.model_name
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )

        self.peft_model = PeftModel.from_pretrained(self.model, path)

        state_file = path / "training_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.best_loss = state.get("best_loss", float("inf"))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        if self.peft_model is None:
            raise RuntimeError("Model not loaded. Call setup() or load_checkpoint() first.")

        self.peft_model.eval()

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.peft_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt) :].strip()
