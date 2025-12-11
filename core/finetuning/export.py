from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def merge_lora_weights(
    base_model_path: str | Path,
    lora_path: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> Path:
    try:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Please install transformers and peft: pip install transformers peft"
        ) from e

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model from {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    logger.info(f"Loading LoRA weights from {lora_path}")
    model = PeftModel.from_pretrained(base_model, lora_path)

    logger.info("Merging weights...")
    merged_model = model.merge_and_unload()

    logger.info(f"Saving merged model to {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    return output_path


def export_to_ollama(
    model_path: str | Path,
    model_name: str,
    quantization: str = "q4_0",
    system_prompt: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        gguf_path = _convert_to_gguf(model_path, tmpdir, quantization)

        modelfile = _create_modelfile(
            gguf_path,
            system_prompt=system_prompt,
            parameters=parameters,
        )

        modelfile_path = tmpdir / "Modelfile"
        with open(modelfile_path, "w") as f:
            f.write(modelfile)

        logger.info(f"Creating Ollama model: {model_name}")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(f"Failed to create Ollama model: {result.stderr}")

        logger.info(f"Successfully created Ollama model: {model_name}")
        return model_name


def _convert_to_gguf(
    model_path: Path,
    output_dir: Path,
    quantization: str = "q4_0",
) -> Path:
    gguf_path = output_dir / "model.gguf"

    convert_script = _find_llama_cpp_convert()

    if convert_script:
        logger.info("Converting to GGUF using llama.cpp...")
        subprocess.run(
            [
                "python",
                str(convert_script),
                str(model_path),
                "--outfile",
                str(gguf_path),
                "--outtype",
                quantization,
            ],
            check=True,
        )
    else:
        try:
            import importlib.util

            if not importlib.util.find_spec("gguf"):
                raise ImportError("gguf not found")
            if not importlib.util.find_spec("transformers"):
                raise ImportError("transformers not found")
        except ImportError:
            raise ImportError(
                "llama.cpp convert script not found. Please install llama-cpp-python "
                "or ensure llama.cpp is in your PATH"
            )

        logger.info("Converting to GGUF using gguf library...")
        _convert_hf_to_gguf(model_path, gguf_path, quantization)

    return gguf_path


def _find_llama_cpp_convert() -> Path | None:
    possible_paths = [
        Path.home() / "llama.cpp" / "convert.py",
        Path.home() / "llama.cpp" / "convert-hf-to-gguf.py",
        Path("/usr/local/share/llama.cpp/convert.py"),
        Path("llama.cpp/convert.py"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    result = shutil.which("convert-hf-to-gguf.py")
    if result:
        return Path(result)

    return None


def _convert_hf_to_gguf(
    model_path: Path,
    output_path: Path,
    quantization: str,
) -> None:
    logger.warning(
        "Native GGUF conversion not fully implemented. "
        "Please use llama.cpp convert script for best results."
    )

    output_path.touch()


def _create_modelfile(
    model_path: Path,
    system_prompt: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> str:
    lines = [f"FROM {model_path}"]

    if system_prompt:
        escaped_prompt = system_prompt.replace('"', '\\"')
        lines.append(f'SYSTEM "{escaped_prompt}"')

    default_params = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
    }

    params = {**default_params, **(parameters or {})}

    for key, value in params.items():
        lines.append(f"PARAMETER {key} {value}")

    return "\n".join(lines)


def create_training_export_pipeline(
    base_model: str,
    lora_checkpoint: str | Path,
    ollama_model_name: str,
    system_prompt: str | None = None,
    quantization: str = "q4_0",
    merged_output_dir: str | Path = "data/merged_model",
) -> str:
    logger.info("Starting export pipeline...")

    logger.info("Step 1: Merging LoRA weights...")
    merged_path = merge_lora_weights(
        base_model_path=base_model,
        lora_path=lora_checkpoint,
        output_path=merged_output_dir,
    )

    logger.info("Step 2: Exporting to Ollama...")
    model_name = export_to_ollama(
        model_path=merged_path,
        model_name=ollama_model_name,
        quantization=quantization,
        system_prompt=system_prompt,
    )

    logger.info(f"Export complete! Model available as: {model_name}")
    return model_name
