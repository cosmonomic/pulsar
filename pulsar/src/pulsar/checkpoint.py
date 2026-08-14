import hashlib
import json
import os
import shutil
import zipfile
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Self

import safetensors
import torch
from tokenizers import Tokenizer
from torch import Tensor
from xdg_base_dirs import xdg_cache_home, xdg_data_dirs, xdg_data_home


class PulsarModelLoadingError(RuntimeError):
    pass


@dataclass
class PulsarModelConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_norm_eps: float
    eos_id: int
    max_position_embeddings: int = 0

    @classmethod
    def from_dict(cls, raw: dict) -> Self:
        n_attention_heads = raw["num_attention_heads"]
        return cls(
            vocab_size=raw["vocab_size"],
            hidden_size=raw["hidden_size"],
            intermediate_size=raw["intermediate_size"],
            n_layers=raw["num_hidden_layers"],
            n_heads=n_attention_heads,
            n_kv_heads=raw["num_key_value_heads"],
            head_dim=raw.get("head_dim") or raw["hidden_size"] // n_attention_heads,
            rope_theta=raw["rope_theta"],
            rms_norm_eps=raw["rms_norm_eps"],
            eos_id=raw["eos_token_id"],
            max_position_embeddings=raw.get("max_position_embeddings", 0),
        )


class PulsarModel(NamedTuple):
    config: PulsarModelConfig
    weights: dict[str, Tensor]
    tokenizer: Tokenizer


def iter_data_paths() -> Generator[Path]:
    if path := os.environ.get("PULSAR_DATA_DIR"):
        yield Path(path).expanduser()

    yield xdg_data_home() / "pulsar"

    for path in xdg_data_dirs():
        yield path / "pulsar"


def model_path(model: str) -> Path:
    filename = f"{model}.pulsar"
    for data_path in iter_data_paths():
        if (path := (data_path / filename)).exists():
            return path

    raise PulsarModelLoadingError(f"model {model} not found")


def new_model_path(model: str) -> Path:
    return next(iter_data_paths()) / f"{model}.pulsar"


def load_config(path: zipfile.Path) -> PulsarModelConfig:
    with (path / "config.json").open("r") as f:
        raw = json.load(f)
    return PulsarModelConfig.from_dict(raw)


def load_tokenizer(path: zipfile.Path) -> Tokenizer:
    raw = (path / "tokenizer.json").read_text(encoding="utf-8")
    return Tokenizer.from_str(raw)


def get_shard_cache_dir(archive_path: Path) -> Path:
    "get or create cache dir for shards, including a tmp dir for atomic extraction"
    stat = archive_path.stat()
    key = hashlib.sha256(
        f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()[:16]
    cache_dir = xdg_cache_home() / "pulsar" / "shards" / key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir.resolve()


def cache_shard(shard_path: zipfile.Path, cache_dir: Path) -> Path:
    if (cached_path := cache_dir / shard_path.name).exists():
        return cached_path

    tmp_path = cached_path.with_suffix(cached_path.suffix + ".tmp")

    with (
        shard_path.open("rb") as fsrc,
        tmp_path.open("wb") as fdst,
    ):
        shutil.copyfileobj(fsrc, fdst)

    tmp_path.replace(cached_path)

    return cached_path


def load_weights(
    archive: zipfile.ZipFile,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}

    assert archive.filename is not None
    cache_dir = get_shard_cache_dir(Path(archive.filename))

    for shard_path in zipfile.Path(archive, at="weights/").glob("*.safetensors"):
        cached_shard_path = cache_shard(shard_path, cache_dir)

        with safetensors.safe_open(
            cached_shard_path, framework="pt", device=str(device)
        ) as f:
            for name in f.keys():
                out[name] = f.get_tensor(name).to(dtype=dtype, non_blocking=True)

    return out


def load(
    path: Path,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> PulsarModel:
    with zipfile.ZipFile(path, mode="r") as archive:
        archive_path = zipfile.Path(archive)

        config = load_config(archive_path)
        tokenizer = load_tokenizer(archive_path)
        weights = load_weights(archive, device=device, dtype=dtype)

    return PulsarModel(config=config, weights=weights, tokenizer=tokenizer)
