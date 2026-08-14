"""Pack a downloaded HuggingFace-layout checkpoint into a Pulsar model archive.

Reads::

    <src>/config.json
    <src>/tokenizer.json
    <src>/model.safetensors                    (or a sharded model-0000N-of-*.safetensors
                                                  set named by model.safetensors.index.json)
    <src>/generation_config.json                (optional)

and writes a single zip archive of config.json, tokenizer.json and
weights/*.safetensors; see pulsar.checkpoint for how that archive is read back.

eos_token_id is resolved once here, from generation_config.json falling back to
config.json (Qwen ships it as a list; the first entry is taken), and written into
the archived config.json as a plain int; the archive never carries
generation_config.json itself.
"""

import argparse
import json
import zipfile
from pathlib import Path

from pulsar.checkpoint import PulsarModelConfig, new_model_path


def resolve_eos(src: Path, cfg_raw: dict) -> int:
    gen_path = src / "generation_config.json"
    gen_raw = json.loads(gen_path.read_text()) if gen_path.exists() else cfg_raw
    eos = gen_raw.get("eos_token_id", cfg_raw.get("eos_token_id", -1))
    if isinstance(eos, list):
        eos = eos[0] if eos else -1
    return int(eos)


def weight_shards(src: Path) -> list[Path]:
    single = src / "model.safetensors"
    if single.exists():
        return [single]
    index = src / "model.safetensors.index.json"
    if not index.exists():
        raise FileNotFoundError(
            f"no model.safetensors or model.safetensors.index.json under {src}"
        )
    weight_map = json.loads(index.read_text())["weight_map"]
    return [src / shard for shard in sorted(set(weight_map.values()))]


def migrate(src: Path, dst: Path) -> None:
    cfg_raw = json.loads((src / "config.json").read_text())
    cfg_raw["eos_token_id"] = resolve_eos(src, cfg_raw)
    PulsarModelConfig.from_dict(cfg_raw)  # validate shape before writing

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        zf.writestr("config.json", json.dumps(cfg_raw))
        zf.write(src / "tokenizer.json", "tokenizer.json")
        for shard in weight_shards(src):
            zf.write(shard, f"weights/{shard.name}")
    tmp.replace(dst)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", type=Path, help="downloaded HuggingFace checkpoint directory")
    p.add_argument(
        "--out",
        type=Path,
        help="archive path; default <data_path>/<src name>.pulsar",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dst = args.out or new_model_path(args.src.name)
    migrate(args.src, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
