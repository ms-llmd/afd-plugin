#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Generate a P2pNcclAFDConnector recipe script (and deploy manifest) from a JSON config.

See configs/README.md (sibling to this file) for the config format and
configs/examples/ for worked examples. Output is consumed by the deploy-afd-k8s skill
(.agents/skills/deploy-afd-k8s/SKILL.md): a single recipe.sh for
placement.node_mode == "single", or a pair of attention/ffn recipe scripts
for "multi", plus a <name>.manifest.json describing what was generated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

AFD_FFN_HOST_PLACEHOLDER = "AFD_FFN_HOST_PLACEHOLDER"
SINGLE_NODE_AFD_HOST = "127.0.0.1"
VLLM_HTTP_HOST = "127.0.0.1"
AFD_CONNECTOR_TYPE = "P2pNcclAFDConnector"
DEFAULT_AFD_CONNECTOR_PORT = 6269
DEFAULT_CLIENT_PORT = 18305


def _connector_port() -> int:
    """Base AFD rendezvous port, overridable via the AFD_CONNECTOR_PORT env var."""
    return int(os.environ.get("AFD_CONNECTOR_PORT", DEFAULT_AFD_CONNECTOR_PORT))


def _client_port() -> int:
    """vLLM OpenAI-compatible HTTP port, overridable via the AFD_CLIENT_PORT env var."""
    return int(os.environ.get("AFD_CLIENT_PORT", DEFAULT_CLIENT_PORT))


class ConfigError(ValueError):
    """Raised when the input config fails validation."""


def validate(config: dict[str, Any]) -> None:
    """Enforce the rules documented in configs/README.md."""
    topology = config["topology"]
    if topology["strategy"] != "colocation":
        raise ConfigError(
            "topology.strategy must be 'colocation'; "
            "prefill_decode_disaggregation is out of scope for deploy-afd-k8s"
        )

    afd_enabled = topology["afd_enabled"]
    node_mode = config["placement"]["node_mode"]
    dbo_enabled = config.get("dbo", {}).get("enabled", False)

    if afd_enabled:
        for field in ("num_attention_ranks", "num_ffn_ranks", "ffn"):
            if field not in topology:
                raise ConfigError(f"topology.{field} is required when afd_enabled is true")
        num_a = topology["num_attention_ranks"]
        num_f = topology["num_ffn_ranks"]
        if num_a < num_f or num_a % num_f != 0:
            raise ConfigError(
                f"num_attention_ranks ({num_a}) must be >= num_ffn_ranks ({num_f}) "
                "and evenly divisible by it"
            )
        attn = topology["attention"]
        ffn = topology["ffn"]
        if attn["data_parallel_size"] * attn["tensor_parallel_size"] != num_a:
            raise ConfigError("topology.attention data_parallel_size * tensor_parallel_size must equal num_attention_ranks")
        if ffn["data_parallel_size"] * ffn["tensor_parallel_size"] != num_f:
            raise ConfigError("topology.ffn data_parallel_size * tensor_parallel_size must equal num_ffn_ranks")
    else:
        if dbo_enabled:
            raise ConfigError("dbo.enabled must be false when afd_enabled is false (baseline recipes don't use DBO)")
        if node_mode != "single":
            raise ConfigError("placement.node_mode must be 'single' when afd_enabled is false (nothing to split across nodes)")

    if node_mode == "multi" and not afd_enabled:
        raise ConfigError("placement.node_mode 'multi' requires topology.afd_enabled to be true")

    if config["execution"]["mode"] == "graph" and "cudagraph_capture_size" not in config["execution"]:
        raise ConfigError("execution.cudagraph_capture_size is required when execution.mode is 'graph'")


def _extra_flags(extra_serve_args: dict[str, Any]) -> list[str]:
    flags = []
    for key, value in extra_serve_args.items():
        if value is True:
            flags.append(f"--{key}")
        else:
            flags.append(f"--{key} {value}")
    return flags


def _additional_config_flag(role: str, port: int, num_attention_ranks: int, num_ffn_ranks: int, host: str) -> str:
    payload = {
        "afd": {
            "role": role,
            "connector": AFD_CONNECTOR_TYPE,
            "host": host,
            "port": port,
            "num_attention_ranks": num_attention_ranks,
            "num_ffn_ranks": num_ffn_ranks,
        }
    }
    body = json.dumps(payload)
    return f"--additional-config '{body}'"


def _build_serve_block(
    *,
    cuda_devices: list[int],
    data_parallel_size: int,
    tensor_parallel_size: int,
    enable_expert_parallel: bool,
    extra_serve_args: dict[str, Any],
    max_model_len: int,
    afd: dict[str, Any] | None,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    dbo: dict[str, Any] | None,
    execution: dict[str, Any],
    client_port: int,
    trust_remote_code: bool,
    log_name: str,
) -> str:
    flags = [
        f"--data-parallel-size {data_parallel_size}",
        f"--tensor-parallel-size {tensor_parallel_size}",
    ]
    if enable_expert_parallel:
        flags.append("--enable-expert-parallel")
    flags.extend(_extra_flags(extra_serve_args))
    flags.append(f"--max-model-len {max_model_len}")
    if afd is not None:
        flags.append(
            _additional_config_flag(
                afd["role"], afd["port"], afd["num_attention_ranks"], afd["num_ffn_ranks"], afd["host"]
            )
        )
    flags.append(f"--max-num-seqs {max_num_seqs}")
    flags.append(f"--max-num-batched-tokens {max_num_batched_tokens}")
    if dbo is not None:
        flags.append("--enable-dbo")
        flags.append(f"--dbo-decode-token-threshold {dbo['decode_token_threshold']}")
        flags.append(f"--dbo-prefill-token-threshold {dbo['prefill_token_threshold']}")
    if execution["mode"] == "eager":
        flags.append("--enforce-eager")
    else:
        size = execution["cudagraph_capture_size"]
        flags.append(f"--max-cudagraph-capture-size {size}")
        compilation_config = json.dumps({"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [size]})
        flags.append(f"--compilation-config '{compilation_config}'")
    flags.append(f"--host {VLLM_HTTP_HOST}")
    flags.append(f"--port {client_port}")
    if trust_remote_code:
        flags.append("--trust-remote-code")

    devices = ",".join(str(d) for d in cuda_devices)
    lines = [f'CUDA_VISIBLE_DEVICES={devices} uv run vllm serve "$MODEL_PATH" \\']
    for flag in flags[:-1]:
        lines.append(f"    {flag} \\")
    lines.append(f"    {flags[-1]} > {log_name} 2>&1 &")
    return "\n".join(lines)


def _script_preamble(model_id: str) -> str:
    return f"MODEL_PATH=${{MODEL_PATH:-{model_id}}}\n"


def _dbo_config(config: dict[str, Any]) -> dict[str, Any] | None:
    dbo = config.get("dbo", {})
    return dbo if dbo.get("enabled", False) else None


def generate(config: dict[str, Any]) -> dict[str, Path | str]:
    """Return {relative_output_name: file_contents} for every file this config produces."""
    model = config["model"]
    topology = config["topology"]
    serving = config["serving"]
    execution = config["execution"]
    afd_enabled = topology["afd_enabled"]
    node_mode = config["placement"]["node_mode"]
    name = config["name"]

    common_kwargs = dict(
        enable_expert_parallel=serving.get("enable_expert_parallel", True),
        extra_serve_args=model.get("extra_serve_args", {}),
        max_model_len=serving["max_model_len"],
        max_num_seqs=serving["max_num_seqs"],
        max_num_batched_tokens=serving["max_num_batched_tokens"],
        execution=execution,
        client_port=_client_port(),
        trust_remote_code=model.get("trust_remote_code", True),
    )

    outputs: dict[str, str] = {}

    if not afd_enabled:
        attn = topology["attention"]
        total = attn["data_parallel_size"] * attn["tensor_parallel_size"]
        block = _build_serve_block(
            cuda_devices=list(range(total)),
            data_parallel_size=attn["data_parallel_size"],
            tensor_parallel_size=attn["tensor_parallel_size"],
            afd=None,
            dbo=None,
            log_name="attn.log",
            **common_kwargs,
        )
        script = _script_preamble(model["model_id"]) + "\n" + block + "\n\nwait\n"
        outputs[f"{name}.recipe.sh"] = script
        return outputs

    num_a = topology["num_attention_ranks"]
    num_f = topology["num_ffn_ranks"]
    attn = topology["attention"]
    ffn = topology["ffn"]
    port = _connector_port()
    dbo = _dbo_config(config)

    def afd_block(role: str, host: str, devices: list[int], group: dict[str, Any], log_name: str) -> str:
        return _build_serve_block(
            cuda_devices=devices,
            data_parallel_size=group["data_parallel_size"],
            tensor_parallel_size=group["tensor_parallel_size"],
            afd={
                "role": role,
                "port": port,
                "num_attention_ranks": num_a,
                "num_ffn_ranks": num_f,
                "host": host,
            },
            dbo=dbo,
            log_name=log_name,
            **common_kwargs,
        )

    if node_mode == "single":
        attn_block = afd_block("attention", SINGLE_NODE_AFD_HOST, list(range(num_a)), attn, "attn.log")
        ffn_block = afd_block("ffn", SINGLE_NODE_AFD_HOST, list(range(num_a, num_a + num_f)), ffn, "ffn.log")
        script = _script_preamble(model["model_id"]) + "\n" + attn_block + "\n\n" + ffn_block + "\n\nwait\n"
        outputs[f"{name}.recipe.sh"] = script
    else:
        attn_block = afd_block("attention", AFD_FFN_HOST_PLACEHOLDER, list(range(num_a)), attn, "attn.log")
        ffn_block = afd_block("ffn", AFD_FFN_HOST_PLACEHOLDER, list(range(num_f)), ffn, "ffn.log")
        outputs[f"{name}.attention.recipe.sh"] = _script_preamble(model["model_id"]) + "\n" + attn_block + "\n\nwait\n"
        outputs[f"{name}.ffn.recipe.sh"] = _script_preamble(model["model_id"]) + "\n" + ffn_block + "\n\nwait\n"

    return outputs


def build_manifest(config: dict[str, Any], recipe_paths: dict[str, str]) -> dict[str, Any]:
    model = config["model"]
    topology = config["topology"]
    afd_enabled = topology["afd_enabled"]
    node_mode = config["placement"]["node_mode"]
    model_id = model["model_id"]

    if afd_enabled:
        num_a = topology["num_attention_ranks"]
        num_f = topology["num_ffn_ranks"]
        base_port = _connector_port()
        ffn_port_range = [base_port, base_port + num_f]
        if node_mode == "single":
            gpu_count: Any = num_a + num_f
            recipes = {"single": recipe_paths["single"]}
        else:
            gpu_count = {"attention": num_a, "ffn": num_f}
            recipes = {"attention": recipe_paths["attention"], "ffn": recipe_paths["ffn"]}
    else:
        ffn_port_range = None
        attn = topology["attention"]
        gpu_count = attn["data_parallel_size"] * attn["tensor_parallel_size"]
        recipes = {"single": recipe_paths["single"]}

    return {
        "name": config["name"],
        "model_id": model_id,
        "model_id_form": "path" if model_id.startswith("/") else "hf",
        "afd_enabled": afd_enabled,
        "node_mode": node_mode,
        "gpu_count": gpu_count,
        "max_num_batched_tokens": config["serving"]["max_num_batched_tokens"],
        "client_port": _client_port(),
        "ffn_port_range": ffn_port_range,
        "recipes": recipes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to a deploy config JSON file")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "generated",
        help="Directory to write the generated recipe script(s) and manifest into (default: generated/ next to this script)",
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    validate(config)

    files = generate(config)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for filename, contents in files.items():
        out_path = args.out_dir / filename
        out_path.write_text(contents)
        out_path.chmod(0o755)
        role = "single" if filename.endswith(".recipe.sh") and ".attention." not in filename and ".ffn." not in filename else (
            "attention" if ".attention." in filename else "ffn"
        )
        written[role] = str(out_path)
        print(f"wrote {out_path}")

    manifest = build_manifest(config, written)
    manifest_path = args.out_dir / f"{config['name']}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
