#!/usr/bin/env python3
"""Prepare Gemma 4 E4B for OpenVINO Model Server (OVMS) on the NPU of Meteor Lake (NPU 3720).

The official OpenVINO IR (OpenVINO/gemma-4-E4B-it-int4-ov) has asymmetric INT4 weights, which the NPU compiler
of NPU 3720 rejects ("Unsupported DequantizeOp configuration"). This script takes the FP16 language model of
OpenVINO/gemma-4-E4B-it-fp16-ov, compresses its weights with NNCF to symmetric INT4 with a group size of 128
(data-free), and combines it with the other files of the INT4 model (tokenizers, the text and per-layer embeddings,
the vision encoder and the configuration), which OVMS runs on the CPU.

The model files are downloaded to the Hugging Face cache. The conversion is skipped if the output directory already
contains a conversion of the same revisions with the same settings and library versions.
Usage: python -m agx_ai.openvino.convert_gemma4 OUTPUT_DIR --int4-revision COMMIT --fp16-revision COMMIT
"""

import argparse
import importlib.metadata
from pathlib import Path
import shutil

from huggingface_hub import snapshot_download
import nncf
import openvino as ov

from agx_ai.openvino.export import convert_once

INT4_REPO = "OpenVINO/gemma-4-E4B-it-int4-ov"
FP16_REPO = "OpenVINO/gemma-4-E4B-it-fp16-ov"
LANGUAGE_MODEL = "openvino_language_model"
# Channel-wise INT4 (group size -1) decodes 20 % faster on the NPU, but made clearly more arithmetic mistakes
# (data-free and with AWQ and scale estimation), so the group size of 128 is used (see openvino-l14-npu-e4b/README.md).
GROUP_SIZE = 128


def convert(output: Path, int4_revision: str, fp16_revision: str) -> None:
    """Write the model with the INT4_SYM language model to the output directory."""
    # Everything except the language model of the INT4 model. Only the language model of the FP16 model.
    int4 = Path(snapshot_download(INT4_REPO, revision=int4_revision, ignore_patterns=[f"{LANGUAGE_MODEL}.*"]))
    fp16 = Path(snapshot_download(FP16_REPO, revision=fp16_revision, allow_patterns=[f"{LANGUAGE_MODEL}.*"]))
    for path in int4.iterdir():
        if path.is_file() and not path.name.startswith("."):
            # The files in the cache are symlinks to the blobs, so the targets are copied.
            shutil.copyfile(path, output / path.name)

    model = ov.Core().read_model(fp16 / f"{LANGUAGE_MODEL}.xml")
    # The embeddings and the LM head (the last MatMul) are kept in INT8 by default, which the NPU supports.
    # An INT4 LM head was not faster on the NPU.
    model = nncf.compress_weights(
        model, mode=nncf.CompressWeightsMode.INT4_SYM, group_size=GROUP_SIZE, ratio=1.0,
    )
    ov.save_model(model, output / f"{LANGUAGE_MODEL}.xml", compress_to_fp16=False)


def main() -> None:
    """Convert the model unless an up-to-date conversion exists."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Directory of the converted model")
    parser.add_argument("--int4-revision", required=True, help=f"Commit of {INT4_REPO}")
    parser.add_argument("--fp16-revision", required=True, help=f"Commit of {FP16_REPO}")
    args = parser.parse_args()

    info = {
        "repos": {INT4_REPO: args.int4_revision, FP16_REPO: args.fp16_revision},
        "weights": {"mode": "int4_sym", "group_size": GROUP_SIZE, "ratio": 1.0},
        "versions": {name: importlib.metadata.version(name) for name in ("nncf", "openvino")},
    }
    convert_once(args.output, info, lambda tmp: convert(tmp, args.int4_revision, args.fp16_revision))


if __name__ == "__main__":
    main()
