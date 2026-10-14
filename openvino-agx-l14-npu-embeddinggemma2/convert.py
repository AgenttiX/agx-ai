#!/usr/bin/env python3
"""Export the text model of google/embeddinggemma-2 to OpenVINO IR for OpenVINO Model Server (OVMS).

The exported model outputs the per-token embeddings (768 dimensions, after the projection of the model),
and OVMS does the mean pooling and the normalization (--pooling MEAN), like the sentence-transformers
configuration of the model (modules.json). The vision and audio encoders are not exported.

The model files are downloaded to the Hugging Face cache. The export is skipped if the output directory
already contains an export of the same revision with the same versions of the libraries.
Usage: convert.py OUTPUT_DIR --revision COMMIT
"""

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil

import openvino as ov
import openvino_tokenizers
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers.integrations.sdpa_attention
import transformers.models.embedding_gemma2.modeling_embedding_gemma2

REPO_ID = "google/embeddinggemma-2"
# The model card states a context of 8192 tokens, but the text config inherits 262144 from Gemma 4.
# OVMS reads the maximum input length from max_position_embeddings in config.json.
MAX_POSITION_EMBEDDINGS = 8192
MARKER = "export.json"


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat the key/value heads for grouped-query attention with a Gather instead of expand and reshape.

    The result is the same, but expand and reshape become Broadcast and Reshape with a Concat for the target shape,
    which the NPU plugin (NPUW) of OpenVINO 2026.4 takes for a KV cache (find_kv_cache_concat). It then handles the
    model as a causal decoder and fails ("Failed to check SDPA nodes") when OVMS uses NPUW, i.e. for --max_length 1024
    or more. With a Gather, the model is handled as a bidirectional encoder and run as a single forward pass.
    """
    if n_rep == 1:
        return hidden_states
    index = [head for head in range(hidden_states.shape[1]) for _ in range(n_rep)]
    return hidden_states[:, index]


class TextModel(torch.nn.Module):
    """The text model with the inputs and the output that OVMS expects."""

    def __init__(self, model: torch.nn.Module) -> None:
        """Wrap the text model of EmbeddingGemma 2."""
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the per-token embeddings."""
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def export(output: Path, revision: str) -> None:
    """Export the model, the tokenizer and the configuration to the output directory."""
    transformers.integrations.sdpa_attention.repeat_kv = repeat_kv
    transformers.models.embedding_gemma2.modeling_embedding_gemma2.repeat_kv = repeat_kv

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=revision)
    model = AutoModel.from_pretrained(REPO_ID, revision=revision, dtype=torch.float32).language_model.eval()
    example = tokenizer(["title: none | text: An example document."], return_tensors="pt")
    with torch.no_grad():
        ov_model = ov.convert_model(
            TextModel(model),
            example_input={"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]},
            input=[("input_ids", [-1, -1], ov.Type.i64), ("attention_mask", [-1, -1], ov.Type.i64)],
        )
    ov_model.outputs[0].get_tensor().set_names({"last_hidden_state"})
    ov.save_model(ov_model, output / "openvino_model.xml", compress_to_fp16=True)

    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, output / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output / "openvino_detokenizer.xml")
    tokenizer.save_pretrained(output)

    config = AutoConfig.from_pretrained(REPO_ID, revision=revision).text_config
    config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    config.save_pretrained(output)


def main() -> None:
    """Export the model unless an up-to-date export exists."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Directory of the exported model")
    parser.add_argument("--revision", required=True, help="Commit of google/embeddinggemma-2")
    args = parser.parse_args()
    output: Path = args.output

    info = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("openvino", "openvino-tokenizers", "torch", "transformers")
        },
    }
    marker = output / MARKER
    if marker.is_file() and json.loads(marker.read_text(encoding="utf-8")) == info:
        print(f"{output} is up to date: {json.dumps(info)}", flush=True)
        return

    # Export to a temporary directory and replace the old export only when the new one is complete,
    # so that OVMS never sees a partial export.
    tmp = output.with_name(output.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    print(f"Exporting {REPO_ID} at {args.revision} to {output}", flush=True)
    export(tmp, args.revision)
    (tmp / MARKER).write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(output, ignore_errors=True)
    tmp.rename(output)
    print(f"Exported: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
