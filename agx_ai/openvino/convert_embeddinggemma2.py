#!/usr/bin/env python3
"""Export the text model of google/embeddinggemma-2 to OpenVINO IR for OpenVINO Model Server (OVMS).

The exported model outputs the per-token embeddings (768 dimensions, after the projection of the model),
and OVMS does the mean pooling and the normalization (--pooling MEAN), like the sentence-transformers
configuration of the model (modules.json). The vision and audio encoders are not exported.

The model files are downloaded to the Hugging Face cache. The export is skipped if the output directory
already contains an export of the same revision with the same export settings and versions of the libraries.
Usage: python -m agx_ai.openvino.convert_embeddinggemma2 OUTPUT_DIR --revision COMMIT
"""

import argparse
import importlib.metadata
from pathlib import Path

import openvino as ov
import openvino_tokenizers
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
import transformers.integrations.sdpa_attention
import transformers.models.embedding_gemma2.modeling_embedding_gemma2

from agx_ai.openvino.export import convert_once

REPO_ID = "google/embeddinggemma-2"
# The model card states a context of 8192 tokens, but the text config inherits 262144 from Gemma 4.
# OVMS reads the maximum input length from max_position_embeddings in config.json.
MAX_POSITION_EMBEDDINGS = 8192
# Added to the attention scores of the masked keys. Finite, so that the sum of two masks fits in FP16 on the NPU.
MASK_VALUE = -30000.0


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
    """The text model with the inputs and the output that OVMS expects.

    The attention masks are given to the model as additive float masks: the padding as a [batch, 1, 1, keys] vector
    and the sliding window as a [queries, keys] band that depends only on the input length. The default boolean
    [batch, 1, queries, keys] masks of transformers are built at run time with BitwiseAnd and Select, which run slowly
    on the SHAVE cores of the NPU, whereas the band is a constant for the static input length on the NPU.
    The padded query positions attend to the valid tokens like the others, and their outputs are excluded
    by the mean pooling of OVMS.
    """

    def __init__(self, model: torch.nn.Module, sliding_window: int) -> None:
        """Wrap the text model of EmbeddingGemma 2."""
        super().__init__()
        self.model: torch.nn.Module = model
        self.sliding_window: int = sliding_window

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the per-token embeddings."""
        padding = (1 - attention_mask[:, None, None, :].to(torch.float32)) * MASK_VALUE
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        # Bidirectional sliding window: a token attends to the tokens within the (inclusive) window
        # (sliding_window_bidirectional_overlay of transformers).
        outside = (positions[:, None] - positions[None, :]).abs() > self.sliding_window
        window = outside.to(torch.float32) * MASK_VALUE
        masks = {"full_attention": padding, "sliding_attention": padding + window}
        return self.model(input_ids=input_ids, attention_mask=masks).last_hidden_state


def export(output: Path, revision: str) -> None:
    """Export the model, the tokenizer and the configuration to the output directory.

    The model is saved with FP32 weights. The NPU compiles it in FP16 anyway, but with FP16 weights in the IR,
    the decompression nodes of the weights stop the constant folding when OVMS reshapes the model to the static input
    length on the NPU, and e.g. the rotary embeddings are then computed on every inference (10 % slower).
    """
    transformers.integrations.sdpa_attention.repeat_kv = repeat_kv
    transformers.models.embedding_gemma2.modeling_embedding_gemma2.repeat_kv = repeat_kv

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, revision=revision)
    if tokenizer is None:
        raise RuntimeError(f"{REPO_ID} has no tokenizer")
    model = AutoModel.from_pretrained(REPO_ID, revision=revision, dtype=torch.float32).language_model.eval()
    config = AutoConfig.from_pretrained(REPO_ID, revision=revision).text_config
    example = tokenizer(["title: none | text: An example document."], return_tensors="pt")
    with torch.no_grad():
        ov_model = ov.convert_model(
            TextModel(model, config.sliding_window),
            example_input={"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]},
            input=[("input_ids", [-1, -1], ov.Type.i64), ("attention_mask", [-1, -1], ov.Type.i64)],
        )
    ov_model.outputs[0].get_tensor().set_names({"last_hidden_state"})
    ov.save_model(ov_model, output / "openvino_model.xml", compress_to_fp16=False)

    ov_tokenizer, ov_detokenizer = openvino_tokenizers.convert_tokenizer(tokenizer, with_detokenizer=True)
    ov.save_model(ov_tokenizer, output / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, output / "openvino_detokenizer.xml")
    tokenizer.save_pretrained(output)

    config.max_position_embeddings = MAX_POSITION_EMBEDDINGS
    config.save_pretrained(output)


def main() -> None:
    """Export the model unless an up-to-date export exists."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path, help="Directory of the exported model")
    parser.add_argument("--revision", required=True, help="Commit of google/embeddinggemma-2")
    args = parser.parse_args()

    info = {
        "repo_id": REPO_ID,
        "revision": args.revision,
        # Changes of the export itself, which have to cause a new export.
        "export": {"weights": "fp32", "masks": "additive"},
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("openvino", "openvino-tokenizers", "torch", "transformers")
        },
    }
    convert_once(args.output, info, lambda tmp: export(tmp, args.revision))


if __name__ == "__main__":
    main()
