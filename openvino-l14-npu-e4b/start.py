#!/usr/bin/env python3
"""Download the model files of the reviewed commit to the Hugging Face cache and start the server.

The server reads the models from the snapshot in the cache (G4_MODEL, G4_WORK),
and the NPU compilation cache is in G4_NPU_CACHE, which is added to gemma4_npu.py in the Dockerfile.
"""

import os
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

REPO_ID = "ryugyosoft/gemma-4-E4B-it-npu"


def main() -> None:
    """Download the model if needed and replace this process with the server."""
    # The revision is set in the Dockerfile, so that the model files match the reviewed code files.
    revision = os.environ["G4_REVISION"]
    # Only the missing files are downloaded. Without a network connection, the cached snapshot is used.
    models = Path(snapshot_download(REPO_ID, revision=revision, allow_patterns=["models/*"])) / "models"
    os.environ["G4_MODEL"] = str(models / "gemma-4-E4B-it-int4-ov")
    os.environ["G4_WORK"] = str(models)
    server = Path(__file__).with_name("server.py")
    os.execv(sys.executable, [sys.executable, str(server), *sys.argv[1:]])


if __name__ == "__main__":
    main()
