"""Converters that prepare models for OpenVINO Model Server (OVMS) on the Intel NPU.

They are run in the converter container of ``openvino-npu-convert``, which has the dependencies of the ``openvino``
extra of this project. The modules do not depend on the rest of :mod:`agx_ai`.

Modules:
  convert_embeddinggemma2 - export of the text model of google/embeddinggemma-2
  convert_gemma4          - symmetric INT4 version of Gemma 4 E4B for NPU 3720 (Meteor Lake)
  export                  - skipping up-to-date conversions and replacing the old ones atomically
"""
