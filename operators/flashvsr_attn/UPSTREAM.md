# Provenance

This directory contains the FlashVSR+ project's own Triton stream
sparse-attention implementation. It is maintained in this repository and is
not copied from a separate upstream project.

Its public Python package name is `flashvsr_attn`. The implementation is an
optional backend for stream inference; PyTorch SDPA remains available as the
fallback when Triton is not installed or the user selects the generic backend.
