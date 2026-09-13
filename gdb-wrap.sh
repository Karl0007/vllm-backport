#!/bin/bash
# Runs the image's entrypoint script through python under cuda-gdb so an illegal access
# stops in the debugger and reports the faulting kernel. Diagnostic only (VLLM_GDB=1).
exec /usr/local/cuda/bin/cuda-gdb --batch -ex run --args \
  python3 /usr/local/bin/vllm serve "$@"
