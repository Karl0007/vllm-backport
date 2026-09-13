#!/bin/bash
# Runs the image's entrypoint through python under cuda-gdb with memcheck enabled, so an
# illegal access is reported with the faulting kernel and address. Diagnostic only.
exec /usr/local/cuda/bin/cuda-gdb --batch \
  -ex "set cuda memcheck on" \
  -ex "set cuda api_failures stop" \
  -ex run \
  --args python3 /usr/local/bin/vllm serve "$@"
