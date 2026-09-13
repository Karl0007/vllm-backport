#!/bin/bash
# Attach cuda-gdb and let the *device* exception stop it (API failures ignored, since
# their stop point is a later host API call, not the faulting kernel).
PID="$1"
exec /usr/local/cuda/bin/cuda-gdb --batch \
  -ex "set cuda api_failures ignore" \
  -ex "attach $PID" \
  -ex continue \
  -ex "info cuda kernels" \
  -ex "bt"
