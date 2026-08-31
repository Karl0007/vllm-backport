#!/usr/bin/env bash
# Resumable download of wtdcode/GLM-5.3-Flash-AWQ-W4A16 (~178G) via hf-mirror.
# vllm26 env carries huggingface_hub 1.28; snapshot_download resumes in-place.
set -euo pipefail
source /home/kk/miniconda3/etc/profile.d/conda.sh
conda activate vllm26
export HF_ENDPOINT=https://hf-mirror.com
# hf_xet's CAS client bypasses the mirror (cas-server.xethub.hf.co -> 401);
# force classic LFS download which does respect HF_ENDPOINT.
export HF_HUB_DISABLE_XET=1
exec python - <<'EOF'
import time
from huggingface_hub import snapshot_download
while True:
    try:
        p = snapshot_download(
            "wtdcode/GLM-5.3-Flash-AWQ-W4A16",
            local_dir="/srv/models/wtdcode/GLM-5.3-Flash-AWQ-W4A16",
            max_workers=8,
        )
        print("DONE", p, flush=True)
        break
    except Exception as e:
        print("RETRY after:", e, flush=True)
        time.sleep(10)
EOF
