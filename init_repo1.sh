#!/usr/bin/env bash
set -euo pipefail

REPO=~/tinyllama-lora
mkdir -p $REPO/{data/{raw,processed},configs,scripts,src,outputs,logs}
cd $REPO

# 깃 초기화
if [ ! -d .git ]; then
  git init
fi

# README
cat > README.md <<'EOF'
# TinyLlama LoRA 튜닝 파이프라인 (WSL)

연구/개발 기본 구조:
- `data/` : 데이터(raw/processed)
- `configs/` : 학습/평가 설정
- `scripts/` : 실행 스크립트(환경/학습/평가 등)
- `src/` : 파이썬 소스
- `outputs/` : 체크포인트/결과물
- `logs/` : 런타임 로그

## 빠른 시작
```bash
bash scripts/env_setup.sh            # 가상환경 + 패키지 설치
source .venv/bin/activate
python -m src.dataset --preview 3    # 전처리 미리보기(샘플 출력)
```
EOF

# .gitignore
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.venv/

# Checkpoints & logs
outputs/
logs/

# OS
.DS_Store
Thumbs.db

# Secrets
.env
.env.*
EOF

# 샘플 학습 설정(필요시 수정)
cat > configs/train_qlora.yaml <<'EOF'
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output_dir: outputs/tinyllama-qlora-exp1
seed: 42

# QLoRA
load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]

# Train
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 2
num_train_epochs: 3
learning_rate: 2.0e-4
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.0
max_seq_length: 2048
packing: true
gradient_checkpointing: true
bf16: true

# Logging/Save
logging_steps: 20
save_steps: 500
eval_steps: 500
save_total_limit: 3
report_to: ["wandb"]
wandb_project: "tinyllama-lora"
wandb_run_name: "exp1_qlora_nf4"

# Dataloader
num_workers: 4
pin_memory: true

# Eval
eval_ratio: 0.01
EOF

# scripts/run_train.sh (자리표시자)
cat > scripts/run_train.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
echo ">>> run_train.sh 자리입니다. trainer 코드 붙여 넣은 뒤 실행하세요."
EOF
chmod +x scripts/run_train.sh

# requirements.txt (torch는 env_setup에서 별도로 설치)
cat > requirements.txt <<'EOF'
transformers==4.43.3
datasets==2.20.0
peft==0.13.2
bitsandbytes==0.43.2
accelerate==0.34.2
trl==0.9.6
einops==0.8.0
safetensors==0.4.4
sentencepiece==0.2.0
wandb==0.17.8
tqdm==4.66.5
pyyaml==6.0.2
protobuf==5.27.3
# 선택
tiktoken==0.7.0
EOF

# scripts/env_setup.sh — venv, torch(CUDA), 필수 설치
cat > scripts/env_setup.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# 1) 가상환경
python3 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip wheel setuptools

# 2) PyTorch (CUDA 12.1용; WSL은 Win 드라이버만 맞으면 OK)
pip install --index-url https://download.pytorch.org/whl/cu121 "torch==2.4.*" "torchvision==0.19.*" "torchaudio==2.4.*" \
  || pip install "torch==2.4.*"

# 3) 나머지 의존성
pip install -r requirements.txt

# 4) flash-attn (가능 시만)
pip install flash-attn --no-build-isolation || echo "[info] flash-attn 설치 생략"

# 5) 개발 도구
sudo apt-get update -y && sudo apt-get install -y git git-lfs tmux htop nvtop || true
git lfs install

# 6) 환경 파일 템플릿
cat > .env.example <<'EOT'
# 복사해서 .env 로 사용하세요.
WANDB_API_KEY=
WANDB_PROJECT=tinyllama-lora
HF_TOKEN=
EOT

echo
echo "[ok] 환경 준비 완료."
echo "사용 방법:"
echo "  source .venv/bin/activate"
echo "  export \$(grep -v '^#' .env | xargs)   # (선택) .env 사용 시"
echo "  huggingface-cli login --token \$HF_TOKEN   # (선택)"
echo "  wandb login \$WANDB_API_KEY                # (선택)"
EOF
chmod +x scripts/env_setup.sh

# src/__init__.py
cat > src/__init__.py <<'EOF'
# src package
EOF

# src/dataset.py — Alpaca/messages 자동 감지 -> text 필드 생성
cat > src/dataset.py <<'EOF'
import argparse, glob, json
from typing import Dict, Any, List
from datasets import load_dataset, Dataset
import os

USER_TAG = "<|user|>: "
ASSIST_TAG = "<|assistant|>: "
EOS = "</s>"

def _format_alpaca(ex: Dict[str, Any]) -> Dict[str, str]:
    instr = (ex.get("instruction") or "").strip()
    inp   = (ex.get("input") or "").strip()
    out   = (ex.get("output") or "").strip()
    user  = instr if not inp else instr + "\n\n" + inp
    text  = f"{USER_TAG}{user}\n{ASSIST_TAG}{out}{EOS}"
    return {"text": text}

def _format_messages(ex: Dict[str, Any]) -> Dict[str, str]:
    msgs: List[Dict[str,str]] = ex.get("messages") or []
    parts = []
    for m in msgs:
        role = m.get("role","").lower()
        content = (m.get("content") or "").strip()
        if role == "user":
            parts.append(f"{USER_TAG}{content}")
        elif role == "assistant":
            parts.append(f"{ASSIST_TAG}{content}")
        else:
            # system 등은 현재 템플릿에서 생략
            continue
    text = "\n".join(parts)
    if not text.endswith(EOS):
        text += EOS
    return {"text": text}

def _detect_and_format(ex: Dict[str, Any]) -> Dict[str,str]:
    if "messages" in ex:
        return _format_messages(ex)
    if "instruction" in ex and "output" in ex:
        return _format_alpaca(ex)
    # fallback: 전체를 assistant 응답으로 취급
    raw = json.dumps(ex, ensure_ascii=False)
    return {"text": f"{USER_TAG}{raw}\n{ASSIST_TAG}{''}{EOS}"}

def build_dataset(pattern: str) -> Dataset:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    ds = load_dataset("json", data_files={"train": files})["train"]
    ds = ds.map(_detect_and_format, remove_columns=[c for c in ds.column_names if c!="text"])
    return ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/processed/*.jsonl", help="JSONL 경로 글롭")
    ap.add_argument("--preview", type=int, default=0, help="상위 N개 출력")
    args = ap.parse_args()

    ds = build_dataset(args.glob)
    print(ds)
    if args.preview > 0:
        for i in range(min(args.preview, len(ds))):
            print(f"\n--- sample {i} ---\n{ds[i]['text'][:500]}")

if __name__ == "__main__":
    main()
EOF

# 첫 커밋(이미 커밋된 적 없으면)
git add .
if ! git rev-parse --quiet --verify HEAD >/dev/null; then
  git commit -m "init: skeleton + requirements/env_setup/dataset"
fi

echo
echo "[done] 템플릿 리포 준비 완료: $REPO"
echo "다음 순서:"
echo "  1) bash scripts/env_setup.sh"
echo "  2) source .venv/bin/activate"
echo "  3) (데이터 넣기) data/processed/*.jsonl"
echo "  4) python -m src.dataset --preview 3"
