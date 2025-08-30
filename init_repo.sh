#!/usr/bin/env bash
set -euo pipefail

REPO=/mnt/d/GitHub/MeQuest/tinyllama-lora
mkdir -p $REPO/{data/{raw,processed},configs,scripts,src,outputs,logs}

cd $REPO
git init

# 기본 파일들
cat > README.md <<'EOF'
# TinyLlama LoRA 튜닝 파이프라인

연구/개발을 위한 기본 구조.
- `data/` : 데이터셋(raw/processed)
- `configs/` : 학습/평가 설정
- `scripts/` : 실행 스크립트
- `src/` : 파이썬 소스
- `outputs/` : 체크포인트
- `logs/` : 학습 로그
EOF

cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.venv/

# Checkpoints
outputs/
logs/

# OS
.DS_Store
Thumbs.db

# Env
.env
EOF

cat > scripts/run_train.sh <<'EOF'
#!/usr/bin/env bash
# 학습 실행 스크립트 (추후 코드 삽입)
echo ">>> run_train.sh 실행됨 (학습 코드 자리)"
EOF
chmod +x scripts/run_train.sh

cat > configs/train_qlora.yaml <<'EOF'
# 학습용 설정 예시
model_name: TinyLlama/TinyLlama-1.1B-Chat-v1.0
output_dir: outputs/exp1
num_train_epochs: 3
per_device_train_batch_size: 8
EOF

cat > src/__init__.py <<'EOF'
# src 패키지 초기화
EOF
