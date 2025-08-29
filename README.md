# MeQuest

나를 탐구하고 성장을 지향하는 학습 플랫폼

## 폴더 구조

repo/
├─ src/ # 실제 라이브러리/모듈 코드 (패키지로 설치 가능하게 구성)
├─ scripts/ # 일회성/배치용 실행 스크립트 (train.py, eval.py 등)
├─ notebooks/ # 실험·탐색용 노트북(.ipynb). 결과 출력은 커밋 전에 비우기
├─ configs/ # 하이퍼파라미터·경로·실험 설정(YAML/JSON)
├─ data/
│ ├─ raw/ # 원천 데이터(수정 금지, 가급적 DVC로만 관리)
│ └─ processed/ # 전처리된 데이터셋(캐시 성격, 재생성 가능)
├─ models/ # 체크포인트, tokenizer, 최종 산출물(용량 큼 → DVC)
├─ runs/ # 실험 로그, 메트릭, 아티팩트(MLflow/W&B가 쓰는 디렉토리)
├─ tests/ # 단위/통합 테스트(재현성 확보)
├─ pyproject.toml # 의존성/패키징 메타 (또는 requirements.txt)
├─ .pre-commit-config.yaml
├─ .gitignore
└─ dvc.yaml / .dvc/ # DVC 파이프라인/원격 설정 파일들
