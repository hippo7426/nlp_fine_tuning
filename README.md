# KoGPT2 Korean Poetry Fine-tuning

이 프로젝트는 KoGPT2 모델을 기반으로 한국어 시 생성을 위한 fine-tuning을 수행합니다.

## 프로젝트 구조

```
fine_tuning/
├── data/
│   └── prompt_dataset.json    # 학습 데이터셋
├── config.py                  # 설정 파일
├── data_utils.py             # 데이터 전처리 유틸리티
├── trainer.py                # 모델 trainer 클래스
├── evaluation.py             # 평가 함수들
├── main.py                   # 메인 실행 스크립트
├── environment.yml           # Conda 환경 설정
├── README.md                 # 이 파일
├── outputs/                  # 출력 파일들
├── saved_models/            # 저장된 모델들
└── logs/                    # 로그 파일들
```

## 환경 설정

### 1. Conda 환경 생성
```bash
# Conda 환경 생성 및 활성화
conda env create -f environment.yml
conda activate kogpt2-finetuning
```

### 2. 수동 설치 (선택사항)
```bash
pip install torch transformers datasets tokenizers accelerate evaluate bert-score pandas numpy matplotlib seaborn tqdm scikit-learn wandb tensorboard
```

## 데이터셋 형식

데이터셋은 JSON 형식으로, 다음과 같은 구조를 가져야 합니다:

```json
[
  {
    "prompt": "<|topic:명옥헌|>\n",
    "poem": "오늘도 새로 보탠 죄 많아\n맑은 하늘조차\n진흙탕 바다로다\n..."
  }
]
```

## 사용법

### 기본 사용법 (전체 파이프라인 실행)
```bash
python main.py --mode all
```

### 학습만 실행
```bash
python main.py --mode train --epochs 5 --lr 3e-5
```

### 평가만 실행
```bash
python main.py --mode evaluate --model-path saved_models/best_model
```

### 시 생성만 테스트
```bash
python main.py --mode generate --model-path saved_models/best_model
```

### 커맨드라인 옵션

#### 학습 전략
- `--full-finetuning`: 전체 모델 fine-tuning (기본값)
- `--head-only`: Head 부분만 fine-tuning

#### 하드웨어 설정
- `--gpu`: GPU 사용 (기본값)
- `--cpu`: CPU 강제 사용

#### 학습 파라미터
- `--epochs N`: 학습 epoch 수 (기본값: 3)
- `--lr RATE`: 학습률 (기본값: 5e-5)
- `--batch-size N`: 배치 크기 (기본값: 4)
- `--max-length N`: 최대 시퀀스 길이 (기본값: 512)

#### 모델 설정
- `--model-name NAME`: 사용할 모델명 (기본값: skt/kogpt2-base-v2)

#### 실행 모드
- `--mode MODE`: 실행 모드 (train/evaluate/generate/all)
- `--model-path PATH`: 평가나 생성에 사용할 모델 경로

## 출력 파일

### 저장된 모델
- `saved_models/best_model/`: 검증 손실이 가장 낮은 모델
- `saved_models/checkpoint_epoch_N/`: 각 에포크별 체크포인트

### 평가 결과
- `outputs/evaluation_results.txt`: 평가 메트릭 결과
- `outputs/training_history.png`: 학습 과정 그래프

### 로그
- `logs/`: 학습 과정 로그 파일들

## 예시 실행

### 1. GPU로 전체 fine-tuning (5 epochs)
```bash
python main.py --mode all --epochs 5 --gpu
```

### 2. CPU로 head-only fine-tuning
```bash
python main.py --mode train --head-only --cpu --epochs 3
```

### 3. 기존 모델로 시 생성 테스트
```bash
python main.py --mode generate --model-path saved_models/best_model
```

## LLM 사용 관련

프로젝트 코드를 작성하는데 Claude사의 Sonnet 모델을 활용하였으며, 세부 기능 추가와 디버깅은 수작업으로 진행하였습니다.