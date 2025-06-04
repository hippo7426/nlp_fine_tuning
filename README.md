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

## 주요 기능

### 1. 특수 토큰 처리
- `<|topic:` 및 `|>` 토큰을 자동으로 토크나이저에 추가
- 모델 임베딩 크기를 토크나이저 어휘 크기에 맞춰 자동 리사이즈

### 2. 유연한 Fine-tuning 전략
- **Full Fine-tuning**: 전체 모델 파라미터 학습
- **Head-only Fine-tuning**: 상위 레이어만 학습 (메모리 효율적)

### 3. 자동 데이터 분할
- 학습/검증/테스트 데이터를 8:1:1 비율로 자동 분할
- scikit-learn의 train_test_split 사용

### 4. 종합적인 평가
- **Perplexity**: 언어 모델 성능 측정
- **BERTScore**: 생성된 시와 참조 시의 의미적 유사도 측정

### 5. GPU/CPU 자동 감지
- GPU 사용 가능 여부를 자동으로 감지
- Windows 환경 호환성 고려

### 6. 학습 과정 시각화
- 손실 함수 그래프
- 학습률 스케줄 그래프
- 학습 과정 자동 저장

## 설정 옵션

`config.py`에서 다음 설정들을 조정할 수 있습니다:

```python
# 모델 설정
model_name = "skt/kogpt2-base-v2"
max_length = 512

# 학습 설정
num_epochs = 3
batch_size = 4
learning_rate = 5e-5
warmup_steps = 100

# Fine-tuning 전략
full_finetuning = True
freeze_layers = 0

# 생성 설정
max_new_tokens = 200
temperature = 0.8
top_k = 50
top_p = 0.9
```

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

## 생성 예시

```
Topic: 자연
------------------------------
푸른 하늘 아래 펼쳐진
들녘에서 바람이 불어오고
꽃들이 춤추며 노래하네
자연의 품에서 쉬어가리
------------------------------
```

## 문제 해결

### CUDA 메모리 부족
- 배치 크기를 줄입니다: `--batch-size 2`
- CPU 사용을 고려합니다: `--cpu`

### 학습 속도가 느림
- Head-only fine-tuning을 사용합니다: `--head-only`
- Gradient accumulation steps를 조정합니다 (config.py)

### 생성 품질이 낮음
- 더 많은 에포크로 학습: `--epochs 10`
- 학습률을 조정: `--lr 3e-5`
- 생성 파라미터를 조정 (config.py의 temperature, top_k, top_p)

## 라이선스

이 프로젝트는 교육 목적으로 제작되었습니다. KoGPT2 모델의 라이선스 정책을 따릅니다.

  감사합니다! 