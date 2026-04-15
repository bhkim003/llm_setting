<div align="center">

# 🦙 LLM Setting

**Llama 3.2 1B 모델 분석 & 벤치마크 평가 툴킷**

LLM의 내부 구조를 탐색하고, 표준 벤치마크로 성능을 평가하세요.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Transformers-4.40%2B-FFD21E?style=for-the-badge)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

</div>

---

## 📖 소개

**LLM Setting**은 LLM(대규모 언어 모델)을 처음 접하는 분들을 위한 학습용 툴킷입니다.

Meta의 **Llama 3.2 1B** 모델을 예시로, 모델의 내부 아키텍처를 한글 주석과 함께 상세히 분석하고, EleutherAI의 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)를 활용하여 다양한 벤치마크로 모델 성능을 평가할 수 있습니다.

### ✨ 주요 기능

| 기능 | 설명 |
|------|------|
| 🔍 **모델 구조 분석** | Transformer 아키텍처의 각 모듈(Embedding, Attention, MLP, Norm 등)을 상세히 분석 |
| 📊 **벤치마크 평가** | HellaSwag, ARC, MMLU 등 표준 벤치마크로 모델 성능 측정 |
| 🎓 **한글 주석** | 모든 코드에 한글 주석을 달아 LLM 입문자도 쉽게 이해 가능 |
| ⚡ **원클릭 실행** | 셸 스크립트 하나로 환경 설정부터 평가까지 자동화 |

---

## 🏗️ 프로젝트 구조

```
llm_setting/
├── 📄 inspect_model.py    # 모델 구조 분석 스크립트
├── 📄 run_lm_eval.py      # lm_eval 벤치마크 실행 스크립트
├── 📄 run_eval.sh          # 전체 파이프라인 자동화 셸 스크립트
├── 📄 requirements.txt     # Python 패키지 의존성
├── 📄 LICENSE              # MIT 라이선스
└── 📄 README.md            # 이 문서
```

---

## 🚀 빠른 시작

### 1. 사전 준비

- **Python** 3.9 이상
- **CUDA** 지원 GPU (권장, CPU도 가능)
- **Llama 3.2 1B** 모델 다운로드

```bash
# 모델 다운로드 (Hugging Face CLI 사용)
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir /data2/llm_download/Llama-3.2-1B
```

### 2. 설치

```bash
# 리포지토리 클론
git clone https://github.com/bhkim003/llm_setting.git
cd llm_setting

# 패키지 설치
pip install -r requirements.txt
```

### 3. 실행

```bash
# 방법 1: 셸 스크립트로 전체 파이프라인 실행
chmod +x run_eval.sh
./run_eval.sh

# 방법 2: 개별 스크립트 실행
python inspect_model.py      # 모델 구조 분석
python run_lm_eval.py        # 벤치마크 평가
```

---

## 🔧 실행 옵션

### 셸 스크립트 (`run_eval.sh`)

| 옵션 | 설명 |
|------|------|
| `./run_eval.sh` | 기본 실행 (모델 분석 + 기본 벤치마크) |
| `./run_eval.sh --quick` | 빠른 테스트 (10% 샘플만 사용) |
| `./run_eval.sh --full` | 전체 벤치마크 (모든 태스크) |
| `./run_eval.sh --eval-only` | 모델 분석 건너뛰고 평가만 실행 |

### Python 스크립트 (`run_lm_eval.py`)

```bash
python run_lm_eval.py                        # 기본 태스크 평가
python run_lm_eval.py --tasks hellaswag      # 특정 태스크만 평가
python run_lm_eval.py --tasks all            # 모든 기본 태스크 평가
python run_lm_eval.py --batch_size 8         # 배치 사이즈 조정
```

---

## 🧠 모델 아키텍처 개요

`inspect_model.py`를 실행하면 아래와 같은 Llama 3.2 1B의 내부 구조를 확인할 수 있습니다.

```
LlamaForCausalLM
├── model (LlamaModel)
│   ├── embed_tokens         ← 토큰 → 벡터 변환
│   ├── layers × N           ← 트랜스포머 블록
│   │   ├── self_attn        ← 셀프 어텐션 (Q, K, V, O)
│   │   ├── mlp              ← 피드포워드 (SwiGLU)
│   │   ├── input_layernorm  ← Pre-Norm (RMSNorm)
│   │   └── post_attention_layernorm
│   ├── norm                 ← 최종 RMSNorm
│   └── rotary_emb           ← RoPE 위치 인코딩
└── lm_head                  ← 로짓 출력 → 다음 토큰 예측
```

---

## 📊 지원 벤치마크

| 벤치마크 | 카테고리 | 설명 |
|----------|----------|------|
| **HellaSwag** | 상식 추론 | 상황에 이어질 자연스러운 문장 선택 |
| **ARC-Easy / ARC-Challenge** | 과학 추론 | 초등학교 수준 과학 문제 |
| **MMLU** | 종합 지식 | 57개 분야 객관식 문제 |
| **TruthfulQA** | 사실성 | 모델의 허위 정보 생성 경향 평가 |
| **Winogrande** | 상식 추론 | 대명사 지칭 대상 추론 |
| **GSM8K** | 수학 추론 | 초등학교 수준 수학 문제 |

---

## 📦 의존성

| 패키지 | 최소 버전 | 용도 |
|--------|----------|------|
| `torch` | 2.1.0 | 딥러닝 프레임워크 |
| `transformers` | 4.40.0 | 모델 로딩 및 추론 |
| `lm-eval` | 0.4.0 | 벤치마크 평가 프레임워크 |
| `accelerate` | 0.25.0 | 멀티 GPU / 혼합 정밀도 지원 |
| `sentencepiece` | 0.1.99 | 서브워드 토크나이저 |
| `protobuf` | 3.20.0 | SentencePiece 모델 파일 파싱 |

> ⚠️ **참고**: PyTorch는 본인의 CUDA 버전에 맞게 설치해야 합니다. [PyTorch 공식 사이트](https://pytorch.org/get-started/locally/)에서 확인하세요.

---

## 📝 라이선스

이 프로젝트는 [MIT 라이선스](LICENSE) 하에 배포됩니다.

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**

Made with ❤️ by [bhkim003](https://github.com/bhkim003)

</div>