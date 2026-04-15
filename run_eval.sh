#!/bin/bash
# =============================================================================
#  Llama 3.2 1B lm_eval 벤치마크 실행 스크립트
# =============================================================================
#
#  이 스크립트는 Llama 3.2 1B 모델을 로딩하고 lm_eval 벤치마크를 실행하기 위한
#  전체 환경 설정과 실행 과정을 자동화합니다.
#
#  사용법:
#    chmod +x run_eval.sh   # 실행 권한 부여 (최초 1회)
#    ./run_eval.sh           # 기본 실행 (모델 분석 + 기본 벤치마크)
#    ./run_eval.sh --quick   # 빠른 테스트 (10% 샘플만)
#    ./run_eval.sh --full    # 전체 벤치마크 (모든 태스크)
#    ./run_eval.sh --eval-only  # 모델 분석 건너뛰고 평가만 실행
#
# =============================================================================

set -euo pipefail  # 에러 발생 시 즉시 중단, 미정의 변수 사용 시 에러, 파이프라인 에러 감지

# =============================================================================
# 환경 변수 설정
# =============================================================================

# 모델 경로 (허깅페이스에서 미리 다운로드해둔 경로)
export MODEL_PATH="/data2/llm_download/Llama-3.2-1B"

# GPU 설정
# - 사용할 GPU 번호를 지정합니다 (0부터 시작)
# - 여러 개 사용하려면: export CUDA_VISIBLE_DEVICES=0,1
# - GPU가 없으면 이 줄을 주석처리하면 CPU로 동작합니다
export CUDA_VISIBLE_DEVICES=0

# Hugging Face 캐시 디렉토리 (선택사항)
# - 토크나이저나 데이터셋을 캐싱할 경로를 지정합니다
# export HF_HOME="/data2/hf_cache"

# PyTorch 메모리 설정 (OOM 에러 방지)
# - GPU 메모리 단편화를 줄이기 위한 설정입니다
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# tokenizers 병렬처리 경고 방지
export TOKENIZERS_PARALLELISM=false

# =============================================================================
# 색상 설정 (터미널 출력용)
# =============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # 색상 리셋

# =============================================================================
# 헬퍼 함수
# =============================================================================

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# =============================================================================
# 1단계: 환경 확인
# =============================================================================

print_header "1단계: 환경 확인"

# Python 버전 확인 및 명령어 저장
# python3을 우선 사용하고, 없으면 python을 사용합니다
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python이 설치되어 있지 않습니다!"
    exit 1
fi

echo "Python 버전: $($PYTHON_CMD --version)"

# GPU 확인
echo ""
echo "GPU 정보:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null || {
        print_warning "nvidia-smi 실행 실패. GPU가 없거나 드라이버가 설치되지 않았을 수 있습니다."
    }
else
    print_warning "nvidia-smi를 찾을 수 없습니다. CPU 모드로 실행됩니다."
fi

# 모델 경로 확인
echo ""
echo "모델 경로 확인:"
if [ -d "$MODEL_PATH" ]; then
    print_success "모델 디렉토리 존재: $MODEL_PATH"
    echo "  파일 목록:"
    ls -lh "$MODEL_PATH" | head -20
else
    print_error "모델 디렉토리를 찾을 수 없습니다: $MODEL_PATH"
    echo "  허깅페이스에서 모델을 먼저 다운로드해주세요."
    echo "  예시: huggingface-cli download meta-llama/Llama-3.2-1B --local-dir $MODEL_PATH"
    exit 1
fi

# =============================================================================
# 2단계: 패키지 설치
# =============================================================================

print_header "2단계: 필요한 패키지 설치"

# 스크립트가 있는 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "작업 디렉토리: $SCRIPT_DIR"

# pip 업그레이드
echo ""
echo "pip 업그레이드 중..."
pip install --upgrade pip --quiet

# requirements.txt 설치
if [ -f "requirements.txt" ]; then
    echo "requirements.txt에서 패키지 설치 중..."
    pip install -r requirements.txt --quiet
    print_success "패키지 설치 완료"
else
    print_warning "requirements.txt를 찾을 수 없습니다. 수동으로 패키지를 설치합니다."
    pip install torch transformers lm-eval accelerate sentencepiece protobuf --quiet
    print_success "패키지 설치 완료"
fi

# =============================================================================
# 3단계: 모델 구조 분석 (선택사항)
# =============================================================================

# --eval-only 옵션이 아닌 경우에만 모델 분석 실행
if [[ "$1" != "--eval-only" ]]; then
    print_header "3단계: 모델 구조 분석"
    echo "Llama 3.2 1B 모델의 내부 구조를 분석합니다..."
    echo ""

    $PYTHON_CMD inspect_model.py 2>&1 | tee model_inspection.log

    print_success "모델 분석 완료 (로그: model_inspection.log)"
else
    print_warning "모델 분석을 건너뜁니다 (--eval-only 모드)"
fi

# =============================================================================
# 4단계: lm_eval 벤치마크 실행
# =============================================================================

print_header "4단계: lm_eval 벤치마크 평가"

# 실행 모드에 따른 옵션 설정
EVAL_ARGS=""

case "$1" in
    --quick)
        # 빠른 테스트 모드: 10% 샘플만 사용
        echo "🏃 빠른 테스트 모드 (10% 샘플)"
        EVAL_ARGS="--limit 0.1"
        ;;
    --full)
        # 전체 평가 모드: 모든 태스크, 전체 샘플
        echo "🏋️ 전체 평가 모드 (모든 태스크)"
        EVAL_ARGS="--tasks all"
        ;;
    --eval-only)
        # 평가만 실행 (기본 태스크)
        echo "📊 평가만 실행 모드 (기본 태스크)"
        EVAL_ARGS=""
        ;;
    *)
        # 기본 모드: 기본 태스크, 전체 샘플
        echo "📊 기본 평가 모드"
        EVAL_ARGS=""
        ;;
esac

echo ""
echo "lm_eval 벤치마크를 시작합니다..."
echo "이 과정은 GPU 성능에 따라 수 분~수십 분 소요됩니다."
echo ""

# 결과 디렉토리 생성
mkdir -p eval_results

# 타임스탬프 생성
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# lm_eval 실행
$PYTHON_CMD run_lm_eval.py \
    $EVAL_ARGS \
    --output "eval_results/results_${TIMESTAMP}.json" \
    2>&1 | tee "eval_results/eval_log_${TIMESTAMP}.log"

print_success "벤치마크 평가 완료!"

# =============================================================================
# 5단계: 결과 요약
# =============================================================================

print_header "5단계: 결과 요약"

echo "평가 결과 파일들:"
ls -lh eval_results/ 2>/dev/null || echo "  (결과 파일 없음)"

echo ""
print_success "모든 작업이 완료되었습니다!"
echo ""
echo "다음 명령어로 결과를 확인할 수 있습니다:"
echo "  cat eval_results/results_${TIMESTAMP}.json | $PYTHON_CMD -m json.tool"
echo ""
echo "다른 옵션으로 다시 실행하려면:"
echo "  ./run_eval.sh --quick      # 빠른 테스트 (10% 샘플)"
echo "  ./run_eval.sh --full       # 전체 평가 (모든 태스크)"
echo "  ./run_eval.sh --eval-only  # 모델 분석 건너뛰기"
echo ""
