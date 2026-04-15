"""
=============================================================================
  Llama 3.2 1B lm_eval 벤치마크 실행 스크립트 (run_lm_eval.py)
=============================================================================

이 스크립트는 EleutherAI의 lm-evaluation-harness를 사용하여
Llama 3.2 1B 모델의 성능을 다양한 벤치마크 태스크로 평가합니다.

lm-evaluation-harness란?
→ LLM(대규모 언어 모델)의 성능을 표준화된 벤치마크로 측정하는 프레임워크입니다.
→ 수백 가지의 평가 태스크를 제공하며, 모델 간 공정한 비교가 가능합니다.
→ GitHub: https://github.com/EleutherAI/lm-evaluation-harness

사용법:
    python run_lm_eval.py                          # 기본 태스크들로 평가
    python run_lm_eval.py --tasks hellaswag        # 특정 태스크만 평가
    python run_lm_eval.py --tasks all              # 모든 기본 태스크 평가
    python run_lm_eval.py --batch_size 8           # 배치 사이즈 조정
"""

import argparse
import json
import os
from datetime import datetime

import lm_eval
from lm_eval.models.huggingface import HFLM


# =============================================================================
# 모델 경로 설정
# =============================================================================
MODEL_PATH = "/data2/llm_download/Llama-3.2-1B"

# =============================================================================
# 결과 저장 경로
# =============================================================================
RESULTS_DIR = "./eval_results"


# =============================================================================
# lm_eval 벤치마크 태스크 설명
# =============================================================================
# 아래는 LLM 평가에 자주 사용되는 대표적인 태스크들입니다.
# 각 태스크가 무엇을 평가하는지, 어떤 형식인지 상세히 설명합니다.
#
# ※ 참고: lm_eval에는 수백 개의 태스크가 있지만,
#   여기서는 가장 널리 사용되고 의미 있는 태스크들을 선별했습니다.
# =============================================================================

TASK_DESCRIPTIONS = {
    # -------------------------------------------------------------------------
    # [1] HellaSwag - 상식 추론 (Commonsense Reasoning)
    # -------------------------------------------------------------------------
    # 유형: 4지선다형 (Multiple Choice)
    # 평가 지표: accuracy (정확도), accuracy_norm (정규화 정확도)
    #
    # 설명:
    #   주어진 상황 설명 다음에 가장 자연스러운 후속 문장을 고르는 태스크입니다.
    #   일상생활의 상식적인 추론 능력을 평가합니다.
    #
    # 예시:
    #   "A woman is outside with a bucket. She..."
    #   (A) jumps into the bucket  ← 비상식적
    #   (B) pours water from the bucket onto flowers  ← 정답 (상식적)
    #   (C) eats the bucket
    #   (D) throws the bucket into space
    # -------------------------------------------------------------------------
    "hellaswag": {
        "name": "HellaSwag",
        "category": "상식 추론 (Commonsense Reasoning)",
        "description": "상황에 이어질 가장 자연스러운 문장을 고르는 태스크",
        "metric": "acc_norm (정규화 정확도)",
        "num_fewshot": 10,  # few-shot: 예시를 10개 보여주고 문제를 품
    },

    # -------------------------------------------------------------------------
    # [2] ARC (AI2 Reasoning Challenge) - 과학 상식
    # -------------------------------------------------------------------------
    # 유형: 4지선다형 (Multiple Choice)
    # 평가 지표: accuracy, accuracy_norm
    #
    # 설명:
    #   초등학교~중학교 수준의 과학 시험 문제입니다.
    #   ARC-Easy(쉬운 문제)와 ARC-Challenge(어려운 문제)로 나뉩니다.
    #   보통 ARC-Challenge를 더 많이 사용합니다.
    #
    # 예시 (ARC-Challenge):
    #   "Which of these is a non-renewable resource?"
    #   (A) Trees  (B) Solar energy  (C) Coal ← 정답  (D) Wind
    # -------------------------------------------------------------------------
    "arc_challenge": {
        "name": "ARC-Challenge",
        "category": "과학 추론 (Science Reasoning)",
        "description": "초중등 수준 과학 문제 중 어려운 문제들",
        "metric": "acc_norm (정규화 정확도)",
        "num_fewshot": 25,
    },
    "arc_easy": {
        "name": "ARC-Easy",
        "category": "과학 추론 (Science Reasoning)",
        "description": "초중등 수준 과학 문제 중 쉬운 문제들",
        "metric": "acc_norm (정규화 정확도)",
        "num_fewshot": 25,
    },

    # -------------------------------------------------------------------------
    # [3] WinoGrande - 대명사 해석 (Coreference Resolution)
    # -------------------------------------------------------------------------
    # 유형: 이진 선택 (Binary Choice)
    # 평가 지표: accuracy
    #
    # 설명:
    #   문장 속 대명사가 가리키는 대상을 맞추는 태스크입니다.
    #   상식적인 지식을 활용한 언어 이해 능력을 평가합니다.
    #
    # 예시:
    #   "The trophy doesn't fit in the brown suitcase because it is too [big/small]."
    #   → "it"이 trophy를 가리키면 "big", suitcase를 가리키면 "small"
    # -------------------------------------------------------------------------
    "winogrande": {
        "name": "WinoGrande",
        "category": "상식 추론 / 대명사 해석",
        "description": "대명사가 가리키는 대상을 상식으로 추론하는 태스크",
        "metric": "acc (정확도)",
        "num_fewshot": 5,
    },

    # -------------------------------------------------------------------------
    # [4] PIQA - 물리적 직관 (Physical Intuition)
    # -------------------------------------------------------------------------
    # 유형: 이진 선택 (Binary Choice)
    # 평가 지표: accuracy, accuracy_norm
    #
    # 설명:
    #   물리적 세계에 대한 직관적 이해를 평가합니다.
    #   일상적인 물리 상식을 묻는 질문에 답합니다.
    #
    # 예시:
    #   "To separate egg whites from the yolk, you can..."
    #   (A) use a plastic bottle to suck up the yolk ← 정답
    #   (B) use a hammer to crack the yolk
    # -------------------------------------------------------------------------
    "piqa": {
        "name": "PIQA",
        "category": "물리적 상식 (Physical Commonsense)",
        "description": "물리적 세계에 대한 직관적 이해를 평가하는 태스크",
        "metric": "acc_norm (정규화 정확도)",
        "num_fewshot": 0,
    },

    # -------------------------------------------------------------------------
    # [5] BoolQ - 예/아니오 질의응답
    # -------------------------------------------------------------------------
    # 유형: 예/아니오 (Boolean QA)
    # 평가 지표: accuracy
    #
    # 설명:
    #   주어진 지문을 읽고, 질문에 대해 예/아니오로 답하는 태스크입니다.
    #   독해력과 추론 능력을 평가합니다.
    #
    # 예시:
    #   지문: "The Eiffel Tower is located in Paris, France..."
    #   질문: "Is the Eiffel Tower in Germany?"
    #   답: No (아니오)
    # -------------------------------------------------------------------------
    "boolq": {
        "name": "BoolQ",
        "category": "독해 / 질의응답 (Reading Comprehension)",
        "description": "지문을 읽고 예/아니오로 답하는 태스크",
        "metric": "acc (정확도)",
        "num_fewshot": 0,
    },

    # -------------------------------------------------------------------------
    # [6] MMLU (Massive Multitask Language Understanding) - 종합 지식
    # -------------------------------------------------------------------------
    # 유형: 4지선다형 (Multiple Choice)
    # 평가 지표: accuracy
    #
    # 설명:
    #   57개 과목(수학, 역사, 과학, 법률, 의학 등)에 걸친 종합 지식 평가입니다.
    #   고등학교~대학원 수준의 문제가 포함되어 있어서
    #   모델의 전반적인 지식 수준을 측정하는 데 널리 사용됩니다.
    #
    #   ※ 주의: MMLU는 57개 서브태스크를 모두 돌리면 시간이 오래 걸립니다.
    #   처음에는 일부 과목만 테스트하는 것을 추천합니다.
    #
    # 예시 (추상대수학):
    #   "Find the degree of the extension Q(sqrt(2), sqrt(3)) over Q."
    #   (A) 1  (B) 2  (C) 4 ← 정답  (D) 6
    # -------------------------------------------------------------------------
    "mmlu": {
        "name": "MMLU",
        "category": "종합 지식 (General Knowledge)",
        "description": "57개 과목에 걸친 종합 지식 평가 (고등학교~대학원 수준)",
        "metric": "acc (정확도)",
        "num_fewshot": 5,
    },

    # -------------------------------------------------------------------------
    # [7] TruthfulQA - 진실성 평가
    # -------------------------------------------------------------------------
    # 유형: 다지선다형 (Multiple Choice)
    # 평가 지표: accuracy (mc2)
    #
    # 설명:
    #   모델이 흔한 오해나 미신에 빠지지 않고 사실대로 답하는지 평가합니다.
    #   인간이 자주 틀리는 함정 질문들로 구성되어 있습니다.
    #
    # 예시:
    #   "Can colds be caused by cold weather?"
    #   → 흔한 오해: "Yes" (추운 날씨가 감기의 원인)
    #   → 정답: "No, colds are caused by viruses" (감기는 바이러스에 의해 발생)
    # -------------------------------------------------------------------------
    "truthfulqa_mc2": {
        "name": "TruthfulQA (MC2)",
        "category": "진실성 (Truthfulness)",
        "description": "흔한 오해/미신에 빠지지 않고 사실대로 답하는지 평가",
        "metric": "acc (정확도)",
        "num_fewshot": 0,
    },

    # -------------------------------------------------------------------------
    # [8] GSM8K - 수학 문제 풀이
    # -------------------------------------------------------------------------
    # 유형: 자유 생성 (Open-ended Generation)
    # 평가 지표: exact_match (정확 일치)
    #
    # 설명:
    #   초등학교 수준의 수학 단어 문제를 풀어내는 태스크입니다.
    #   단순 계산이 아니라 문장을 이해하고 여러 단계의 추론이 필요합니다.
    #
    # 예시:
    #   "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast
    #    and bakes 4 into muffins. She sells the rest at $2 each.
    #    How much does she make every day?"
    #   → 16 - 3 - 4 = 9 eggs, 9 × $2 = $18 (정답)
    # -------------------------------------------------------------------------
    "gsm8k": {
        "name": "GSM8K",
        "category": "수학 추론 (Math Reasoning)",
        "description": "초등학교 수준의 다단계 수학 문제 풀기",
        "metric": "exact_match (정확 일치)",
        "num_fewshot": 5,
    },

    # -------------------------------------------------------------------------
    # [9] OpenBookQA - 오픈북 과학 시험
    # -------------------------------------------------------------------------
    # 유형: 4지선다형 (Multiple Choice)
    # 평가 지표: accuracy, accuracy_norm
    #
    # 설명:
    #   기본적인 과학 사실을 바탕으로 추론하는 태스크입니다.
    #   "오픈북" 시험처럼 기본 지식은 주어지고, 이를 응용하는 능력을 평가합니다.
    #
    # 예시:
    #   사실: "Metals conduct electricity."
    #   질문: "A copper wire can be used to..."
    #   (A) insulate a house  (B) carry electrical current ← 정답
    #   (C) block sound  (D) absorb water
    # -------------------------------------------------------------------------
    "openbookqa": {
        "name": "OpenBookQA",
        "category": "과학 추론 (Science Reasoning)",
        "description": "기본 과학 사실을 응용하여 추론하는 태스크",
        "metric": "acc_norm (정규화 정확도)",
        "num_fewshot": 0,
    },
}

# =============================================================================
# 기본으로 실행할 태스크 목록
# =============================================================================
# 처음 돌려볼 때는 가볍고 대표적인 태스크 몇 개만 선택합니다.
# 전부 돌리면 시간이 매우 오래 걸릴 수 있습니다 (특히 MMLU, GSM8K).
DEFAULT_TASKS = [
    "hellaswag",      # 상식 추론 (~10분)
    "arc_challenge",   # 과학 추론 (~5분)
    "winogrande",      # 대명사 해석 (~5분)
    "piqa",            # 물리적 상식 (~5분)
    "boolq",           # 독해력 (~5분)
]


def print_task_info(task_names: list[str]) -> None:
    """
    실행할 태스크들의 상세 정보를 출력합니다.

    Args:
        task_names: 실행할 태스크 이름 리스트
    """
    print("\n" + "=" * 80)
    print("  📋 실행할 벤치마크 태스크 목록")
    print("=" * 80)

    for i, task_name in enumerate(task_names, 1):
        if task_name in TASK_DESCRIPTIONS:
            info = TASK_DESCRIPTIONS[task_name]
            print(f"\n  [{i}] {info['name']} (태스크명: {task_name})")
            print(f"      카테고리: {info['category']}")
            print(f"      설명: {info['description']}")
            print(f"      평가 지표: {info['metric']}")
            print(f"      Few-shot 수: {info['num_fewshot']}")
        else:
            print(f"\n  [{i}] {task_name}")
            print("      (사전 정의된 설명이 없는 태스크입니다)")


def run_evaluation(
    task_names: list[str],
    batch_size: int | str = "auto",
    num_fewshot: int | None = None,
    limit: float | None = None,
    output_path: str | None = None,
) -> dict:
    """
    lm_eval을 사용하여 모델 평가를 실행합니다.

    Args:
        task_names: 평가할 태스크 이름 리스트
        batch_size: 배치 크기. "auto"이면 자동 결정, 숫자를 넣으면 고정
        num_fewshot: few-shot 예시 수 (None이면 태스크 기본값 사용)
        limit: 평가할 샘플 수 제한 (None이면 전체 평가, 0.1이면 10%만)
                 → 처음 테스트할 때 limit=0.1로 빠르게 돌려볼 수 있음
        output_path: 결과 저장 경로

    Returns:
        lm_eval 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print("  🚀 lm_eval 벤치마크 평가 시작")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 모델 래퍼(wrapper) 생성
    # -------------------------------------------------------------------------
    # HFLM: Hugging Face 모델을 lm_eval이 사용할 수 있는 형태로 감싸는 래퍼
    # - pretrained: 모델 경로 또는 Hugging Face Hub ID
    # - dtype: 모델의 데이터 타입 (float16 = 반정밀도, 메모리 절약)
    # - batch_size: 한 번에 처리할 샘플 수
    #   "auto"로 설정하면 GPU 메모리에 맞게 자동 조절
    print(f"\n  모델 로딩 중: {MODEL_PATH}")
    print(f"  배치 크기: {batch_size}")

    model = HFLM(
        pretrained=MODEL_PATH,
        dtype="float16",
        batch_size=batch_size,
    )

    # -------------------------------------------------------------------------
    # 태스크별 few-shot 설정
    # -------------------------------------------------------------------------
    # num_fewshot이 명시적으로 지정되지 않으면, 각 태스크의 권장 값을 사용
    fewshot = num_fewshot

    # -------------------------------------------------------------------------
    # 평가 실행
    # -------------------------------------------------------------------------
    # lm_eval.simple_evaluate: 평가를 실행하는 메인 함수
    # - model: 평가할 모델 (HFLM으로 감싼 것)
    # - tasks: 평가할 태스크 이름 리스트
    # - num_fewshot: few-shot 예시 수
    # - limit: 평가 샘플 수 제한 (빠른 테스트용)
    # - batch_size: 배치 크기
    print(f"\n  평가 태스크: {', '.join(task_names)}")
    if limit:
        print(f"  ⚠️  샘플 제한: {limit} (빠른 테스트 모드)")
    print(f"  평가를 시작합니다... (태스크에 따라 수 분~수십 분 소요)")
    print()

    results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_names,
        num_fewshot=fewshot,
        limit=limit,
        batch_size=batch_size,
    )

    return results


def print_results(results: dict) -> None:
    """
    평가 결과를 보기 좋게 출력합니다.

    Args:
        results: lm_eval에서 반환한 결과 딕셔너리
    """
    print("\n" + "=" * 80)
    print("  📊 벤치마크 평가 결과")
    print("=" * 80)

    if "results" not in results:
        print("  결과가 없습니다.")
        return

    for task_name, task_results in results["results"].items():
        task_info = TASK_DESCRIPTIONS.get(task_name, {})
        display_name = task_info.get("name", task_name)
        category = task_info.get("category", "기타")

        print(f"\n  📌 {display_name} ({task_name})")
        print(f"     카테고리: {category}")

        # 주요 메트릭 출력
        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)):
                if "stderr" in metric_name:
                    # 표준오차는 ± 형태로 표시
                    print(f"     {metric_name}: ±{value:.4f}")
                else:
                    # 퍼센트로 변환하여 출력 (0~1 범위인 경우)
                    if 0 <= value <= 1:
                        print(f"     {metric_name}: {value:.4f} ({value * 100:.2f}%)")
                    else:
                        print(f"     {metric_name}: {value:.4f}")


def save_results(results: dict, output_path: str) -> None:
    """
    평가 결과를 JSON 파일로 저장합니다.

    Args:
        results: lm_eval 결과 딕셔너리
        output_path: 저장 경로
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # datetime 등 직렬화 불가능한 객체 처리
    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=default_serializer)

    print(f"\n  💾 결과가 저장되었습니다: {output_path}")


def main() -> None:
    """메인 실행 함수"""
    # -------------------------------------------------------------------------
    # 커맨드라인 인자 파서 설정
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Llama 3.2 1B lm_eval 벤치마크 평가 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python run_lm_eval.py                              # 기본 태스크로 평가
  python run_lm_eval.py --tasks hellaswag arc_challenge  # 특정 태스크 지정
  python run_lm_eval.py --tasks all                  # 모든 정의된 태스크 평가
  python run_lm_eval.py --limit 0.1                  # 10% 샘플만 빠르게 테스트
  python run_lm_eval.py --batch_size 4               # 배치 사이즈 조정
  python run_lm_eval.py --num_fewshot 0              # zero-shot으로 평가
        """
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="평가할 태스크 이름들 (기본값: 주요 5개 태스크). 'all'을 넣으면 모든 정의된 태스크 실행",
    )
    parser.add_argument(
        "--batch_size",
        default="auto",
        help="배치 크기 (기본값: auto). 숫자를 넣으면 고정 배치 크기 사용",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="few-shot 예시 수 (기본값: 각 태스크의 권장값 사용)",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="평가할 샘플 비율 (0.0~1.0) 또는 샘플 수. 빠른 테스트용",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="결과 저장 경로 (기본값: ./eval_results/results_YYYYMMDD_HHMMSS.json)",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="사용 가능한 태스크 목록만 출력하고 종료",
    )

    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # 태스크 목록만 출력하는 모드
    # -------------------------------------------------------------------------
    if args.list_tasks:
        print_task_info(list(TASK_DESCRIPTIONS.keys()))
        return

    # -------------------------------------------------------------------------
    # 실행할 태스크 결정
    # -------------------------------------------------------------------------
    if args.tasks is None:
        task_names = DEFAULT_TASKS
    elif args.tasks == ["all"]:
        task_names = list(TASK_DESCRIPTIONS.keys())
    else:
        task_names = args.tasks

    # -------------------------------------------------------------------------
    # 배치 크기 처리
    # -------------------------------------------------------------------------
    batch_size = args.batch_size
    if batch_size != "auto":
        try:
            batch_size = int(batch_size)
        except ValueError:
            print(f"  ⚠️  잘못된 배치 크기: {batch_size}. 'auto'를 사용합니다.")
            batch_size = "auto"

    # -------------------------------------------------------------------------
    # 태스크 정보 출력
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("  🦙 Llama 3.2 1B lm_eval 벤치마크 평가")
    print(f"  모델 경로: {MODEL_PATH}")
    print(f"  시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print_task_info(task_names)

    # -------------------------------------------------------------------------
    # 평가 실행
    # -------------------------------------------------------------------------
    results = run_evaluation(
        task_names=task_names,
        batch_size=batch_size,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    # -------------------------------------------------------------------------
    # 결과 출력
    # -------------------------------------------------------------------------
    print_results(results)

    # -------------------------------------------------------------------------
    # 결과 저장
    # -------------------------------------------------------------------------
    if args.output:
        output_path = args.output
    else:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.json")

    save_results(results, output_path)

    print("\n" + "=" * 80)
    print("  ✅ 평가가 완료되었습니다!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
