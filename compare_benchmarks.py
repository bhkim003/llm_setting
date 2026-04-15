"""
=============================================================================
  벤치마크 결과 비교 리포트 생성기 (compare_benchmarks.py)
=============================================================================

이 스크립트는 run_lm_eval.py가 저장한 결과 JSON을 읽어,
대표 공개 모델 기준치와 비교한 텍스트 리포트(.log)를 생성합니다.

사용 예시:
    python compare_benchmarks.py --result eval_results/results_20260415_225718.json
    python compare_benchmarks.py --result eval_results/results_20260415_225718.json --output eval_results/report.log
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, Optional


# -----------------------------------------------------------------------------
# 공개 리더보드/모델 카드에서 널리 알려진 대표 수치(근사치) 스냅샷
# -----------------------------------------------------------------------------
# 주의:
# - 평가 세팅(버전, prompt format, few-shot, tokenizer, 정규화 방식)에 따라
#   수치는 달라질 수 있습니다.
# - 본 비교는 "대략적인 위치 파악" 용도입니다.
# -----------------------------------------------------------------------------
REFERENCE_MODELS: Dict[str, Dict[str, float]] = {
    "Llama-2-7B": {
        "hellaswag": 0.78,
        "arc_challenge": 0.53,
        "winogrande": 0.72,
        "piqa": 0.79,
        "boolq": 0.73,
    },
    "Mistral-7B-v0.1": {
        "hellaswag": 0.81,
        "arc_challenge": 0.60,
        "winogrande": 0.78,
        "piqa": 0.83,
        "boolq": 0.83,
    },
    "Qwen2.5-7B-Instruct": {
        "hellaswag": 0.85,
        "arc_challenge": 0.66,
        "winogrande": 0.80,
        "piqa": 0.84,
        "boolq": 0.85,
    },
    "Llama-3.1-8B-Instruct": {
        "hellaswag": 0.85,
        "arc_challenge": 0.67,
        "winogrande": 0.80,
        "piqa": 0.84,
        "boolq": 0.85,
    },
}

# perplexity 기준치(근사치, 낮을수록 좋음)
PPL_REFERENCE_MODELS: Dict[str, Dict[str, float]] = {
    "Llama-2-7B": {
        "wikitext_word_perplexity": 5.5,
    },
    "Mistral-7B-v0.1": {
        "wikitext_word_perplexity": 4.8,
    },
    "Qwen2.5-7B-Instruct": {
        "wikitext_word_perplexity": 4.5,
    },
    "Llama-3.1-8B-Instruct": {
        "wikitext_word_perplexity": 4.7,
    },
}

# 어떤 지표를 우선 읽을지(태스크별)
PRIMARY_METRIC_KEYS = {
    "hellaswag": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "winogrande": ["acc,none", "acc_norm,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "boolq": ["acc,none", "acc_norm,none"],
}


def _pick_metric(task_result: Dict[str, object], task_name: str) -> Optional[float]:
    """태스크 결과에서 비교용 메트릭을 선택한다."""
    candidates = PRIMARY_METRIC_KEYS.get(task_name, ["acc,none", "acc_norm,none"])

    for key in candidates:
        value = task_result.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    for value in task_result.values():
        if isinstance(value, (int, float)):
            return float(value)

    return None


def load_result_scores(result_path: str) -> Dict[str, float]:
    """JSON 결과 파일에서 태스크별 점수를 로딩한다."""
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        return {}

    scores: Dict[str, float] = {}
    for task_name, task_result in raw_results.items():
        if not isinstance(task_result, dict):
            continue

        metric = _pick_metric(task_result, task_name)
        if metric is not None:
            scores[task_name] = metric

    return scores


def load_wikitext_perplexity(result_path: str) -> Optional[float]:
    """JSON 결과에서 wikitext word_perplexity를 추출한다."""
    with open(result_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_results = payload.get("results", {})
    if not isinstance(raw_results, dict):
        return None

    wikitext_result = raw_results.get("wikitext")
    if not isinstance(wikitext_result, dict):
        return None

    for key in ["word_perplexity,none", "word_perplexity"]:
        value = wikitext_result.get(key)
        if isinstance(value, (int, float)):
            return float(value)

    for metric_name, value in wikitext_result.items():
        if "word_perplexity" in metric_name and isinstance(value, (int, float)):
            return float(value)

    return None


def build_report(
    result_path: str,
    model_name: str,
    scores: Dict[str, float],
    wikitext_word_perplexity: Optional[float],
) -> str:
    """비교 리포트 문자열 생성"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("=" * 88)
    lines.append("LLM 벤치마크 비교 리포트")
    lines.append("=" * 88)
    lines.append(f"생성 시각: {now}")
    lines.append(f"대상 모델: {model_name}")
    lines.append(f"결과 파일: {result_path}")
    lines.append("")
    lines.append("[비교 기준]")
    lines.append("- 공개 리포트 기반 대표 수치(근사치)이며, 세팅 차이로 절대 비교는 제한적입니다.")
    lines.append("- 상대적 위치(입문/중간/상위)를 빠르게 파악하는 용도로 해석하세요.")
    lines.append("")

    tasks = ["hellaswag", "arc_challenge", "winogrande", "piqa", "boolq"]

    header = f"{'Task':<15} {'YourModel':>10} {'Llama-2-7B':>12} {'BestRef':>10} {'GapToBest':>11}"
    lines.append(header)
    lines.append("-" * len(header))

    avg_user = []
    avg_l2 = []
    avg_best = []

    for task in tasks:
        your_score = scores.get(task)
        l2_score = REFERENCE_MODELS["Llama-2-7B"].get(task)

        best_model = None
        best_score = None
        for ref_model, ref_scores in REFERENCE_MODELS.items():
            s = ref_scores.get(task)
            if s is None:
                continue
            if best_score is None or s > best_score:
                best_score = s
                best_model = ref_model

        if your_score is None:
            lines.append(f"{task:<15} {'N/A':>10} {l2_score * 100:>11.2f}% {best_score * 100:>9.2f}% {'N/A':>11}")
            continue

        gap_to_best = your_score - best_score if best_score is not None else None
        gap_str = "N/A" if gap_to_best is None else f"{gap_to_best * 100:+.2f}p"

        lines.append(
            f"{task:<15} {your_score * 100:>9.2f}% {l2_score * 100:>11.2f}% {best_score * 100:>9.2f}% {gap_str:>11}"
        )
        lines.append(f"{'':<15} {'':>10} {'':>12} {'':>10} ({best_model})")

        avg_user.append(your_score)
        avg_l2.append(l2_score)
        avg_best.append(best_score)

    lines.append("")
    lines.append("[요약]")

    if avg_user:
        user_mean = sum(avg_user) / len(avg_user)
        l2_mean = sum(avg_l2) / len(avg_l2)
        best_mean = sum(avg_best) / len(avg_best)

        lines.append(f"- 내 모델 평균 점수: {user_mean * 100:.2f}%")
        lines.append(f"- Llama-2-7B 평균: {l2_mean * 100:.2f}%")
        lines.append(f"- 상위 기준 평균: {best_mean * 100:.2f}%")
        lines.append(f"- Llama-2-7B 대비 차이: {(user_mean - l2_mean) * 100:+.2f}p")
        lines.append(f"- 상위 기준 대비 차이: {(user_mean - best_mean) * 100:+.2f}p")

        if user_mean >= l2_mean:
            lines.append("- 해석: 입문 기준(Llama-2-7B)과 비슷하거나 상회합니다.")
        else:
            lines.append("- 해석: 입문 기준(Llama-2-7B) 대비 아직 개선 여지가 큽니다.")
    else:
        lines.append("- 결과에서 비교 가능한 태스크 점수를 찾지 못했습니다.")

    lines.append("")
    lines.append("[Perplexity 비교 (WikiText)]")
    lines.append("- 지표 해석: 낮을수록 좋음")

    if wikitext_word_perplexity is None:
        lines.append("- wikitext perplexity를 결과에서 찾지 못했습니다.")
        lines.append("  (run_lm_eval.py에서 wikitext 태스크가 실행되었는지 확인하세요)")
    else:
        lines.append(f"- 내 모델 word_perplexity: {wikitext_word_perplexity:.4f}")

        l2_ppl = PPL_REFERENCE_MODELS["Llama-2-7B"].get("wikitext_word_perplexity")
        lines.append(f"- Llama-2-7B word_perplexity(기준): {l2_ppl:.4f}")
        lines.append(f"- Llama-2-7B 대비 차이: {wikitext_word_perplexity - l2_ppl:+.4f}")

        best_model = None
        best_ppl = None
        for ref_model, ref_scores in PPL_REFERENCE_MODELS.items():
            ref_ppl = ref_scores.get("wikitext_word_perplexity")
            if ref_ppl is None:
                continue
            if best_ppl is None or ref_ppl < best_ppl:
                best_ppl = ref_ppl
                best_model = ref_model

        lines.append(f"- 참고 상위 기준(best) word_perplexity: {best_ppl:.4f} ({best_model})")
        lines.append(f"- 상위 기준 대비 차이: {wikitext_word_perplexity - best_ppl:+.4f}")

        if wikitext_word_perplexity <= l2_ppl:
            lines.append("- 해석: perplexity 기준에서 Llama-2-7B 이상 수준입니다.")
        else:
            lines.append("- 해석: perplexity 기준에서 Llama-2-7B 대비 개선 여지가 있습니다.")

    lines.append("")
    lines.append("[참고]")
    lines.append("- 본 리포트는 모델 상대 위치를 빠르게 점검하기 위한 자동 보고서입니다.")
    lines.append("- 엄밀 비교가 필요하면 동일 harness 버전/프롬프트/샷 수로 재평가하세요.")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="lm_eval 결과를 기준 모델과 비교해 .log 리포트를 생성합니다.")
    parser.add_argument("--result", required=True, help="run_lm_eval.py 결과 JSON 파일 경로")
    parser.add_argument("--output", default=None, help="리포트 저장 경로(.log). 기본: 결과 파일명 기반 자동 생성")
    parser.add_argument("--model-name", default="Llama-3.2-1B (current run)", help="리포트에 표시할 대상 모델 이름")
    args = parser.parse_args()

    result_path = args.result
    if not os.path.isfile(result_path):
        raise FileNotFoundError(f"결과 파일을 찾을 수 없습니다: {result_path}")

    if args.output:
        output_path = args.output
    else:
        stem, _ = os.path.splitext(result_path)
        output_path = f"{stem}_comparison.log"

    scores = load_result_scores(result_path)
    wikitext_word_perplexity = load_wikitext_perplexity(result_path)
    report = build_report(
        result_path=result_path,
        model_name=args.model_name,
        scores=scores,
        wikitext_word_perplexity=wikitext_word_perplexity,
    )

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"비교 리포트 저장 완료: {output_path}")


if __name__ == "__main__":
    main()
