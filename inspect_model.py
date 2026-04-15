"""
=============================================================================
  Llama 3.2 1B 모델 구조 분석 스크립트 (inspect_model.py)
=============================================================================

이 스크립트는 Hugging Face에서 다운로드한 Llama 3.2 1B 모델을 불러와서
모델의 내부 구조(아키텍처)를 상세히 출력하고, 각 모듈이 무슨 역할을 하는지
한글 주석으로 설명합니다.

LLM(대규모 언어 모델)을 처음 접하는 분들이 모델 내부를 이해하는 데 도움이
되도록 작성했습니다.

사용법:
    python inspect_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# =============================================================================
# 1. 모델 경로 설정
# =============================================================================
# 허깅페이스에서 미리 다운로드해둔 Llama 3.2 1B 모델의 로컬 경로
MODEL_PATH = "/data2/llm_download/Llama-3.2-1B"


def print_separator(title: str) -> None:
    """구분선과 제목을 출력하는 헬퍼 함수"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def inspect_config() -> None:
    """
    모델의 설정(config) 정보를 출력합니다.

    config에는 모델의 하이퍼파라미터들이 저장되어 있습니다.
    - hidden_size: 임베딩 벡터의 차원 수
    - num_hidden_layers: 트랜스포머 레이어(블록)의 개수
    - num_attention_heads: 어텐션 헤드의 개수
    - vocab_size: 모델이 알고 있는 토큰(단어/서브워드)의 총 개수
    등등
    """
    print_separator("1단계: 모델 설정(Config) 확인")

    # AutoConfig: 모델 폴더에서 config.json을 읽어서 설정 객체를 생성
    config = AutoConfig.from_pretrained(MODEL_PATH)

    print("\n[모델 설정 전체 출력]")
    print(config)

    print("\n[주요 하이퍼파라미터 설명]")
    print(f"  - hidden_size (은닉 차원): {config.hidden_size}")
    print("    → 각 토큰이 모델 내부에서 표현되는 벡터의 크기입니다.")
    print(f"  - num_hidden_layers (트랜스포머 레이어 수): {config.num_hidden_layers}")
    print("    → 트랜스포머 블록이 몇 층 쌓여있는지를 나타냅니다.")
    print("    → 층이 많을수록 모델이 더 복잡한 패턴을 학습할 수 있습니다.")
    print(f"  - num_attention_heads (어텐션 헤드 수): {config.num_attention_heads}")
    print("    → 멀티헤드 어텐션에서 병렬로 동작하는 어텐션의 개수입니다.")
    print(f"  - num_key_value_heads (KV 헤드 수): {config.num_key_value_heads}")
    print("    → GQA(Grouped Query Attention)에서 Key/Value에 사용하는 헤드 수입니다.")
    print("    → 어텐션 헤드 수보다 적으면 GQA를 사용하여 메모리를 절약합니다.")
    print(f"  - intermediate_size (FFN 중간 차원): {config.intermediate_size}")
    print("    → Feed-Forward Network(FFN)의 중간 레이어 크기입니다.")
    print(f"  - vocab_size (어휘 크기): {config.vocab_size}")
    print("    → 모델이 인식할 수 있는 토큰(단어/서브워드)의 총 개수입니다.")
    print(f"  - max_position_embeddings (최대 시퀀스 길이): {config.max_position_embeddings}")
    print("    → 모델이 한 번에 처리할 수 있는 최대 토큰 수입니다.")
    print(f"  - rms_norm_eps (RMSNorm 엡실론): {config.rms_norm_eps}")
    print("    → 정규화 시 0으로 나누는 것을 방지하기 위한 아주 작은 값입니다.")
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta")

    if rope_theta is not None:
        print(f"  - rope_theta (RoPE 베이스 주파수): {rope_theta}")
        print("    → Rotary Position Embedding에서 사용하는 기저 주파수입니다.")
        print("    → 위치 정보를 인코딩하는 데 사용됩니다.")
    else:
        print("  - rope_theta (RoPE 베이스 주파수): 설정에서 찾지 못했습니다.")


def inspect_model_architecture() -> None:
    """
    모델을 실제로 로딩하고, 내부 아키텍처를 상세히 출력합니다.

    Llama 모델의 전체 구조:
    ┌─────────────────────────────────────────┐
    │  LlamaForCausalLM (최상위 래퍼)          │
    │  ├── model (LlamaModel)                 │
    │  │   ├── embed_tokens (임베딩 레이어)     │
    │  │   ├── layers (트랜스포머 블록 x N)     │
    │  │   │   ├── self_attn (셀프 어텐션)      │
    │  │   │   ├── mlp (피드포워드 네트워크)     │
    │  │   │   ├── input_layernorm             │
    │  │   │   └── post_attention_layernorm    │
    │  │   ├── norm (최종 RMSNorm)             │
    │  │   └── rotary_emb (RoPE)              │
    │  └── lm_head (출력 레이어)               │
    └─────────────────────────────────────────┘
    """
    print_separator("2단계: 모델 아키텍처 상세 분석")

    print("\n모델을 로딩 중입니다... (약간의 시간이 소요될 수 있습니다)")

    # -------------------------------------------------------------------------
    # 모델 로딩
    # -------------------------------------------------------------------------
    # AutoModelForCausalLM: 인과적 언어 모델(Causal LM)을 자동으로 로드
    # - "Causal"이란 왼쪽에서 오른쪽으로만 텍스트를 생성하는 방식을 의미
    # - torch_dtype=torch.float16: 메모리 절약을 위해 16비트 부동소수점 사용
    # - device_map="auto": 사용 가능한 GPU에 자동으로 모델을 배치
    #   (GPU가 없으면 CPU에 로드됨)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # FP16으로 메모리 절약 (1B 모델 ≈ 약 2GB)
        device_map="auto",          # GPU 자동 배치
    )

    # -------------------------------------------------------------------------
    # 전체 모델 구조 출력
    # -------------------------------------------------------------------------
    print("\n[전체 모델 구조]")
    print(model)

    # -------------------------------------------------------------------------
    # 각 모듈별 상세 설명
    # -------------------------------------------------------------------------
    print_separator("3단계: 각 모듈별 상세 설명")

    # --- embed_tokens (토큰 임베딩 레이어) ---
    print("\n[1] embed_tokens (토큰 임베딩 레이어)")
    print(f"    {model.model.embed_tokens}")
    print("    → 역할: 입력 토큰 ID(정수)를 고차원 벡터로 변환합니다.")
    print("    → 예시: 토큰 ID 1234 → [0.12, -0.34, 0.56, ...] (hidden_size 차원의 벡터)")
    print("    → 이 벡터가 트랜스포머 레이어의 입력이 됩니다.")

    # --- 트랜스포머 레이어 ---
    num_layers = len(model.model.layers)
    print(f"\n[2] layers (트랜스포머 블록) - 총 {num_layers}개")
    print("    → 역할: 모델의 핵심 부분으로, 입력을 반복적으로 처리하여")
    print("      문맥을 이해하고 다음 토큰을 예측할 수 있는 표현을 만듭니다.")

    # 첫 번째 레이어를 예시로 상세 분석
    first_layer = model.model.layers[0]

    # --- Self Attention ---
    print(f"\n  [2-1] self_attn (셀프 어텐션) - 레이어 0 기준")
    print(f"       {first_layer.self_attn}")
    print("       → 역할: 입력 시퀀스의 각 토큰이 다른 모든 토큰과의 관계를")
    print("         계산하여 문맥 정보를 반영합니다.")
    print("       → '나는 사과를 먹었다'에서 '먹었다'가 '사과를'에 주목하는 것처럼,")
    print("         어떤 토큰이 어떤 토큰에 주의를 기울여야 하는지 학습합니다.")

    # Q, K, V, O 프로젝션 레이어 설명
    print(f"\n    [2-1-a] q_proj (Query 프로젝션)")
    print(f"            {first_layer.self_attn.q_proj}")
    print("            → 역할: 입력을 Query 벡터로 변환합니다.")
    print("            → Query는 '내가 무엇을 찾고 있는가'를 나타냅니다.")

    print(f"\n    [2-1-b] k_proj (Key 프로젝션)")
    print(f"            {first_layer.self_attn.k_proj}")
    print("            → 역할: 입력을 Key 벡터로 변환합니다.")
    print("            → Key는 '나는 어떤 정보를 가지고 있는가'를 나타냅니다.")

    print(f"\n    [2-1-c] v_proj (Value 프로젝션)")
    print(f"            {first_layer.self_attn.v_proj}")
    print("            → 역할: 입력을 Value 벡터로 변환합니다.")
    print("            → Value는 '실제로 전달할 정보의 내용'입니다.")

    print(f"\n    [2-1-d] o_proj (Output 프로젝션)")
    print(f"            {first_layer.self_attn.o_proj}")
    print("            → 역할: 어텐션 결과를 원래 차원으로 다시 변환합니다.")
    print("            → 여러 어텐션 헤드의 결과를 합친 후 최종 출력을 만듭니다.")

    # --- MLP (Feed-Forward Network) ---
    print(f"\n  [2-2] mlp (Multi-Layer Perceptron / 피드포워드 네트워크)")
    print(f"       {first_layer.mlp}")
    print("       → 역할: 어텐션이 모은 정보를 비선형 변환하여 더 풍부한 표현을 만듭니다.")
    print("       → Llama는 SwiGLU 활성화 함수를 사용합니다.")

    print(f"\n    [2-2-a] gate_proj (게이트 프로젝션)")
    print(f"            {first_layer.mlp.gate_proj}")
    print("            → 역할: SwiGLU에서 게이팅(정보 흐름 제어)에 사용됩니다.")

    print(f"\n    [2-2-b] up_proj (업 프로젝션)")
    print(f"            {first_layer.mlp.up_proj}")
    print("            → 역할: 입력을 더 높은 차원(intermediate_size)으로 확장합니다.")

    print(f"\n    [2-2-c] down_proj (다운 프로젝션)")
    print(f"            {first_layer.mlp.down_proj}")
    print("            → 역할: 확장된 차원을 다시 원래 차원(hidden_size)으로 줄입니다.")

    # --- Layer Normalization ---
    print(f"\n  [2-3] input_layernorm (입력 레이어 정규화)")
    print(f"       {first_layer.input_layernorm}")
    print("       → 역할: 셀프 어텐션에 입력되기 전에 값을 정규화합니다.")
    print("       → RMSNorm을 사용하여 학습 안정성을 높입니다.")
    print("       → Pre-Norm 구조: 레이어 입력 전에 정규화를 수행합니다.")

    print(f"\n  [2-4] post_attention_layernorm (어텐션 후 레이어 정규화)")
    print(f"       {first_layer.post_attention_layernorm}")
    print("       → 역할: MLP에 입력되기 전에 값을 정규화합니다.")

    # --- 최종 Norm ---
    print(f"\n[3] norm (최종 RMSNorm)")
    print(f"    {model.model.norm}")
    print("    → 역할: 마지막 트랜스포머 레이어의 출력을 정규화합니다.")
    print("    → lm_head에 입력되기 전 최종 정규화 단계입니다.")

    # --- LM Head (출력 레이어) ---
    print(f"\n[4] lm_head (언어 모델 헤드)")
    print(f"    {model.lm_head}")
    print("    → 역할: 트랜스포머의 출력을 어휘(vocabulary) 크기의 로짓(logits)으로 변환합니다.")
    print("    → 이 로짓에 softmax를 적용하면 다음 토큰의 확률 분포가 됩니다.")
    print("    → 예시: [0.01, 0.02, ..., 0.85, ...] → '사과' 토큰이 85% 확률로 다음에 올 것")

    # -------------------------------------------------------------------------
    # 파라미터 통계
    # -------------------------------------------------------------------------
    print_separator("4단계: 모델 파라미터 통계")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  총 파라미터 수: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"  학습 가능 파라미터 수: {trainable_params:,} ({trainable_params / 1e9:.2f}B)")
    print(f"  모델 메모리 (FP16 기준): 약 {total_params * 2 / 1e9:.2f} GB")

    print("\n[레이어별 파라미터 수]")
    for name, param in model.named_parameters():
        # 첫 번째 레이어와 마지막 레이어만 출력 (전부 출력하면 너무 김)
        if "layers.0." in name or "layers." not in name:
            print(f"  {name:60s} | shape: {str(param.shape):30s} | 파라미터 수: {param.numel():>12,}")


def inspect_tokenizer() -> None:
    """
    토크나이저를 분석합니다.

    토크나이저란?
    → 텍스트를 모델이 이해할 수 있는 숫자(토큰 ID)로 변환하는 도구입니다.
    → 예시: "안녕하세요" → [12345, 67890] (토큰 ID 리스트)
    """
    print_separator("5단계: 토크나이저 분석")

    # AutoTokenizer: 모델에 맞는 토크나이저를 자동으로 로드
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"\n  토크나이저 타입: {type(tokenizer).__name__}")
    print(f"  어휘 크기: {tokenizer.vocab_size:,}")
    print(f"  모델 최대 길이: {tokenizer.model_max_length:,}")
    print(f"  특수 토큰들:")
    print(f"    - BOS (시작) 토큰: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
    print(f"    - EOS (종료) 토큰: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
    if tokenizer.pad_token:
        print(f"    - PAD (패딩) 토큰: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")
    else:
        print("    - PAD (패딩) 토큰: 설정되지 않음")

    # 토크나이저 동작 예시
    print("\n[토크나이저 동작 예시]")
    test_texts = [
        "Hello, how are you?",
        "대한민국의 수도는 서울입니다.",
        "The capital of South Korea is Seoul.",
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded_tokens = [tokenizer.decode([t]) for t in tokens]
        print(f"\n  입력: '{text}'")
        print(f"  토큰 ID: {tokens}")
        print(f"  디코딩된 토큰: {decoded_tokens}")
        print(f"  토큰 수: {len(tokens)}")


def main() -> None:
    """메인 실행 함수"""
    print("=" * 80)
    print("  🦙 Llama 3.2 1B 모델 구조 분석기")
    print("  모델 경로: " + MODEL_PATH)
    print("=" * 80)

    # 1단계: 모델 설정 확인 (가벼운 작업 - 모델 로딩 불필요)
    inspect_config()

    # 2~4단계: 모델 로딩 및 아키텍처 분석
    inspect_model_architecture()

    # 5단계: 토크나이저 분석
    inspect_tokenizer()

    print_separator("분석 완료!")
    print("\n  모든 분석이 완료되었습니다.")
    print("  이제 run_lm_eval.py를 실행하여 벤치마크 평가를 진행할 수 있습니다.")
    print()


if __name__ == "__main__":
    main()
