import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def analyze_token_lengths(
    data, tokenizer_name="allenai/specter2_base", sample_size=None
):
    """
    데이터셋의 title + abstract 토큰 길이를 분석합니다.

    Args:
        data: 논문 데이터 리스트 (각 항목은 'title'과 'text' 또는 'abstract' 키를 가져야 함)
        tokenizer_name: 사용할 토크나이저 이름
        sample_size: 분석할 샘플 수 (None이면 전체 데이터)

    Returns:
        토큰 길이 통계 및 시각화
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # 샘플링 (필요한 경우)
    if sample_size and sample_size < len(data):
        import random

        data_sample = random.sample(data, sample_size)
    else:
        data_sample = data

    token_lengths = []
    truncated_count = 0
    max_token_limit = 512  # SPECTER2의 최대 토큰 길이

    for item in tqdm(data_sample, desc="Calculating token lengths"):
        title = item.get("title", "")

        # abstract 또는 text 필드 확인
        abstract = item.get("abstract", item.get("text", ""))

        # title + sep_token + abstract 형식으로 결합
        full_text = title + tokenizer.sep_token + abstract

        # 토큰화
        tokens = tokenizer.encode(full_text)
        token_length = len(tokens)
        token_lengths.append(token_length)

        # 잘림 여부 확인
        if token_length > max_token_limit:
            truncated_count += 1

    # 통계 계산
    token_lengths = np.array(token_lengths)
    stats = {
        "mean": np.mean(token_lengths),
        "median": np.median(token_lengths),
        "min": np.min(token_lengths),
        "max": np.max(token_lengths),
        "std": np.std(token_lengths),
        "truncated_percentage": (truncated_count / len(data_sample)) * 100,
        "truncated_count": truncated_count,
        "total_samples": len(data_sample),
    }

    # 백분위수 계산
    percentiles = [25, 50, 75, 90, 95, 99]
    for p in percentiles:
        stats[f"{p}th_percentile"] = np.percentile(token_lengths, p)

    # 여유 공간 계산 (512 토큰 한도 기준)
    remaining_tokens = max_token_limit - token_lengths
    remaining_tokens[remaining_tokens < 0] = 0  # 음수 값은 0으로 처리

    stats["avg_remaining_tokens"] = np.mean(remaining_tokens)
    stats["median_remaining_tokens"] = np.median(remaining_tokens)

    # 히스토그램 플롯
    plt.figure(figsize=(12, 6))
    plt.hist(token_lengths, bins=50, alpha=0.7)
    plt.axvline(
        x=max_token_limit,
        color="r",
        linestyle="--",
        label=f"Max token limit ({max_token_limit})",
    )
    plt.title("Distribution of Token Lengths (Title + Abstract)")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 주요 통계 표시
    for stat, value in stats.items():
        if isinstance(value, (int, float)):
            print(f"{stat}: {value:.2f}")
        else:
            print(f"{stat}: {value}")

    return stats, token_lengths, plt
