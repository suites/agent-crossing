import numpy as np


def validate_embedding_dimension(
    embedding: np.ndarray, *, expected_dimension: int
) -> None:
    """Validate embedding vector shape for runtime safety.

    Raises
    ------
    ValueError
        If embedding is not a 1D vector of the expected size.
    """
    if embedding.ndim != 1:
        raise ValueError("Embedding must be a 1D vector")
    if embedding.shape[0] != expected_dimension:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dimension}, got {embedding.shape[0]}"
        )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    두 벡터 a와 b 사이의 코사인 유사도를 계산한다.
    - 반환 범위: `[-1.0, 1.0]`

    공식:
    - `cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)`
    """
    if a.shape != b.shape:
        raise ValueError(
            f"Embedding vectors must have matching shapes: a={a.shape}, b={b.shape}"
        )
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def min_max_normalize(scores: list[float]) -> list[float]:
    """
    Min-Max 정규화를 수행한다.
    - 반환 범위: `[0.0, 1.0]`

    공식:
    - `norm_x = (x - min_x) / (max_x - min_x)`
    """
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if max_score == min_score:
        return [0.0 for _ in scores]
    return [(x - min_score) / (max_score - min_score) for x in scores]
