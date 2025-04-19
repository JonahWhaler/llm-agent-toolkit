import charade
from math import ceil


def reconstruct_chunk(partial_chunk: list[str]) -> str:
    """
    Reconstructs a single text string from a list of partial chunks.

    This method ensures proper spacing between chunks and correctly handles punctuation.

    Args:
        partial_chunk (list[str]): A list of text segments to be combined.

    Returns:
        str: The reconstructed text string.
    """
    reconstructed = []
    previous_chunk = ""

    for chunk in partial_chunk:
        if previous_chunk:
            if "#" in chunk or "`" in chunk:
                reconstructed.append("\n")
            elif (
                chunk not in {".", "?", "!", "\n", "\t"}
                and previous_chunk != "\n"
            ):
                reconstructed.append(" ")
        reconstructed.append(chunk)
        previous_chunk = chunk

    return "".join(reconstructed)


def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a give text string.

    This is a naive estimation and may not be accurate for all tokenization methods.
    """
    byte_sentence = text.encode("utf-8")
    result = charade.detect(byte_sentence)
    is_ascii: bool = result["encoding"] == "ascii"

    if is_ascii:
        return ceil(len(text) * 0.5)

    return ceil(len(text) * 0.6)
