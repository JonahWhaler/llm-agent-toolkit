import logging
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


def reconstruct_chunk_v2(partial_chunk: list[str], level: str) -> str:
    """
        Reconstructs a text chunk from a list of text units.

        The reconstruction method depends on the specified level (e.g., "section"
        or "sentence").

        Args:
            partial_chunk (List[str]): A list of text units.
            level (str): The level of text unit ("section" or "sentence").

        Returns:
            output (str): 
            The reconstructed text chunk.
        """
    if level == "section":
        return "\n\n".join([chunk.strip() for chunk in partial_chunk])
    return " ".join([chunk.strip() for chunk in partial_chunk])


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


def all_within_chunk_size(
    lines: list[str], grouping: list[tuple[int, int]], chunk_size: int
) -> bool:
    """
    Check if all chunks in the grouping are within the specified chunk size.

    Args:
        lines (List[str]): A list of the original text units.
        grouping (List[Tuple[int, int]]): A list of tuples defining the boundaries of each chunk.
        chunk_size (int): The size of each chunk in terms of tokens.

    Returns:
        bool: True if all chunks are within the specified chunk size, False otherwise.
    """
    logger = logging.getLogger(__name__)

    for g_start, g_end in grouping:
        g_line = reconstruct_chunk_v2(lines[g_start:g_end], level="sentence")
        g_tc = estimate_token_count(g_line)
        if g_tc > chunk_size:
            logger.warning(
                f"Chunk {g_start}-{g_end} exceeds chunk size: {g_tc} > {chunk_size}"
            )
            return False

    return True
