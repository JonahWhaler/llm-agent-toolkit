from .._chunkers import Chunker, UniformInitializer


class FixedCharacterChunker(Chunker):
    def __init__(self, config: dict):
        super().__init__(config)

    def split(self, long_text: str) -> list[str]:
        chunk_size: int = self.config.get("chunk_size", None)
        stride_rate: float = self.config.get("stride_rate", 1)
        if chunk_size is None:
            raise KeyError("'chunk_size'")
        stride: int = int(chunk_size * stride_rate)
        output_list = []
        for offset in range(0, len(long_text), stride):
            chunk = long_text[offset : offset + chunk_size]
            output_list.append(chunk)
        return output_list


class FixedGroupChunker(Chunker):
    def __init__(self, config: dict):
        super().__init__(config)

    def split(self, long_text: str) -> list[str]:
        k: int = self.config.get("k", None)
        if k is None:
            raise KeyError("'K'")
        lines = self._split(long_text)
        initializer = UniformInitializer(len(lines), k, "back")
        grouping = initializer.init()
        output_list = []
        for g_start, g_end in grouping:
            chunk = lines[g_start:g_end]
            g_string = self.reconstruct_chunk(chunk)
            output_list.append(g_string)
        return output_list
