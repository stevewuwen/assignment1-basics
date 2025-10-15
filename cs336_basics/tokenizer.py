from dataclasses import dataclass
import os

@dataclass
class Tokenizer:
    vocab: dict[int, bytes] # 把词语在词库里面的id转化为对应的bytes， 比如说经过bpe之后， 300代表"ab"
    merges: list[tuple[bytes, bytes]] # 合并起来的字节，比如说"ab"
    special_tokens: list[str] # 特殊token，不会被分词，会被直接转化为一个词库里面的id，比如说<|endoftext|>->30

def tokenzier_bpe_trainer(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.

    """
    # 读取
    raise NotImplementedError
    