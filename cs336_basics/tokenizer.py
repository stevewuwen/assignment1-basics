from dataclasses import dataclass
import os
from cs336_basics.pretokenization_example import find_chunk_boundaries
import regex as re
from multiprocessing import Pool
from collections import Counter, defaultdict
import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO
import tqdm

# regex for word
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

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
    # 读取完整的数据
    # 应该交给子进程去读
    
    num_processes = 10
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    with Pool(num_processes) as p:
        results = p.starmap(read_and_split, list(zip([input_path for _ in range(len(boundaries[:-1]))], boundaries[:-1], boundaries[1:])))
    c = Counter()
    for result in results:
        c.update(result)
    p.close()
    # c就是 preword 的个数，现在只需要增量去统计每一个token前后出现的次数就可以了
    # 接下来就把work转化为token了，目前的词汇表就是special token + 256 个基本字符
    merges: list[tuple[bytes, bytes]] = []
    vocab: dict[bytes, int]={} # bytes -> int
    # 初始化特殊字符
    vocab[b"<|endoftext|>"] = 0
    # 初始化前256个字节
    for i in range(256):
        vocab[bytes([i])] = len(vocab)
    
    # mp的key只有
    mp: dict[tuple[bytes], int] = {}
    # 进行分词
    for word, freq in c.items():
        word_bytes = word.encode('utf-8')
        # 当对一个bytes类型进行遍历，每次得到的都是int类型，范围在0-255之间
        mp[tuple(bytes([byte]) for byte in word_bytes)] = freq

    status_bar = tqdm.tqdm(desc="bpe分词器训练中", total=vocab_size-len(vocab))
    while len(vocab) < vocab_size:
        # 统计每一个词里面的相邻词出现的次数
        byte_pair_feq = defaultdict(int)
        for l, freq in mp.items():
            for first, second in zip(l[:-1], l[1:]):
                byte_pair_feq[(first, second)] += freq
        max_com: tuple[bytes, bytes] = tuple()
        max_freq: int = -1
        for item, freq in byte_pair_feq.items():
            if freq > max_freq:
                max_com, max_freq= item, freq
        merges.append(max_com)
        merged_pair = merges[-1]
        new_token = b''.join(max_com)
        vocab[new_token] = len(vocab)
        # 更新mp，防止采的重复的值
        new_mp: dict[tuple, int] = {}
        for word_tuple, freq in mp.items():
            new_word_tuple = []
            index = 0
            n = len(word_tuple)
            while index < n:
                if index < n-1 and (word_tuple[index], word_tuple[index+1]) == merged_pair:
                    new_word_tuple.append(new_token)
                    index+=2
                else:
                    new_word_tuple.append(word_tuple[index])
                    index+=1
            new_mp[tuple(new_word_tuple)] = freq
        mp = new_mp
        status_bar.update(1)
    new_vocab: dict[int, bytes] = {v:k for k, v in vocab.items()}
    return new_vocab, merges

def read_and_split(input_path: str, start: int, end: int)->Counter[str]:
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end-start).decode("utf-8", errors='ignore')
    combined_pat_str = "|".join("<|endoftext|>")+"|" + PAT.pattern
    combined_pat = re.compile(combined_pat_str)
    tokens = re.findall(combined_pat, chunk)
    return Counter(tokens)