import json
import glob
import os
from pathlib import Path
import sys
import bz2

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
from tqdm import tqdm

from lit_llama import Tokenizer
import lit_llama.packed_dataset as packed_dataset

def get_files(folder):
    subfolder_file_lists = {}
    for subdir, dirs, files in os.walk(folder):
        if subdir == folder:  # Skip the root folder
            continue
        file_list = []
        for file in files:
            if file.endswith(".jsonl.bz2"):
                # construct full file path
                file_path = subdir + os.sep + file
                file_list.append(file_path)
        subfolder_file_lists[subdir] = file_list
    return subfolder_file_lists

filenames_sample = [
    "arxiv_sample.jsonl",
    "book_sample.jsonl",
    "c4_sample.jsonl",
    "cc_2019-30_sample.jsonl",
    "cc_2020-05_sample.jsonl",
    "cc_2021-04_sample.jsonl",
    "cc_2022-05_sample.jsonl",
    "cc_2023-06_sample.jsonl",
    "github_sample.jsonl",
    "stackexchange_sample.jsonl",
    "wikipedia_sample.jsonl",
]

filename_sets = {
    "arxiv": "arxiv/arxiv*",
    "book": "book/book*",
    "c4": "c4/c4-train*",
    "common_crawl": "common_crawl/*",
    "github": "github/filtered*",
    "stackexchange": "stackexchange/stackexchange*",
    "wikipedia": "wikipedia/wiki*",
}

filename_sets = {
    "impresso": "../data/*.jsonl.bz2"
}

def prepare_sample(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match = ""
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    for name in filenames_sample:
        if match and match not in name:
            continue

        filepath = source_path / name

        if not filepath.is_file():
            raise RuntimeError(
                f"Input file not found at {filepath}. \n"
                "Make sure you download the data, e.g. wget -i https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through \n"
                "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T \n"
                "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
            )

        prefix, _ = os.path.splitext(name)

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=prefix,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        print(f"Processing {name}")

        with open(filepath, encoding="utf-8") as f:
            for row in tqdm(f):
                text = json.loads(row)["text"]
                text_ids = tokenizer.encode(text)
                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match: str = ""
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    subfolder_file_lists = get_files(source_path)

    for set_name, pattern in filename_sets.items():
        if match and match not in set_name:
            continue

        is_cc = set_name == "common_crawl"

        # filenames = glob.glob(os.path.join(source_path, pattern), recursive=True)

        # if not filenames:
        #     raise RuntimeError(
        #         f"No files matching {pattern} found at {source_path}. \n"
        #         "Make sure you download the data, e.g. wget -i https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through \n"
        #         "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T \n"
        #         "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
        #     )

        builder = packed_dataset.PackedDatasetBuilder(
            outdir=destination_path,
            prefix=set_name,
            chunk_size=chunk_size,
            sep_token=tokenizer.bos_id,
            dtype="auto",
            vocab_size=tokenizer.vocab_size,
        )

        for subdir, file_list in subfolder_file_lists.items():
            for filepath in file_list:
                # filepath = source_path / name

                print(f"Processing {filepath}")

                if is_cc:
                    with bz2.open(filepath, "rt") as f: #, encoding="utf-8"
                        for row in tqdm(f):
                            loaded_row = json.loads(row)
                            if "ft" in loaded_row:
                                text = loaded_row["ft"]
                                # print(text)
                                text_ids = tokenizer.encode(text)
                                builder.add_array(np.array(text_ids, dtype=builder.dtype))
                else:
                    with bz2.open(filepath, "rt") as f:
                        for row in tqdm(f):
                            loaded_row = json.loads(row)
                            if "ft" in loaded_row:
                                text = loaded_row["ft"]
                                text_ids = tokenizer.encode(text)
                                builder.add_array(np.array(text_ids, dtype=builder.dtype))

        builder.write_reminder()


def prepare(
    source_path: Path = Path("../data/"), #RedPajama-Data-1T-Sample
    tokenizer_path: Path = Path("../checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("../data/impresso"),
    chunk_size: int = 2049 * 1024,  # 2048 block size + 1 for causal (from LLama), 1024 blocks
    sample: bool = False,
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    if sample:
        prepare_sample(
            source_path=source_path,
            tokenizer_path=tokenizer_path,
            destination_path=destination_path,
            chunk_size=chunk_size,
            match=match,
        )
    else:
        prepare_full(
            source_path=source_path,
            tokenizer_path=tokenizer_path,
            destination_path=destination_path,
            chunk_size=chunk_size,
            match=match,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(prepare)
