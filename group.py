import argparse
import random
from typing import List

from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

def group_batches(ds: Dataset, batch_size: int, text_fields: List[str] = ["sentence"]):
    model = SentenceTransformer("all-mpnet-base-v2")
    sentences = []
    for ex in ds:
        sentences.append(" ".join([ex[field] for field in text_fields]))
    cosine_scores = util.cos_sim(embeddings, embeddings)
    min_score = torch.min(cosine_scores)

    remaining = set(range(len(ds)))
    new_ds = []
    for _ in tqdm(range(len(ds) // batch_size)):
        idx = random.choice(list(remaining))
        scores = cosine_scores[idx]
        sorted_scores, sorted_indices = torch.topk(scores, batch_size, largest=True, sorted=True)
        for i in sorted_indices:
            remaining.remove(i.item())
            new_ds.append(ds[i.item()])
            cosine_scores[:, i] = min_score - 100

    for idx in remaining:
        new_ds.append(ds[idx])
    
    del model
    del embeddings
    del cosine_scores
    new_ds = Dataset.from_list(new_ds)
    return Dataset.from_list(new_ds)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="glue")
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument("--text_fields", nargs="+", default=["sentence"])

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    dataset = load_dataset(args.dataset_name, args.dataset_config_name)

    dataset["train"] = group_batches(
        dataset["train"], batch_size=args.batch_size, text_fields=args.text_fields
    )

    dataset.save_to_disk(args.output_file)


if __name__ == "__main__":
    main()
    