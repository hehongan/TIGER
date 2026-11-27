import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_args():
    data_dir = Path(__file__).resolve().parent / "Beauty"
    parser = argparse.ArgumentParser(description="Build time-decayed user embeddings")
    parser.add_argument("--train_path", type=Path, default=data_dir / "train.parquet")
    parser.add_argument("--item_emb_path", type=Path, default=data_dir / "item_emb.parquet")
    parser.add_argument("--output_path", type=Path, default=data_dir / "user_emb.parquet")
    parser.add_argument("--decay_alpha", type=float, default=0.1)
    return parser.parse_args()


def aggregate_users(train_data, item_embeddings, decay_alpha):
    item_ids = item_embeddings["ItemID"].astype(int).to_numpy()
    embedding_matrix = np.stack(item_embeddings["embedding"].to_numpy()).astype(np.float32)
    item_to_row = {item_id: row for row, item_id in enumerate(item_ids)}

    train_data = train_data.sort_values("user")
    user_ids = train_data["user"].astype(int).to_numpy()
    expected_ids = np.arange(1, len(user_ids) + 1)
    if not np.array_equal(user_ids, expected_ids):
        raise ValueError("User IDs must be contiguous and start from 1")

    user_embeddings = []
    for row in tqdm(train_data.itertuples(index=False), total=len(train_data), desc="Aggregate users"):
        # train.parquet excludes the validation and test targets.
        history = list(row.history) + [row.target]
        rows = [item_to_row[int(item_id)] for item_id in history]
        positions = np.arange(len(rows), dtype=np.float32)
        weights = np.exp(-decay_alpha * (len(rows) - 1 - positions))
        weights /= weights.sum()
        embedding = np.average(embedding_matrix[rows], axis=0, weights=weights)
        user_embeddings.append(embedding.astype(np.float32))

    return pd.DataFrame({"UserID": user_ids, "embedding": user_embeddings})


if __name__ == "__main__":
    args = parse_args()
    train_df = pd.read_parquet(args.train_path)
    item_emb_df = pd.read_parquet(args.item_emb_path)
    user_emb_df = aggregate_users(train_df, item_emb_df, args.decay_alpha)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    user_emb_df.to_parquet(args.output_path, index=False)
    print(f"User embeddings saved to {args.output_path}: {user_emb_df.shape}")
