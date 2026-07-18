from pathlib import Path

import faiss
import numpy as np


def main():
    project_root = Path(__file__).resolve().parent.parent

    embeddings_path = (
        project_root
        / "outputs"
        / "embeddings"
        / "image_embeddings.npy"
    )

    item_ids_path = (
        project_root
        / "outputs"
        / "embeddings"
        / "item_ids.npy"
    )

    output_dir = (
        project_root
        / "outputs"
        / "faiss"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "fashion.index"

    print("=" * 60)
    print("Loading embeddings...")

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embeddings not found:\n{embeddings_path}"
        )

    embeddings = np.load(embeddings_path).astype(np.float32)

    print(f"Embedding matrix shape: {embeddings.shape}")

    if len(embeddings.shape) != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if np.isnan(embeddings).any():
        raise ValueError("Embeddings contain NaN values.")

    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    num_vectors = embeddings.shape[0]

    print(f"Embedding dimension : {dimension}")
    print(f"Number of vectors   : {num_vectors}")

    print("Creating FAISS index...")
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"Saving index to:\n{index_path}")

    faiss.write_index(index, str(index_path))

    if item_ids_path.exists():
        item_ids = np.load(item_ids_path, allow_pickle=True)

        np.save(
            output_dir / "faiss_item_ids.npy",
            item_ids,
        )

        print(f"Saved {len(item_ids)} item IDs.")

    print("=" * 60)
    print("FAISS index successfully created.")
    print(f"Total vectors indexed: {index.ntotal}")
    print("=" * 60)


if __name__ == "__main__":
    main()
