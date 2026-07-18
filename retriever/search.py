from pathlib import Path
import sqlite3

import faiss
import numpy as np
from datasets import load_from_disk

from models.openclip import OpenCLIPEmbedder
from retriever.rerank import QwenReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "deepfashion_multimodal"
INDEX_PATH = PROJECT_ROOT / "outputs" / "faiss" / "fashion.index"
DB_PATH = PROJECT_ROOT / "outputs" / "metadata" / "fashion.db"
EMBEDDINGS_PATH = PROJECT_ROOT / "outputs" / "embeddings" / "image_embeddings.npy"


class FashionRetriever:

    def __init__(self):

        print("Loading dataset...")
        self.dataset = load_from_disk(str(DATASET_PATH))

        print("Loading OpenCLIP...")
        self.embedder = OpenCLIPEmbedder()

        print("Loading FAISS...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading embeddings...")
        self.embeddings = np.load(
            EMBEDDINGS_PATH,
            mmap_mode="r",
        )

        print("Loading Qwen reranker...")
        self.reranker = QwenReranker()

        print(f"Dataset size: {len(self.dataset)}")

    ##########################################################################

    def _candidate_indices(self, query):

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        q = query.lower()

        conditions = []

        # Gender
        if any(x in q for x in ["women", "woman", "female"]):
            conditions.append("LOWER(category1)='women'")

        if any(x in q for x in ["men", "man", "male"]):
            conditions.append("LOWER(category1)='men'")

        # Tops
        if "top" in q or "tops" in q:

            conditions.append("""
            (
                LOWER(category2) LIKE '%tee%'
                OR LOWER(category2) LIKE '%shirt%'
                OR LOWER(category2) LIKE '%tank%'
                OR LOWER(category2) LIKE '%blouse%'
                OR LOWER(category2) LIKE '%sweater%'
            )
            """)

        # Denim
        if any(x in q for x in ["jeans", "jean", "denim"]):

            conditions.append("""
            (
                LOWER(category2) LIKE '%denim%'
            )
            """)

        # Dress
        if "dress" in q:

            conditions.append("""
            (
                LOWER(category2) LIKE '%dress%'
            )
            """)

        sql = "SELECT idx FROM fashion_items"

        if conditions:
            sql += " WHERE " + " AND ".join(conditions)

        cur.execute(sql)

        rows = cur.fetchall()

        conn.close()

        return [x[0] for x in rows]

    ##########################################################################

    def search(self, query, k=5):

        candidate_pool = max(20, k * 4)

        query_embedding = self.embedder.encode_text(query).astype(np.float32)

        query_embedding = query_embedding.reshape(1, -1)

        faiss.normalize_L2(query_embedding)

        candidates = self._candidate_indices(query)

        ############################################################

        if len(candidates):

            vectors = self.embeddings[candidates].astype(np.float32)

            faiss.normalize_L2(vectors)

            temp = faiss.IndexFlatIP(vectors.shape[1])

            temp.add(vectors)

            scores, ids = temp.search(
                query_embedding,
                min(candidate_pool, len(candidates)),
            )

            final_ids = [candidates[i] for i in ids[0]]

        else:

            scores, ids = self.index.search(
                query_embedding,
                candidate_pool,
            )

            final_ids = ids[0]

        ############################################################

        results = []

        for score, idx in zip(scores[0], final_ids):

            sample = self.dataset[int(idx)]

            results.append(
                {
                    "image": sample["image"],
                    "score": round(float(score), 4),
                    "item_ID": sample["item_ID"],
                    "category1": sample["category1"],
                    "category2": sample["category2"],
                    "category3": sample["category3"],
                    "text": sample["text"],
                }
            )

        ############################################################

        print("Running Qwen reranking...")

        reranked = self.reranker.rerank(
            query,
            results,
        )

        return reranked[:k]


##########################################################################


if __name__ == "__main__":

    retriever = FashionRetriever()

    results = retriever.search(
        "blue denim jeans",
        k=5,
    )

    for i, r in enumerate(results, 1):

        print("=" * 80)

        print(f"Rank : {i}")

        print(f"CLIP : {r['score']}")

        print(f"Qwen : {r['llm_score']}")

        print(r["category1"], "/", r["category2"])

        print(r["reason"])
