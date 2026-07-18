from pathlib import Path
import sqlite3

from datasets import load_from_disk
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "dataset" / "deepfashion_multimodal"
DB_PATH = PROJECT_ROOT / "outputs" / "metadata" / "fashion.db"

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def main():

    print("Loading dataset...")
    dataset = load_from_disk(str(DATASET_PATH))

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS fashion_items(
        idx INTEGER PRIMARY KEY,
        item_id TEXT,
        category1 TEXT,
        category2 TEXT,
        category3 TEXT,
        description TEXT
    )
    """)

    cursor.execute("DELETE FROM fashion_items")

    for idx, sample in enumerate(tqdm(dataset)):

        cursor.execute(
            """
            INSERT INTO fashion_items
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                idx,
                sample["item_ID"],
                sample["category1"],
                sample["category2"],
                sample["category3"],
                sample["text"],
            ),
        )

    conn.commit()
    conn.close()

    print("Database created!")
    print(DB_PATH)


if __name__ == "__main__":
    main()
