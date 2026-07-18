from datasets import load_from_disk
from models.openclip import OpenCLIPEmbedder
from tqdm import tqdm
import numpy as np
import os

dataset = load_from_disk("dataset/deepfashion_multimodal")
embedder = OpenCLIPEmbedder()

embeddings = []
item_ids = []

for sample in tqdm(dataset):
    emb = embedder.encode_image(sample["image"])
    embeddings.append(emb.astype(np.float32))
    item_ids.append(sample["item_ID"])

embeddings = np.stack(embeddings)

os.makedirs("outputs/embeddings", exist_ok=True)

np.save("outputs/embeddings/image_embeddings.npy", embeddings)

np.save(
    "outputs/embeddings/item_ids.npy",
    np.array(item_ids)
)

print("Saved embeddings:", embeddings.shape)
