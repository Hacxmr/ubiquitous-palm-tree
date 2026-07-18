from datasets import load_dataset
import os

os.makedirs("dataset", exist_ok=True)

ds = load_dataset("Marqo/deepfashion-multimodal", split="data")

print(ds)
print(f"Number of samples: {len(ds)}")

# Save locally for faster reuse
ds.save_to_disk("dataset/deepfashion_multimodal")
