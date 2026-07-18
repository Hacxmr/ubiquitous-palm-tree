import torch
import open_clip
from PIL import Image


class OpenCLIPEmbedder:
    def __init__(
        self,
        model_name="ViT-H-14",
        pretrained="laion2b_s32b_b79k",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image: Image.Image):
        image = self.preprocess(image).unsqueeze(0).to(self.device)

        features = self.model.encode_image(image)
        features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str):
        tokens = self.tokenizer([text]).to(self.device)

        features = self.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

        return features.squeeze().cpu().numpy()
