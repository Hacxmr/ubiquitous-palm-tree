from models.qwen_vl import QwenVL
from configs.prompts import FASHION_PROMPT

model = QwenVL()

result = model.analyze(
    "dataset/images/test.jpg",
    FASHION_PROMPT
)

print(result)