import json

from models.qwen_vl import QwenVL


class QwenReranker:

    def __init__(self):
        print("Loading Qwen Reranker...")
        self.model = QwenVL()

    def rerank(self, query, results):

        reranked = []

        for item in results:

            prompt = f"""
You are a fashion retrieval expert.

User Query:
{query}

Carefully inspect the image.

Return ONLY valid JSON.

{{
    "score": <integer from 0 to 100>,
    "reason": "<one sentence>"
}}
"""

            output = self.model.analyze(
                item["image"],
                prompt,
            )

            if isinstance(output, dict):

                score = output.get("score", 0)
                reason = output.get("reason", "")

            else:

                try:
                    parsed = json.loads(output)

                    score = parsed.get("score", 0)
                    reason = parsed.get("reason", "")

                except Exception:

                    score = 0
                    reason = "Could not parse Qwen output."

            item["llm_score"] = score
            item["reason"] = reason

            reranked.append(item)

        reranked.sort(
            key=lambda x: x["llm_score"],
            reverse=True,
        )

        return reranked
