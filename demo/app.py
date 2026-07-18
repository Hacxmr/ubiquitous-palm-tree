from pathlib import Path

import gradio as gr

from retriever.search import FashionRetriever

retriever = FashionRetriever()


def retrieve(query, k):
    results = retriever.search(query, int(k))

    gallery = []

    for r in results:

        caption = f"""
⭐ Score : {r['score']}

👕 {r['category1']} / {r['category2']}

{r['text']}
"""

        gallery.append((r["image"], caption))

    return gallery


demo = gr.Interface(
    fn=retrieve,
    inputs=[
        gr.Textbox(
            label="Search",
            placeholder="blue denim jeans"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=1,
            label="Top K"
        )
    ],
    outputs=gr.Gallery(
        label="Results",
        columns=2,
        height=700,
    ),
    title="Fashion Retrieval System",
    description="OpenCLIP + FAISS + DeepFashion",
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,
    show_error=True,
)
