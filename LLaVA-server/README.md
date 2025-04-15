# LLaVA-server

Provides LLaVA inference through an HTTP server. Supports batched inference and caches embeddings for each image, thereby enabling more efficient generation of multiple responses per image.

## Weights Used
liuhaotian/llava-v1.5-13b: Available at https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main
openai/clip-vit-large-patch14-336: Available at https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main
microsoft/deberta-xlarge-mnli (used for BERT score): Available at https://huggingface.co/microsoft/deberta-xlarge-mnli/tree/main

## Usage
Before launching, you must modify `gunicorn.conf.py` to set the number of GPUs. In our experiment, we used 2 GPUs for llava-server and 6 GPUs for ddpo-pytorch.
```bash
gunicorn "app:create_app()"
```

