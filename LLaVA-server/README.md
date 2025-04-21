# LLaVA-server

Provides LLaVA inference through an HTTP server. Supports batched inference and caches embeddings for each image, thereby enabling more efficient generation of multiple responses per image.

## Weights Used
liuhaotian/llava-v1.5-13b: Available at https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main  
openai/clip-vit-large-patch14-336: Available at https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main  
microsoft/deberta-xlarge-mnli (used for BERT score): Available at https://huggingface.co/microsoft/deberta-xlarge-mnli/tree/main

## Usage
### Install Dependencies
First, install the necessary dependencies by running:
```bash
pip install -e .
```
### Configure GPU Settings
Before launching, open `gunicorn.conf.py` and configure the number of GPUs and the path to llava's weights. 

For convenience, we recommend downloading all files from the [`liuhaotian/llava-v1.5-13b`](https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main) repository on Hugging Face and saving them to the `../llava-weights` folder. This ensures that no further code modifications related to the llava weights path are required.
### Run llava-server
In our experiment, we used 2 GPUs for llava-server and 6 GPUs for ddpo-pytorch.

To start the llava-server, simply execute the following command:
```bash
gunicorn "app:create_app()"
```