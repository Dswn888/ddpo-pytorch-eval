# LLaVA-server

Provides LLaVA inference through an HTTP server. Supports batched inference and caches embeddings for each image, thereby enabling more efficient generation of multiple responses per image.

## Weights Used
liuhaotian/llava-v1.5-13b: Available at https://huggingface.co/liuhaotian/llava-v1.5-13b/tree/main  
openai/clip-vit-large-patch14-336: Available at https://huggingface.co/openai/clip-vit-large-patch14-336/tree/main  
microsoft/deberta-xlarge-mnli (used for BERT score): Available at https://huggingface.co/microsoft/deberta-xlarge-mnli/tree/main

## Usage
Before launching, you must modify `gunicorn.conf.py` to set the number of GPUs and the path to llava's weights. 

We recommend downloading all files from the `liuhaotian/llava-v1.5-13b` repository on the Hugging Face website and storing them in the 
`../llava-weights` folder, so you won't need to make further changes to the code regarding the llava weights path.

In our experiment, we used 2 GPUs for llava-server and 6 GPUs for ddpo-pytorch.

Finally, run the following command in terminal to start the llava-server:
```bash
gunicorn "app:create_app()"
```