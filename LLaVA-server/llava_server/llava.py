import argparse
import torch
import os
import json
from tqdm import tqdm
from typing import Iterable, List
from typing import Iterable, List
import numpy as np

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from PIL import Image
import math

MAX_TOKENS = 64
PROMPT = """You are LLaVA, a large language and vision assistant.You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.Follow the instructions carefully and explain your answers in detail.###Human: Hi!###Assistant: Hi there!  How can I help you today?###Human:"""


def pad_sequences(sequences, pad_token_id, padding_side="left"):
    """
    Pad a list of token id sequences of unequal lengths.
    Supports left or right padding.
    """
    max_length = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if padding_side == "left":
            padded_seq = [pad_token_id] * (max_length - len(seq)) + seq
        else:
            padded_seq = seq + [pad_token_id] * (max_length - len(seq))
        padded.append(padded_seq)
    return padded


def tokenizer_image_token_batch(prompts, tokenizer, image_token_index=IMAGE_TOKEN_INDEX):
    """
    Process a list of prompts using tokenizer_image_token and return a tensor after applying left-side padding.
    """
    # For each prompt, use tokenizer_image_token to get the token id list (without returning a tensor)
    batch_encodings = [
        tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None)
        for prompt in prompts
    ]
    
    # Use pad_sequences to perform left-side padding on the token id lists to construct a tensor of uniform length.
    padded_encodings = pad_sequences(batch_encodings, pad_token_id=tokenizer.pad_token, padding_side=tokenizer.padding_side)
    
    return torch.tensor(padded_encodings, dtype=torch.long)


def load_llava(params_path):
    # Model
    disable_torch_init()
    model_name = get_model_name_from_path(params_path)
    model_base = None
    tokenizer, model, image_processor, context_len = load_pretrained_model(params_path, model_base, model_name)
    model.half()  # align with the format in ddpo-pytorch, using fp16
    
    if getattr(model.config, 'mm_use_im_start_end', False):
        image_tokens = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    else:
        image_tokens = DEFAULT_IMAGE_TOKEN
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    @torch.inference_mode()
    def inference_fn(
        images: Iterable[Image.Image], queries: Iterable[Iterable[str]]
    ) -> List[List[str]]:
        assert len(images) == len(queries)
        assert np.all(len(queries[0]) == len(q) for q in queries)

        queries = np.array(queries)  # (batch_size, num_queries_per_image)

        # preprocess images
        image_tensor = image_processor(images, return_tensors="pt")["pixel_values"]
        images = image_tensor.half().cuda()

        # first, get the activations for the image tokens
        initial_prompts = [PROMPT + image_tokens + " " for _ in range(len(images))]
        
        # conv = conv_templates["vicuna_v1"].copy()
        # conv.append_message(conv.roles[0], initial_prompts[0])
        # conv.append_message(conv.roles[1], None)
        # for i in range(len(initial_prompts)):
        #     initial_prompts[i] = conv.get_prompt()
        
        initial_input_ids = tokenizer_image_token_batch(initial_prompts, tokenizer, IMAGE_TOKEN_INDEX).cuda()
        initial_out = model(initial_input_ids, images=images, use_cache=True)
        initial_key_values = initial_out.past_key_values

        # broadcast the key values across the queries
        # becomes shape (batch_size * num_queries_per_image, ...)
        initial_key_values = [
            [
                x.unsqueeze(1)
                .expand(-1, queries.shape[1], -1, -1, -1)
                .reshape(-1, *x.shape[1:])
                for x in y
            ]
            for y in initial_key_values
        ]

        # flatten queries into one big batch
        flat_queries = queries.reshape(-1)  # (batch_size * num_queries_per_image)

        # prepare inputs for the queries
        prompts = [q + "###" for q in flat_queries]
        input_ids = tokenizer_image_token_batch(prompts, tokenizer, IMAGE_TOKEN_INDEX).cuda()

        # stop upon seeing any of these tokens
        stop_tokens = torch.as_tensor(
            tokenizer.convert_tokens_to_ids(["▁###", "###", "##", "#"]),
            dtype=torch.long,
            device="cuda",
        )

        # generation loop
        output_ids = []
        key_values = initial_key_values
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device="cuda")
        for i in range(MAX_TOKENS):
            out = model(input_ids=input_ids, use_cache=True, past_key_values=key_values)
            key_values = out.past_key_values
            next_tokens = torch.argmax(out.logits[:, -1], dim=-1)

            finished = finished | (next_tokens.unsqueeze(-1) == stop_tokens).any(dim=-1)

            if finished.all():
                break
            output_ids.append(next_tokens)
            input_ids = next_tokens.unsqueeze(-1)

        output_ids = torch.stack(output_ids, dim=-1)
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        # clean outputs
        outputs_clean = []
        for output in outputs:
            for pattern in ["###", "##", "#"]:
                if pattern in output:
                    output = output.split(pattern)[0]

            if "Assistant:" in output:
                output = output.split("Assistant:")[1]
            output = output.split('.')[0] + '.'  # remove some extra interfering sentences from the LLaVA model output
            outputs_clean.append(output.strip())

        # reshape outputs back to (batch_size, num_queries_per_image)
        outputs_clean = np.array(outputs_clean).reshape(queries.shape)

        return outputs_clean.tolist()

    return inference_fn
