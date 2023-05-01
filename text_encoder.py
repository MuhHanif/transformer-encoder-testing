from transformers import (
    CLIPTokenizer,  # module for converting string into numerical token representation
    FlaxCLIPTextModel,  # clip (text encoder part) model
    FlaxT5EncoderModel,  # seed state for numpy, pytorch, & tensorflow
    T5Tokenizer,
)

import jax.numpy as jnp
import jax
from jax import jit
import numpy as np

import matplotlib.pyplot as plt

####[Global var]####
model_dir = "duongna/stable-diffusion-v1-4-flax"
weight_dtype = jnp.float32


# instantiate tokenizer to convert string into numerical token representation
tokenizer = CLIPTokenizer.from_pretrained(model_dir, subfolder="tokenizer")
t5_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

t5_encoder = FlaxT5EncoderModel.from_pretrained(
    "google/flan-t5-base",
    dtype=weight_dtype,
)

# instantiate model from hugging face and get everything in it
text_encoder = FlaxCLIPTextModel.from_pretrained(
    model_dir,
    subfolder="text_encoder",
    dtype=weight_dtype,
    # cpu=True
    # ignore_mismatched_sizes=True
)


text_prompt = (
    "what a very long text that need to be tokenized and converted to embedding vector."
)
# test tokenizer with hardcoded input
inputs = tokenizer(
    text_prompt,
    max_length=77,
    padding="max_length",
    truncation=True,
    return_tensors="np",
)

t5_inputs = t5_tokenizer(
    text_prompt,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="np",
)


token = jnp.array(inputs.input_ids).reshape(1, -1)
mask = jnp.array(inputs.attention_mask).reshape(1, -1)

t5_token = jnp.array(t5_inputs.input_ids).reshape(1, -1)
t5_mask = jnp.array(t5_inputs.attention_mask).reshape(1, -1)

clip = text_encoder(
    input_ids=token, attention_mask=mask, params=text_encoder.params, train=False
)

t5 = t5_encoder(
    input_ids=t5_token, attention_mask=t5_mask, params=t5_encoder.params, train=False
)


print(x)
