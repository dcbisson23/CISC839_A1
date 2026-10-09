from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch


def load_model(model_dict:dict) -> AutoModelForCausalLM:
    model_name = model_dict.get("model")
    gguf = model_dict.get("gguf")
    if gguf is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=gguf, device_map="balanced")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="balanced")
    return model


def load_tokenizer(model_dict:dict) -> AutoTokenizer:
    model_name = model_dict.get("model")
    gguf = model_dict.get("gguf")
    if gguf is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=gguf, device_map="balanced")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="balanced")
    return tokenizer


def unload_model(model):
    # It annoys me that this doesn't work.
    # .ipynb files have weird VRAM management apparently? Uncertain.
    del model
    gc.collect()
    torch.cuda.empty_cache()


def get_data_from_response(response: str) -> dict:
    response_data = {}
    res_split = response
    # Get any thinking blocks first.
    res_split = res_split.split("<\\think>")
    if len(res_split) > 1:
        thinking_blocks = res_split
        res_split = [thinking_blocks.pop(-1)]
        for block in thinking_blocks:
            # Thinking blocks come before anything else, so this should work.
            block.removeprefix("<think>")
        response_data["thinking_blocks"] = thinking_blocks
    res_split = res_split[-1].split("```")
    if len(res_split) > 1:
        # We're only going to evaluate on the last SQL segment if multiple exist.
        query = res_split[-2].removeprefix("sql")
    else:
        # If at this point we don't have any sql blocks, we'll assume the response is formatted correctly.
        query = res_split[-1]
    response_data["query"] = query.strip()

    return response_data
