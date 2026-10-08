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
    del model
    gc.collect()
    torch.cuda.empty_cache()
