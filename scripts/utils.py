from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import torch


def load_model(model_dict:dict) -> AutoModelForCausalLM:
    model_name = model_dict.get("model")
    gguf = model_dict.get("gguf")
    if gguf is not None:
        model = AutoModelForCausalLM.from_pretrained(model_name, gguf_file=gguf, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model


def load_tokenizer(model_dict:dict) -> AutoTokenizer:
    model_name = model_dict.get("model")
    gguf = model_dict.get("gguf")
    if gguf is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_name, gguf_file=gguf, device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    return tokenizer


def unload_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
