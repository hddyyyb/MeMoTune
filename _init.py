import argparse
import json
import os
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from peft import LoftQConfig, LoraConfig, TaskType
from src.mapping import get_peft_model
from safetensors import safe_open


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)       
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)
        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)



def print_model_f(model, name):
    with open("./output/peft_tuners_lora_layer.txt", "at") as external_file:        
        print("=" * 10 + name + "=" * 10, file=external_file)
        print(model, file=external_file)
        for name, param in model.named_parameters():
            if torch.is_tensor(param):
                if param.dtype in [torch.float32, torch.float16]:
                    print(
                        name,
                        param.shape,
                        param.device,
                        param.dtype,
                        param.requires_grad,
                        param.mean().item(),
                        param.max().item(),
                        file=external_file
                    )
                else:
                    print(name, param.shape, param.device, param.dtype, param.requires_grad, file=external_file)

        external_file.close()


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub, token your HF token if the model is private, e.g., llama-2.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="The rank of the LoRA adapter",
    )
    args = parser.parse_args()
    return args
 

def quantize_and_save():
    args = arg_parse()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            token=args.token,
            trust_remote_code=True,
            device_map="auto",
        )
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["bart"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]
    
    elif any(name in args.model_name_or_path.lower() for name in ["t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules =["SelfAttention.q", "SelfAttention.k", "SelfAttention.v",  "SelfAttention.o","DenseReluDense.wi"]
    elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_CLS
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  
    else:
        raise NotImplementedError("Other models not supported yet.")

    loftq_config = LoftQConfig(loftq_bits=args.bits, loftq_iter=args.iter)

    lora_config = LoraConfig(
        task_type=task_type,
        inference_mode=True,
        r=args.rank,
        lora_alpha=16 if task_type is TaskType.CAUSAL_LM and args.bits == 4 else args.rank,
        lora_dropout=0.1,
        target_modules=target_modules,
        init_lora_weights="loftq",
        loftq_config=loftq_config,
    )

    lora_model = get_peft_model(model, lora_config) 
    base_model = lora_model.get_base_model()
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit" + f"-{args.rank}rank"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "loftq_init")

    lora_model.save_pretrained(lora_model_dir)

    print_model_f(lora_model, "lora_model")
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    print_model_f(base_model, "base_model")

    tensors = {}
    with safe_open(os.path.join(lora_model_dir, "adapter_model.safetensors"), framework="pt") as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    torch.save(tensors, os.path.join(lora_model_dir, "adapter_model.bin"))

    with open(os.path.join(lora_model_dir, "adapter_config.json"), "r") as fp:
        adapter_config = json.load(fp)
        adapter_config['base_model_name_or_path'] = base_model_dir 
        adapter_config['init_lora_weights'] = True  
        fp.close()
    with open(os.path.join(lora_model_dir, "adapter_config.json"), "w") as fp:
        json.dump(adapter_config, fp, indent=2)

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_dir, lora_dir = quantize_and_save()

