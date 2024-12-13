from accelerate.utils import set_seed
from dataclasses import dataclass, field
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from src.loralayer import LoraLayer
from src.mapping import get_peft_model
from src.peft_model import PeftModel
from src.trainer import Trainer
from src.trainer_seq2seq import Seq2SeqTrainer
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
from typing import Dict, Optional, Sequence
import argparse
import copy
import logging
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import re
import torch
import transformers

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
ANSWER_PROMPT = "The final answer is: "
QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="LoftQ/Mistral-7B-v0.1-4bit-64rank",
        metadata={"help": "Path to the model."},
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the LoRA adapter. Used in evaluation or resuming from the checkpoint."},
    )
    lora_init: bool = field(
        default=False,
        metadata={"help": "True: Use zero and gaussian initialization; False: Load adapters from LoftQ in HF hub."},
    )
    full_precision:  bool = field(
        default=False,
        metadata={"help": "False: Use bitsandbytes Linear4bit, real quantization"
                          "True: Use quantization equivalent fp16/fp32 weights."
                  },
    )
    rank: int = field(
        default=64,
        metadata={"help": "Rank of LoRA adapters. LoftQ does not require this config. Used for fp16 LoRA or QLoRA."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "LoftQ does not require this config. Used for QLoRA."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={"help": "HF token to access to private models, e.g., meta-llama"},
    )


@dataclass
class DataArguments:
    data_name: str = field(
        default="gsm8k",
        metadata={"help": "Dataset name."}
    )
    batch_size: int = field(default=16, metadata={"help": "Evaluation batch size."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. "},
    )
    expt_name: str = field(
        default="default",
        metadata={"help": "Experiment name"},
    )
    output_dir: Optional[str] = field(default="./GSM8K_ACC")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources: Sequence[str], targets: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        logging.warning("Formatting inputs...")
        sources = [f"{example['question']}{QUESTION_PROMPT}" for example in raw_data]
        targets = [f"{example['answer']}{tokenizer.eos_token}".replace("####", ANSWER_PROMPT) for example in raw_data]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    logging.warning("Downloading Data")
    dataset = load_dataset(data_args.data_name, "main")
    train_set = dataset['train']


    test_set = dataset['test']
    question = [f"{example['question']}{QUESTION_PROMPT}" for example in test_set]
    answer = []


    train_dataset = SupervisedDataset(raw_data=train_set, tokenizer=tokenizer)
    test_dataset = SupervisedDataset(raw_data=test_set, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator), test_set, question, answer
    

def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = float(pred[-1])
    else:
        pred_answer = float(pred[-1])

    if isinstance(pred_answer, str):
        try:
            pred_answer = float(pred_answer)
        except ValueError as e:
            pred_answer = float('inf')
    return pred_answer


def compute_accuracy(pred: list, gold: list):
    acc = 0.0
    for p, g in zip(pred, gold):
        if p == g:
            acc += 1

    return acc / len(pred)



class PerplexityCallback(TrainerCallback):
    def __init__(self):
        self.train_perplexities = []
        self.val_perplexities = []
        self.epochs_train = []
        self.epochs_val = []

    def on_log(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_local_process_zero:
            if 'eval_loss' in state.log_history[-1]:
                val_loss = state.log_history[-1]['eval_loss']
                self.val_perplexities.append(math.exp(val_loss))
                self.epochs_val.append(state.epoch)
            if 'loss' in state.log_history[-1]:
                train_loss = state.log_history[-1]['loss']
                self.train_perplexities.append(math.exp(train_loss))
                self.epochs_train.append(state.epoch)

    def plot_perplexities(self, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(self.epochs_train, self.train_perplexities, label='Training Perplexity')
        plt.plot(self.epochs_val, self.val_perplexities, label='Validation Perplexity')
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.legend()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def save_perplexities_csv(self, csv_path):
        df_train = pd.DataFrame({
            'Epoch': self.epochs_train,
            'Training Perplexity': self.train_perplexities
        })
        df_val = pd.DataFrame({
            'Epoch': self.epochs_val,
            'Validation Perplexity': self.val_perplexities
        })
        df = pd.merge(df_train, df_val, on='Epoch', how='outer').sort_values(by='Epoch').reset_index(drop=True)
        df.to_csv(csv_path, index=False)



def train():
    global args
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    if model_args.full_precision:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            token=model_args.token,
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type='nf4',
            ),
        )
    
        tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        token=model_args.token,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if model_args.lora_init:
        task_type = TaskType.CAUSAL_LM
        if any(name in model_args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        elif any(name in model_args.model_name_or_path.lower() for name in ["phi"]):
            target_modules = ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"]
        else:
            raise ValueError(f"Only support LLAMA, Mistral, Falcon, Phi-2, but got {model_args.model_name_or_path}.")
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=False,
            r=model_args.rank,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights=True,
        )
        model = get_peft_model(model, lora_config)
    elif model_args.adapter_name_or_path is not None:
        model = PeftModel.from_pretrained(
            model,
            model_args.adapter_name_or_path,
            is_trainable=True,
            token=model_args.token,
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            model_args.model_name_or_path,
            subfolder='loftq_init',
            is_trainable=True,
            token=model_args.token,
        )

    data_module, test_set, question, answer = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    training_args.output_dir = os.path.join(
        training_args.output_dir,
        training_args.expt_name,
        model_args.model_name_or_path.split('_')[-1],
        f"ep_{int(training_args.num_train_epochs)}",
        f"lr_{training_args.learning_rate}",
        f"seed_{training_args.seed}",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[PerplexityCallback()], 
        **data_module
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    scale = str(training_args.output_dir).replace('/','_')
    perplexity_callback = next(cb for cb in trainer.callback_handler.callbacks if isinstance(cb, PerplexityCallback))
    perplexity_callback.plot_perplexities(save_path=os.path.join(training_args.output_dir,'./perplexity_'+str(args.logging_steps)+'_gsm8k_lr_'+str(args.learning_rate)+'_dirid_'+scale+'_lmml.png'))
    perplexity_callback.save_perplexities_csv(os.path.join(training_args.output_dir,'perplexity_'+str(args.logging_steps)+'_gsm8k_lr_'+str(args.learning_rate)+'_dirid_'+scale+'_lmml.csv'))

    for example in test_set['answer']:
        ans = example.split('####')[-1]
        ans = ans.replace(',', '')  
        try:
            ans = float(ans)
        except ValueError:
            ans = float("inf")
        answer.append(ans)

    logging.warning("Tokenizing inputs...")
    eval_step = math.ceil(len(question)/data_args.batch_size)
    logging.warning(f"Total example: {len(question)} | eval batch size: {data_args.batch_size}"
                    f"eval steps: {eval_step}")
    question_data = []
    for i in range(eval_step):
        if i < eval_step - 1:
            batch = tokenizer(
                question[i*data_args.batch_size: (i+1)*data_args.batch_size],
                return_tensors="pt",
                padding="longest",
            )
        else:
            batch = tokenizer(
                question[i*data_args.batch_size:],
                return_tensors="pt",
                padding="longest",
            )
        batch['input_len'] = len(batch['input_ids'][0])
        question_data.append(batch)

    model.eval()
    ans_pred_list = []


    gen_kwargs = {
        "max_new_tokens": 256,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.95,
        "do_sample": True,
    }
    set_seed(args.seed)
    for step, batch in enumerate(question_data):
        with torch.no_grad():
            gen_kwargs["input_ids"] = batch["input_ids"].to('cuda')
            gen_kwargs["attention_mask"] = batch["attention_mask"].to('cuda')
            generated_tokens = model.generate(**gen_kwargs)

        pred_tokens = generated_tokens[:, batch['input_len']:]
        decoded_pred = tokenizer.batch_decode(pred_tokens, skip_special_tokens=True)
        print(decoded_pred)
        ans_pred_list += [extract_answer_number(sentence_pred) for sentence_pred in decoded_pred]

    print("prediction", ans_pred_list)
    print("ground truth", answer)

    accuracy = compute_accuracy(answer, ans_pred_list)

    print(f"adapter: {model_args.adapter_name_or_path} | GSM8K test accuracy: {100*accuracy:.2f}% | "
          f"full precision: {model_args.full_precision}")



if __name__ == "__main__":
    train()
