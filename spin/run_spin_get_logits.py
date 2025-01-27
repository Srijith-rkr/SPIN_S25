#!/usr/bin/env python
# 
# Adapted from https://github.com/huggingface/alignment-handbook 

import os
os.environ["HF_HOME"] = "/data/tir/projects/tir7/user_data/srijithr/hf_cache_dir"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # for testing
 
import re
import sys
import yaml 
import json
import logging
import argparse
from tqdm import tqdm

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.utils import pad_to_length # for concatenated forward

# from alignment import (
#     DataArguments,
#     SPINConfig,
#     H4ArgumentParser,
#     ModelArguments,
#     get_datasets,
#     get_kbit_device_map,
#     get_peft_config,
#     get_quantization_config,
#     get_tokenizer,
#     is_adapter_model,
#     SPINTrainer
# )
from datasets import load_dataset



# for lops 
from typing import Dict, List, Tuple, Union

def create_unique_dir_name(base_dir):
    # If base directory does not exist, return it
    if not os.path.exists(base_dir):
        return base_dir
    else:
        # Find the next available directory name with a suffix
        counter = 2
        new_dir = f"{base_dir}_{counter}"
        while os.path.exists(new_dir):
            counter += 1
            new_dir = f"{base_dir}-{counter}"
        return new_dir
    
def apply_chat_template(
    example, tokenizer, task, assistant_prefix="<|assistant|>\n"
):
    def _strip_prefix(s, pattern):
        # Use re.escape to escape any special characters in the pattern
        return re.sub(f"^{re.escape(pattern)}", "", s)

    if all(k in example.keys() for k in ("real", "generated")):
        # Compared to reward modeling, we filter out the prompt, so the text is everything after the last assistant token
        prompt_messages = [[msg for msg in example["real"] if msg["role"] == "user"][0]]
        # Insert system message
        if example["real"][0]["role"] != "system":
            prompt_messages.insert(0, {"role": "system", "content": ""})
        else:
            prompt_messages.insert(0, example["real"][0])

        real_messages = example["real"][1:]
        generated_messages = example["generated"][1:]
        example["text_real"] = tokenizer.apply_chat_template(real_messages, tokenize=False)
        example["text_generated"] = tokenizer.apply_chat_template(generated_messages, tokenize=False)
        example["text_prompt"] = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )
        example["text_real"] = _strip_prefix(example["text_real"], assistant_prefix)
        example["text_generated"] = _strip_prefix(example["text_generated"], assistant_prefix)

        example['real_list'] = example['real']
        example['generated_list'] = example['generated']
    else:
        raise ValueError(
            f"Require `[real, generated]` keys but found {list(example.keys())}"
            )
    return example


########################
    # Functions for logp values 
########################

LABEL_PAD_TOKEN_ID = -100
PADDING_VALUE = 0
def _get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != LABEL_PAD_TOKEN_ID

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == LABEL_PAD_TOKEN_ID] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the real and generated inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'åreal_input_ids' and 'generated_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    concatenated_batch = {}


    max_length = max(batch["real_input_ids"].shape[1], batch["generated_input_ids"].shape[1]) # this one is used

    for k in batch:
        if k.startswith("real") and isinstance(batch[k], torch.Tensor):
            pad_value = LABEL_PAD_TOKEN_ID if "labels" in k or False else PADDING_VALUE# label_pad_token_id -100, padding value = 0
            concatenated_key = k.replace("real", "concatenated")
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value) # from trl.trainer.utils import disable_dropout_in_model, pad_to_length
    for k in batch:
        if k.startswith("generated") and isinstance(batch[k], torch.Tensor):
            pad_value = LABEL_PAD_TOKEN_ID if "labels" in k or False else PADDING_VALUE
            concatenated_key = k.replace("generated", "concatenated")
            concatenated_batch[concatenated_key] = torch.cat(
                (
                    concatenated_batch[concatenated_key],
                    pad_to_length(batch[k], max_length, pad_value=pad_value),
                ),
                dim=0,
            ).to('cuda')

    return concatenated_batch

def concatenated_forward(
        model, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the real and generated inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        len_real = batch["real_labels"].shape[0]

        model_kwargs = {}
    
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            **model_kwargs,
        ).logits.to(torch.float32)

        all_logps = _get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
        )

        real_logps = all_logps[:len_real]
        generated_logps = all_logps[len_real:]

        real_logits = all_logits[:len_real]
        generated_logits = all_logits[len_real:]

        return (real_logps, generated_logps, real_logits, generated_logits)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_frac', type=int, default=0)
    parser.add_argument('--frac_len', type=float, default=0.5)
    return parser.parse_args()


def main():
    MODEL = '/data/tir/projects/tir7/user_data/srijithr/spin_filtering_outputs/iter1_HW_LL_25_percentile-3'
    # REVISION = 'ac6e600eefcce74f5e8bae1035d4f66019e93190' ##### <<<<<<<----------------- CHANGE THIS
    REVISION = 'main'
    OUTPUT_FILE = '/home/srijithr/iterative-alignment/SPIN_implementation/scored_data/genrated_using_iter1_25percentile/synthetic_train_LOGP.json'
    DATA_FILE = '/home/srijithr/iterative-alignment/SPIN_implementation/scored_data/genrated_using_iter1_25percentile/synthetic_train.json'

    args = parse_arguments()
    data_frac = args.data_frac
    frac_len = args.frac_len

    OUTPUT_FILE = OUTPUT_FILE.replace(".json", f"_subset_{data_frac}.json")
    print(OUTPUT_FILE)

    raw_datasets = load_dataset("json", data_files=DATA_FILE)
    # raw_datasets['train'] = raw_datasets['train'].select(range(30)) # for testing

    if frac_len is not None:
        if frac_len < 1:
            frac_len = int(len(raw_datasets['train'])*frac_len)
            
        if frac_len*(data_frac+1) > len(raw_datasets['train']): 
            pass
            raw_datasets['train'] = raw_datasets['train'].select(range(frac_len*data_frac, len(raw_datasets['train']) ))
        else:
            raw_datasets['train'] = raw_datasets['train'].select(range(frac_len*data_frac, frac_len*(data_frac+1) ))

        
    print(f"Training on local dataset with {len(raw_datasets['train'])} samples")

    #####################################
    # Load tokenizer and process datasets
    #####################################
    tokenizer = AutoTokenizer.from_pretrained(
        'alignment-handbook/zephyr-7b-sft-full',
        revision=REVISION,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.truncation_side = 'left'

    from alignment.data import DEFAULT_CHAT_TEMPLATE
    tokenizer.chat_template = DEFAULT_CHAT_TEMPLATE

    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer, "task": "spin"},
        num_proc=5, 
        # remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )


    raw_datasets['train'] = raw_datasets['train'].rename_columns(
        {"text_prompt": "prompt", "text_real": "real", "text_generated": "generated"})


    model_kwargs = dict(
        revision=REVISION,
        device_map='auto',
        torch_dtype= torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL, **model_kwargs)
    sys.path.append('/home/srijithr/iterative-alignment/SPIN_implementation/SPIN/spin')
    from alignment.utils import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
    tokenizer,
    max_length= 1024, #
    max_prompt_length=512, #
    label_pad_token_id=-100, # 
    padding_value=0, # 0
    truncation_mode='keep_end', # 
    is_encoder_decoder=False, # False
    max_target_length=None,
    flag_for_logp = True) 

    dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            # "num_workers": self.args.dataloader_num_workers,
            'shuffle' :False, # False
            "pin_memory":True, # True
            "persistent_workers": False,  # False
            }
    from torch.utils.data import DataLoader
    epoch_iterator  = DataLoader(raw_datasets["train"], **dataloader_params)


    to_save = []
    with torch.no_grad():
        for step, inputs in enumerate(tqdm(epoch_iterator)):
            features , input =  inputs
            datapoint = features[0]
            real_logp, generated_logp, _, _ = concatenated_forward(model, input)
            datapoint['real_logp'] = real_logp.item()
            datapoint['generated_logp'] = generated_logp.item()
            to_save.append(datapoint)       

            if step % 1000 == 0:
                with open(OUTPUT_FILE, "w") as f:
                    json.dump(to_save, f,indent = 4) 
    # if model_args.use_ref_model:

    with open(OUTPUT_FILE, "w") as f:
        json.dump(to_save, f,indent = 4)

    # also saving as jsonl for safety
    with open(OUTPUT_FILE.replace(".json", ".jsonl"), "w") as f:
        for example in to_save:
            f.write(json.dumps(example))
            f.write("\n")

if __name__ == "__main__":
    main()
