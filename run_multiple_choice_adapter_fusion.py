# This code used as a base code the following Hugging Face file
# https://github.com/huggingface/transformers/blob/master/examples/legacy/multiple_choice/run_multiple_choice.py
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional
from re import search
import numpy as np
import torch

from utils import (
    FormatDataset,
    tokenize,
)

from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    AutoModelWithHeads, PreTrainedTokenizer, AdapterType
)
from transformers.adapters.configuration import PfeifferConfig


logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    adapter_1: Optional[str] = field(
        default=None,
        metadata={"help": "path to adapter to fuse"},
    )
    adapter_2: Optional[str] = field(
        default=None,
        metadata={"help": "path to adapter to fuse"},
    )
    adapter_3: Optional[str] = field(
        default=None,
        metadata={"help": "path to adapter to fuse"},
    )
    adapter_4: Optional[str] = field(
        default=None,
        metadata={"help": "path to adapter to fuse"},
    )
    avg_type: Optional[str] = field(
        default="macro",
        metadata={"help": "type of average"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train "})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    sanity_check: bool = field(default=False, metadata={"help": "saved mdoels for sanity check."})
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    # Set seed
    set_seed(training_args.seed)

    # # Data collator
    # data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    # Data pre-processing
    MAPPER = {
        "chemprot": {'ACTIVATOR': 0, 'AGONIST': 1, 'AGONIST-ACTIVATOR': 2, 'AGONIST-INHIBITOR': 3, 'ANTAGONIST': 4,
                     'DOWNREGULATOR': 5, 'INDIRECT-DOWNREGULATOR': 6, 'INDIRECT-UPREGULATOR': 7, 'INHIBITOR': 8,
                     'PRODUCT-OF': 9, 'SUBSTRATE': 10, "SUBSTRATE_PRODUCT-OF": 11, 'UPREGULATOR': 12},
        "citation-intent": {"Background": 0, "CompareOrContrast": 2, "Extends": 4, "Future": 5, "Motivation": 3,
                            "Uses": 1},
        "sciie": {"COMPARE": 0, "CONJUNCTION": 1, "EVALUATE-FOR": 2, "FEATURE-OF": 3, "HYPONYM-OF": 4, "PART-OF": 5,
                  "USED-FOR": 6},
        "rct": {"BACKGROUND": 0, "CONCLUSIONS": 1, "METHODS": 2, "OBJECTIVE": 3, "RESULTS": 4},
        "ag": {1: 0, 2: 1, 3: 2, 4: 3},
        "hyperpartisan_news": {"true": 1, "false": 0},
        "amazon": {"helpful": 1, "unhelpful": 0},
        "imdb": {1: 1, 0: 0},
    }


    def id_mapper(dataset_dir_name, label):
        """ Function to define map label to id """

        for dataset_name in MAPPER:
            if search(dataset_name, dataset_dir_name):
                return MAPPER[dataset_name][label]

    dataset = {'train': {'value': [], 'id': []},
               'dev': {'value': [], 'id': []},
               'test': {'value': [], 'id': []}}

    def get_data(type_data, out_dict):
        """ Function to get the data """""
        with open(data_args.data_dir + f"{type_data}.jsonl") as f:
            for line in f:
                out_dict["value"].append(json.loads(line)["text"])
                id = id_mapper(data_args.data_dir, json.loads(line)["label"])
                out_dict["id"].append(id)

    for dataset_name in MAPPER:
        if search(dataset_name, data_args.data_dir):
            num_labels = len(MAPPER[dataset_name].keys())

    [get_data(k, v) for k, v in dataset.items()]

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    # AutoClasses are here to automatically retrieve the relevant model given
    # the name/path to the pretrained weights/config/vocabulary.
    # This is a generic model class that will be instantiated as one of the model
    # classes of the library (with the option to add multiple flexible heads on
    # top) when created with the from_pretrained() class method or the
    # from_config() class method.
    model = AutoModelWithHeads.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )

    train_dataset = FormatDataset(tokenize(tokenizer, dataset['train']['value']), dataset['train']['id'])
    eval_dataset = FormatDataset(tokenize(tokenizer, dataset['dev']['value']), dataset['dev']['id'])
    test_dataset = FormatDataset(tokenize(tokenizer, dataset['test']['value']), dataset['test']['id'])

    # Fusion training

    # Load the pre-trained adapters we want to fuse
    """
    Implementation guided from:
    https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/03_Adapter_Fusion.ipynb
    """
    # fusion_adapters = ["nli/multinli@ukp", "sts/qqp@ukp", "nli/scitail@ukp"]
    fusion_adapters = []
    if model_args.adapter_1:
        fusion_adapters.append(model_args.adapter_1)
    if model_args.adapter_2:
        fusion_adapters.append(model_args.adapter_2)
    if model_args.adapter_3:
        fusion_adapters.append(model_args.adapter_3)
    if model_args.adapter_4:
        fusion_adapters.append(model_args.adapter_4)

    [model.load_adapter(adapter, config=PfeifferConfig(), with_head=False) for adapter in fusion_adapters]

    adapter_setup = []
    for adapter in model.config.adapters.adapters.keys():
        adapter_setup.append(adapter)
    # Add a fusion layer for all loaded adapters
    model.add_adapter_fusion(adapter_setup, 'dynamic')
    model.set_active_adapters([adapter_setup])

    # Add a classification head for our target task
    model.add_classification_head(data_args.task_name, num_labels=num_labels, overwrite_ok=True)

    # Unfreeze and activate fusion setup
    model.train_adapter_fusion([adapter_setup])

    def compute_metrics(p: EvalPrediction):
        """ Code guided from https://huggingface.co/transformers/v3.0.2/training.html"""
        preds = np.argmax(p.predictions, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            p.label_ids, preds, average=model_args.avg_type, labels=[i for i in range(num_labels)]
        )
        return {"acc": (preds == p.label_ids).mean(), "f1": f1, "precision": precision, "recall": recall}

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        do_save_adapter_fusion=True,
        do_save_full_model=True,
    )

    def save_model(state_dict, checkpoint_path):
        torch.save(state_dict, checkpoint_path)

    # Training
    if training_args.do_train:
        # save model before training
        if data_args.sanity_check:
            save_model(model.state_dict(), training_args.output_dir + "pt_before_model.bin")
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

        # save model after training
        if data_args.sanity_check:
            save_model(model.state_dict(), training_args.output_dir + "pt_after_model.bin")

        # Cache the tokenizer
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate(eval_dataset=eval_dataset)
        output_eval_file = os.path.join(training_args.output_dir, f"{data_args.task_name}_eval_results.txt")

        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
            results.update(result)

    if training_args.do_predict:
        logging.info("*** Test ***")
        test_result = trainer.evaluate(eval_dataset=test_dataset)
        output_test_file = os.path.join(training_args.output_dir, f"{data_args.task_name}_test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results {} *****")
                for key, value in test_result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))
    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
