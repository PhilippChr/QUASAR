import copy
import json
import torch
import numpy as np
from loguru import logger
from torch.utils.data.dataloader import default_collate

import quasar.evaluation as evaluation

INSTRUCTION = "Please answer the following question using the given evidence."
PROMPT = "SR: {sr}\n\nEvidence:\n{evidences}"
IGNORE_INDEX = -100


def _represent_evidences(evidences, num_evidences=None, add_references=False):
    evidence_texts = list()
    for e in evidences:
        if not e["evidence_text"] in evidence_texts:
            evidence_texts.append(e["evidence_text"])
    if not num_evidences is None:
        evidence_texts = evidence_texts[:num_evidences]
    if add_references:
        evidence_texts = [f"[{i+1}: {ev_text}]" for i, ev_text in enumerate(evidence_texts)]
    return "\n".join(evidence_texts)


def _represent_sr(sr):
    return sr


def turn_to_input(turn, tokenizer, max_evidence_tokens, prompt=PROMPT, num_evidences=None, add_references=False):
    sr = _represent_sr(turn["structured_representation"])
    evidences_text = _represent_evidences(turn["top_evidences"], num_evidences, add_references)
    question = turn["question"]    
    evidences_text = trim_text_to_max_tokens(tokenizer, evidences_text, max_num_tokens=max_evidence_tokens)
    input_text = prompt.format(sr=sr, evidences=evidences_text, question=question)
    dialog = [
    {
        "role": "user",
        "content": input_text
    }]
    formatted_prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
    return formatted_prompt


def _get_relevant_evidence_ids(evidences):
    """Make sure the order and sequence of evidences is same as in the input (make sure texts are unique!)."""
    evidence_texts = set()
    unique_evidences = list()
    for e in evidences:
        if not e["evidence_text"] in evidence_texts:
            evidence_texts.add(e["evidence_text"])
            unique_evidences.append(e)
    ans_evidence_ids = [str(i+1) for i, ev in enumerate(unique_evidences) if ev["is_answering_evidence"]]
    ans_evidence_ids_text = ", ".join(ans_evidence_ids)
    return ans_evidence_ids_text


def turn_to_target(turn, tokenizer, max_evidence_tokens, prompt=PROMPT, refuse_to_answer=False, num_evidences=None, add_references=False):
    sr = _represent_sr(turn["structured_representation"])
    evidences_text = _represent_evidences(turn["top_evidences"], num_evidences, add_references)
    evidences_text = trim_text_to_max_tokens(tokenizer, evidences_text, max_num_tokens=max_evidence_tokens)
    question = turn["question"]
    input_text = prompt.format(sr=sr, evidences=evidences_text, question=question)
    target = ", ".join(answer["label"] for answer in turn["answers"])
    if refuse_to_answer:
        target = "UNKNOWN"
    elif add_references:
        ans_evidence_ids_text = _get_relevant_evidence_ids(turn["top_evidences"])
        target += f" [[{ans_evidence_ids_text}]]"
    dialog = [
    {
        "role": "user",
        "content": input_text
    },{
        "role": "assistant",
        "content": target
    }]
    formatted_prompt = tokenizer.apply_chat_template(dialog, tokenize=False, add_generation_prompt=True)
    return formatted_prompt


def trim_text_to_max_tokens(tokenizer, text, max_num_tokens):
    """Trims the given text to the given maximum number of tokens for the tokenizer."""
    tokenized_prediction = tokenizer.encode(text)
    if len(tokenized_prediction) > max_num_tokens:
        logger.debug(f"Trimming input with {len(tokenized_prediction)} here.")
    trimmed_tokenized_prediction = tokenized_prediction[1: max_num_tokens + 1]
    trimmed_prediction = tokenizer.decode(trimmed_tokenized_prediction)
    return trimmed_prediction


def collate_fn(batch):
    """Collate the input data for the batch."""

    def _is_vector(obj):
        return (type(obj) is torch.Tensor) or (
            type(obj).__module__ == np.__name__ and obj.dtype == np.float32
        )

    elem = batch[0]
    # collate instances and mappings (id->entity/evidence) separately
    instances = {
        key: default_collate([d[key] for d in batch]) for key in elem if _is_vector(elem[key])
    }
    mappings = {key: [d[key] for d in batch] for key in elem if not _is_vector(elem[key])}
    instances.update(mappings)
    return instances


class DatasetLLaMAAnswering(torch.utils.data.Dataset):
    def __init__(self, config, tokenizer, path, train=True):
        self.config = config
        self.tokenizer = tokenizer
        if self.config.get("answering_prompt"):
            self.prompt = self.config["answering_prompt"]
        else:
            self.prompt = PROMPT
        
        if train:
            input_encodings, output_encodings, dataset_length = self._load_data(path, train=train)
        else:
            input_encodings, dataset_length = self._load_data(path, train=train)
            output_encodings = None

        self.input_encodings = input_encodings
        self.output_encodings = output_encodings
        self.dataset_length = dataset_length

    def __getitem__(self, idx):
        if self.output_encodings is None:
            return {
                "input_ids": self.input_encodings["input_ids"][idx],
                "attention_mask": self.input_encodings["attention_mask"][idx]
            }
        labels = self.output_encodings[idx]
        item = {
            "input_ids": self.input_encodings["input_ids"][idx],
            "attention_mask": self.input_encodings["attention_mask"][idx],
            "labels": labels,
        }
        return item

    def __len__(self):
        return self.dataset_length

    def _load_data(self, path, train=True):
        """
        Opens the file, and loads the data into
        a format that can be put into the model.

        The input dataset should be annotated using
        the silver_annotation.py class.

        The whole history is given as input.
        """
        inputs = list()
        targets = list()

        # open data
        add_references = self.config.get("ha_add_references", False)
        with open(path, "r") as fp:
            line = fp.readline()
            while line:
                conv = json.loads(line)
                for turn in conv["questions"]:
                    evidences_in_input = turn["top_evidences"]

                    # skip examples for which the specific answer is not in the evidences
                    refuse_to_answer = False
                    if not evaluation.answer_presence(evidences_in_input, turn["answers"])[0]:
                        if self.config.get("ha_refuse_to_answer", False):
                            refuse_to_answer = True
                        elif self.config.get("ha_faithful", False):
                            continue

                    # prepare input
                    num_evidences = self.config.get("ha_max_num_evidences", None)
                    input_ = turn_to_input(turn, self.tokenizer, self.config["ha_max_evidence_tokens"], prompt=self.prompt, num_evidences=num_evidences, add_references=add_references)
                    inputs.append(input_)
                    target_ = turn_to_target(turn, self.tokenizer, self.config["ha_max_evidence_tokens"], prompt=self.prompt, refuse_to_answer=refuse_to_answer, num_evidences=num_evidences, add_references=add_references)
                    targets.append(target_)
                line = fp.readline()

        target_encodings = self.tokenizer(targets, max_length=self.config["ha_max_input_length"], truncation=True, padding="max_length", return_tensors="pt")
        labels = copy.deepcopy(target_encodings["input_ids"])
        
        dataset_length = len(inputs)
        if train:
            for label, input_ in zip(labels, inputs):
                prompt_tokens = self.tokenizer(input_, max_length=self.config["ha_max_input_length"], truncation=True, padding=False, return_tensors="pt")
                prompt_length = len(prompt_tokens)
                label[:prompt_length] = torch.tensor([IGNORE_INDEX] * prompt_length)  # mask the query
            return target_encodings, labels, dataset_length
        else:
            input_encodings = self.tokenizer(inputs, max_length=self.config["ha_max_input_length"], truncation=True, padding="max_length", return_tensors="pt")
            return input_encodings, dataset_length
