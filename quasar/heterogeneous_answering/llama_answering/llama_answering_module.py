import sys
import torch
import random
import numpy as np

import quasar.evaluation as evaluation
import quasar.heterogeneous_answering.llama_answering.dataset_llama_answering as dataset
from quasar.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from quasar.heterogeneous_answering.llama_answering.llama_answering_model import (
    LLaMAModel,
)
from quasar.library.utils import get_config, get_result_logger


class LLaMaAnsweringModule(HeterogeneousAnswering):
    def __init__(self, config, train=True):
        """Initialize LLaMA answering module."""
        super(LLaMaAnsweringModule, self).__init__(config)
        self.result_logger = get_result_logger(config)

        # create model
        self.ha_model = LLaMAModel(config, train=train)

    def train(self, sources=("kb", "text", "table", "info"), **kwargs):
        """Train the model on generated HA data (skip instances for which answer is not there)."""
        # train model
        self.logger.info(f"Starting training...")
        train_path = self.config["answering_train_path"]
        dev_path = self.config["answering_dev_path"]
        self.ha_model.train(train_path, dev_path)
        self.logger.info(f"Finished training.")

    def inference_on_turns(self, input_turns, sources=("kb", "text", "table", "info"), train=False, use_tqdm=True):
        """
        Run HA on a set of turns.
        When train is set to `True`, teacher forcing for th
        GNN inference on the train set is enabled.
        """
        prompt = self._get_prompt()
        num_evidences = self.config.get("ha_max_num_evidences", None)
        add_references = self.config.get("ha_add_references", False)
        input_texts = [
            dataset.turn_to_input(turn, self.ha_model.tokenizer, self.config["ha_max_evidence_tokens"], prompt=prompt, num_evidences=num_evidences, add_references=add_references)
            for turn in input_turns
        ]
        generated_answers = self.ha_model.batch_inference(input_texts)

        for turn, gen_answer in zip(input_turns, generated_answers):
            turn["generated_answer"] = gen_answer
            print("\ngenerated_answer", gen_answer)
            print("gold answers", turn["answers"])
            ranked_answers = evaluation.get_ranked_answers(self.config, gen_answer, turn)
            turn["pred_answers"] = [
                {"id": ans["answer"]["id"], "label": ans["answer"]["label"], "rank": ans["rank"]}
                for ans in ranked_answers
            ]
            self._eval_and_denoise(turn, ranked_answers)
        return input_turns

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a single turn."""
        # prepare input
        prompt = self._get_prompt()
        num_evidences = self.config.get("ha_max_num_evidences", None)
        add_references = self.config.get("ha_add_references", False)
        input_text = dataset.turn_to_input(turn, self.ha_model.tokenizer, self.config["ha_max_evidence_tokens"], prompt=prompt, num_evidences=num_evidences, add_references=add_references)

        # run inference
        generated_answer = self.ha_model.inference(input_text)
        turn["generated_answer"] = generated_answer
        ranked_answers = evaluation.get_ranked_answers(self.config, generated_answer, turn)
        turn["pred_answers"] = [
            {"id": ans["answer"]["id"], "label": ans["answer"]["label"], "rank": ans["rank"]}
            for ans in ranked_answers
        ]
        self._eval_and_denoise(turn, ranked_answers)
        return turn

    def _eval_and_denoise(self, turn, ranked_answers):
        # eval
        if "answers" in turn:
            p_at_1 = evaluation.precision_at_1(ranked_answers, turn["answers"])
            turn["p_at_1"] = p_at_1
            mrr = evaluation.mrr_score(ranked_answers, turn["answers"])
            turn["mrr"] = mrr
            h_at_5 = evaluation.hit_at_5(ranked_answers, turn["answers"])
            turn["h_at_5"] = h_at_5

        # delete noise
        if turn.get("top_evidences"):
            del turn["top_evidences"]
        if turn.get("question_entities"):
            del turn["question_entities"]
        if turn.get("silver_SR"):
            del turn["silver_SR"]
        if turn.get("silver_relevant_turns"):
            del turn["silver_relevant_turns"]
        if turn.get("silver_answering_evidences"):
            del turn["silver_answering_evidences"]

    def _get_prompt(self):
        if self.config.get("answering_prompt"):
            prompt = self.config["answering_prompt"]
        else:
            prompt = dataset.PROMPT
        return prompt

#######################################################################################################################
#######################################################################################################################
if __name__ == "__main__":
    # reproducibility
    SEED = 7
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if len(sys.argv) < 2:
        raise Exception(
            "Usage: python quasar/heterogeneous_answering/llama_answering/llama_answering_module.py --<FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STR>]"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
    config = get_config(config_path)

    # train: train model
    if function == "--train":
        sam = LLaMaAnsweringModule(config)
        sources = sources_str.split("_")
        sam.train(sources=sources)

    # test: add predictions to data
    elif function == "--test":
        sam = LLaMaAnsweringModule(config, train=False)
        input_dir = config["path_to_intermediate_results"]
        output_dir = config["path_to_intermediate_results"]

        qu = config["qu"]
        ers = config["ers"]
        ha = config["ha"]
        method_name = config["name"]
        sources = sources_str.split("_")
        input_path = config["answering_test_path"]
        output_path = config["answering_test_output_path"]
        sam.inference_on_data_split(input_path, output_path, sources)
        print("Done!")
