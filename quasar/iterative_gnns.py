import json
import numpy as np
import os
import sys
import time
import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm

from quasar.heterogeneous_answering.graph_neural_network.graph_neural_network import GNNModule
from quasar.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from quasar.library.utils import get_config, get_logger, store_json_with_mkdir
import quasar.heterogeneous_answering.graph_neural_network.dataset_gnn as dataset

torch.autograd.set_detect_anomaly(True)

SEED = 7
START_DATE = time.strftime("%y-%m-%d_%H-%M", time.localtime())

class IterativeGNNs(HeterogeneousAnswering):
    def __init__(self, config):
        super(IterativeGNNs, self).__init__(config)
        self.logger = get_logger(__name__, config)

        self.configs = []
        self.gnns = []
        self.model_loaded = False

        self.logger.info(f"Config: {json.dumps(self.config, indent=4)}")

    def _prepare_config_for_iteration(self, config_at_i):
        """
        Construct a standard GNN config for the given iteration, from the multi GNN config.
        """
        config_for_iteration = self.config.copy()
        for key, value in config_at_i.items():
            config_for_iteration[key] = value
        return config_for_iteration

    def train(self, sources=("kb", "text", "table", "info"), random_seed=None):
        """
        Train the model on generated HA data (skip instances for which answer is not there).
        """
        self.logger.info(f"Loading data...")

        # initialize GNNs
        self.configs = [
            self._prepare_config_for_iteration(config_at_i)
            for config_at_i in self.config["gnn_train"]
        ]

        self.gnns = list()
        for i in range(len(self.config["gnn_train"])):
            if random_seed:
                # should ensure that sequential training results in same models as individual training
                # -> reset random states before updates
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
            gnn = GNNModule(self.configs[i], iteration=i + 1)
            self.gnns.append(gnn)

        # input paths
        data_dir, _ = self._get_data_dirs(sources)
        method_name = self.config["name"]
        train_path = os.path.join(data_dir, f"train_ers-{method_name}.jsonl")
        dev_path = os.path.join(data_dir, f"dev_ers-{method_name}.jsonl")

        # train the specified GNN variants
        for i in range(len(self.configs)):
            if random_seed:
                # should ensure that sequential training results in same models as individual training
                # -> reset random states before updates
                random.seed(random_seed)
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
            self.gnns[i].train(sources, train_path, dev_path)

        # done
        self.logger.info(f"Finished training.")

    def inference(self, sources=("kb", "text", "table", "info")):
        """
        Run inference for config. Can be used to continue after ERS results.
        """
        self.load()

        # open data
        method_name = self.config["name"]
        _, data_dir = self._get_data_dirs(sources)
        input_path = f"{data_dir}/res_{method_name}_ers.json"
        with open(input_path, "r") as fp:
            input_data = json.load(fp)

        # inference
        self.inference_on_data(input_data, sources)

        # store result
        output_path = f"{data_dir}/res_{method_name}_gold_answers.json"
        store_json_with_mkdir(input_data, output_path)

        # log results
        turns = (turn for conv in input_data for turn in conv["questions"])
        self.log_results(turns)
        self.logger.info(f"Finished inference.")


    def inference_on_data_split(self, input_path, output_path, sources, train=False, jsonl=False):
        """
        Run Iterative GNNs on given data split.
        """
        batch_size = self.config["gnn_inference_batch_size"]

        curr_input_path = input_path
        num_iterations = len(self.config["gnn_inference"])
        with torch.no_grad():
            for i in range(num_iterations):
                self.gnns[i].gnn.eval()
                self.config["gnn_max_evidences"] = self.config["gnn_inference"][i]["gnn_max_evidences"]
                self.config["gnn_max_entities"] = self.config["gnn_inference"][i]["gnn_max_entities"]
                if os.path.exists(f"{curr_input_path}.cache"):
                    self.logger.info(f"Loading data from cache: {curr_input_path}")
                    data = torch.load(f"{curr_input_path}.cache", weights_only=False)
                elif os.path.exists(f"{curr_input_path}.lazyload"):
                    data = dataset.DatasetGNN(self.config, data_path=f"{curr_input_path}.lazyload", train=False, lazy_load=True)
                else:
                    self.logger.info(f"Loading data from {curr_input_path}")
                    data = dataset.DatasetGNN(self.config, data_path=curr_input_path, train=False)

                data_loader = DataLoader(
                    data, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate_fn
                )

                qa_metrics = list()
                top_evidences_list = list()
                ranked_answers_list = list()
                for _, instances in enumerate(tqdm(data_loader)):
                    # move data to gpu (if possible)
                    GNNModule._move_to_cuda(instances)
                    # make prediction / forward pass
                    output = self.gnns[i].gnn(instances, train=False)
                    # aggregate results
                    qa_metrics += output["qa_metrics"]
                    top_evidences_list += [res["top_evidences"] for res in output["evidence_predictions"]]
                    ranked_answers_list += [res["ranked_answers"] for res in output["answer_predictions"]]
                    
                    # free up GPU space
                    del instances
                    del output
                
                del(data)
                del(data_loader)

                top_evidences_iter = iter(top_evidences_list)
                ranked_answers_iter = iter(ranked_answers_list)
                qa_metrics_iter = iter(qa_metrics)

                # set output path
                if i == num_iterations - 1:
                    curr_output_path = output_path
                else:
                    curr_output_path = output_path.replace(".jsonl", f"-iteration{i}.jsonl")

                # log results
                if qa_metrics:
                    metrics = qa_metrics[0].keys()
                    avg_qa_metrics = {
                        key: f"{(sum([res[key] for res in qa_metrics]) / len([res[key] for res in qa_metrics])):.3f}"
                        for key in metrics
                    }
                    avg_qa_metrics["num_questions"] = len(qa_metrics)
                else:
                    avg_qa_metrics = None
                qa_metrics = f"QA metrics (Iteration {i}): {avg_qa_metrics}."
                self.logger.info(qa_metrics)
                
                # write out
                with open(curr_input_path, "r") as fp_in, open(curr_output_path, "w") as fp_out:
                    line = fp_in.readline()
                    while line:
                        conversation = json.loads(line)
                        for turn in conversation["questions"]:
                            cur_qa_metrics = next(qa_metrics_iter)
                            for metric, val in cur_qa_metrics.items():
                                turn[metric] = val
                            turn["top_evidences"] = next(top_evidences_iter)
                            turn["ranked_answers"] = next(ranked_answers_iter)
                        line = fp_in.readline()
                        fp_out.write(json.dumps(conversation) + "\n")

                del(qa_metrics)
                del(top_evidences_list)
                del(ranked_answers_list)
                del(top_evidences_iter)
                del(ranked_answers_iter)
                del(qa_metrics_iter)

                # prepare for next iteration
                curr_input_path = curr_output_path

    def inference_on_turns(self, turns, sources=("kb", "text", "table", "info"), train=False, use_tqdm=True):
        """Run inference on a set of turns."""
        self.load()

        for i in range(len(self.config["gnn_inference"])):
            # compute ans pres
            if "answers" in turns[0]:
                answer_presence_list = [turn["answer_presence"] for turn in turns]
                answer_presence = sum(answer_presence_list) / len(answer_presence_list)
                answer_presence = round(answer_presence, 3)

            self.gnns[i].inference_on_turns(turns, sources, train, use_tqdm)

        # remember top evidences
        for _, turn in enumerate(turns):
            # identify supporting evidences
            turn["supporting_evidences"] = self.get_supporting_evidences(turn)
        return turns

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a single turn."""
        return self.inference_on_turns([turn], sources, train, use_tqdm=False)[0]

    def dev(self, sources=("kb", "text", "table", "info")):
        """Evaluate the iterative GNN on the dev set."""
        self._eval(sources, "dev")

    def test(self, sources=("kb", "text", "table", "info")):
        """Evaluate the iterative GNN on the test set."""
        self._eval(sources, "test")

    def rerank(self, sources=("kb", "text", "table", "info")):
        self._eval(sources, "train")
        self._eval(sources, "dev")
        self._eval(sources, "test")

    def rerank_prepare(self, sources=("kb", "text", "table", "info")):
        """Run GNNs for re-ranking on all splits."""
        # lazyload variant better for larger datasets
        method_name = self.config["name"]
        if self.config["benchmark"] == "timequestions":
            for split in ["train", "dev", "test"]:
                input_dir, _ = self._get_data_dirs(sources)
                # input_path = os.path.join(input_dir, f"{split}_ers-{method_name}-1k-k10.jsonl")
                lazy_load_path = f"{input_path}.lazyload"
                self.logger.info(f"Preparing lazyload dataset at {lazy_load_path}")
                dataset.DatasetGNN.prepare_lazy_load(self.config, input_path, lazy_load_path, train=False)
                self.logger.info(f"Done with lazyload dataset at {lazy_load_path}!")
        else:
            for split in ["train", "dev", "test"]:
                input_dir, _ = self._get_data_dirs(sources)
                input_path = os.path.join(input_dir, f"{split}_ers-{method_name}-1k-k10.jsonl")
                cache_path = f"{input_path}.cache"
                self.logger.info(f"Preparing lazyload dataset at {cache_path}")
                data = dataset.DatasetGNN(self.config, input_path, train=False)
                torch.save(data, cache_path)
                self.logger.info(f"Done with lazyload dataset at {cache_path}!")


    def _eval(self, sources, split="dev"):
        """Evaluate the iterative GNN on the given split."""
        # set paths
        method_name = self.config["name"]
        input_dir, output_dir = self._get_data_dirs(sources)
        # input_path = os.path.join(input_dir, f"{split}_ers-quasar-1k-k10.jsonl")
        input_path = os.path.join(input_dir, f"{split}_ers-{method_name}-1k-k10.jsonl")
        output_path = os.path.join(output_dir, f"{split}_rerank-{method_name}-100.jsonl")

        # evaluate
        self.load()
        self.inference_on_data_split(input_path, output_path, sources, jsonl=True)
        self.logger.info(f"Finished evaluation on {split}-set.")

    def get_supporting_evidences(self, turn):
        """
        Get the supporting evidences for the answer.
        This function overwrites the model-agnostic implementation in
        the HeterogeneousAnswering class. The (top) evidences used in the
        final GNN layer are used as supporting evidences.
        """
        return turn["top_evidences"]

    def get_answering_evidences(self, turn, turn_idx, turns_before_iteration):
        """
        Get the neighboring evidences of the answer in the initial graph,
        i.e. the answering evidences.
        """
        num_explaining_evidences = self.config["ha_max_supporting_evidences"]
        top_evidences = turns_before_iteration[0][turn_idx]["top_evidences"]
        if not turn["ranked_answers"]:
            return []
        answer_entity = turn["ranked_answers"][0]["answer"]
        answering_evidences = [
            ev
            for ev in top_evidences
            if answer_entity["id"] in [item["id"] for item in ev["wikidata_entities"]]
        ]
        answering_evidences = sorted(answering_evidences, key=lambda j: j["score"], reverse=True)

        # pad evidences to same number
        evidences_captured = set([evidence["evidence_text"] for evidence in answering_evidences])
        if len(answering_evidences) < num_explaining_evidences:
            additional_evidences = sorted(top_evidences, key=lambda j: j["score"], reverse=True)
            for ev in additional_evidences:
                if len(answering_evidences) == num_explaining_evidences:
                    break
                if not ev["evidence_text"] in evidences_captured:
                    answering_evidences.append(ev)
                    evidences_captured.add(ev["evidence_text"])

        return answering_evidences

    def load(self):
        """Load models."""
        if not self.model_loaded:
            # initialize and load GNNs
            self.configs = [
                self._prepare_config_for_iteration(config_at_i)
                for config_at_i in self.config["gnn_inference"]
            ]
            self.gnns = [
                GNNModule(self.configs[i], iteration=i + 1)
                for i in range(len(self.config["gnn_inference"]))
            ]
            for gnn in self.gnns:
                gnn.load()

            # remember that model is loaded
            self.model_loaded = True


def main():
    # reproducibility
    SEED = 7
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if len(sys.argv) < 2:
        raise Exception(
            "python quasar/heterogeneous_answering/graph_neural_network/iterative_gnns.py --<FUNCTION> <PATH_TO_CONFIG> [<SOURCES_STR>]"
        )

    function = sys.argv[1]
    config_path = sys.argv[2]
    sources_str = sys.argv[3] if len(sys.argv) > 3 else "kb_text_table_info"
    config = get_config(config_path)

    if function == "--train":
        # train
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.train(sources=sources, random_seed=SEED)

    elif function == "--test":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.test(sources=sources)

    elif function == "--inference":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.inference(sources=sources)

    elif function == "--dev":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.dev(sources=sources)

    elif function == "--rerank":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.rerank(sources=sources)

    elif function == "--rerank_prepare":
        gnn = IterativeGNNs(config)
        sources = sources_str.split("_")
        gnn.rerank_prepare(sources=sources)
    else:
        raise NotImplementedError(f"Function {function} not implemented.")

if __name__ == "__main__":
    main()
