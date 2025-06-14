import os

from quasar.heterogeneous_answering.heterogeneous_answering import HeterogeneousAnswering
from quasar.heterogeneous_answering.graph_neural_network.iterative_gnns import IterativeGNNs
from quasar.heterogeneous_answering.llama_answering.llama_answering_module import LLaMaAnsweringModule


class QuasarAnsweringModule(HeterogeneousAnswering):
    def __init__(self, config, train=False):
        """Initialize QUASAR answering module."""
        super(QuasarAnsweringModule, self).__init__(config)

        # load modules
        self.reranking_module = IterativeGNNs(self.config)
        self.generation_module = LLaMaAnsweringModule(self.config, train=train)

    def train(self, sources=("kb", "text", "table", "info"), **kwargs):
        # init
        input_dir = self.config["path_to_intermediate_results"]
        qu = self.config["qu"]
        ers = self.config["ers"]
        method_name = self.config["name"]
        sources_string = "_".join(sources)
        
        # train reranker
        # self.reranking_module.train(sources=sources)

        # reranker inference on train
        # input_path = os.path.join(
        #     input_dir, qu, ers, sources_string, f"train_ers-{method_name}.jsonl"
        # )
        # output_path = os.path.join(
        #     input_dir, qu, ers, sources_string, f"train_rerank-{method_name}.jsonl"
        # )
        # self.reranking_module.inference_on_data_split(input_path, output_path, sources)

        # reranker inference on dev
        # input_path = os.path.join(
        #     input_dir, qu, ers, sources_string, f"dev_ers-{method_name}.jsonl"
        # )
        # output_path = os.path.join(
        #     input_dir, qu, ers, sources_string, f"dev_rerank-{method_name}.jsonl"
        # )
        # self.reranking_module.inference_on_data_split(input_path, output_path, sources)

        # train generation
        self.generation_module.train(sources=sources)

    def inference_on_turns(self, input_turns, sources=("kb", "text", "table", "info"), train=False, use_tqdm=True):
        """
        Run HA on a set of turns.
        """
        input_turns = self.reranking_module.inference_on_turns(input_turns, sources)
        input_turns = self.generation_module.inference_on_turns(input_turns, sources)
        return input_turns

    def inference_on_turn(self, turn, sources=("kb", "text", "table", "info"), train=False):
        """Run inference on a single turn."""
        turn = self.reranking_module.inference_on_turn(turn, sources)
        turn = self.generation_module.inference_on_turn(turn, sources)
        return turn
    
