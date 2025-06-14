import torch
import transformers

from quasar.library.utils import get_logger


class Encoder(torch.nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config
        self.logger = get_logger(__name__, config)

    def encode_srs_batch(self, srs):
        """Encode all SRs in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def encode_evidences_batch(self, evidences, *args):
        """Encode all evidences in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def encode_entities_batch(self, entities, *args):
        """Encode all entities in the batch."""
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )


    def initialize(self, config):
        # load params
        self.emb_dimension = config["gnn_emb_dimension"]

        self.max_input_length_sr = config["gnn_enc_sr_max_input"]
        self.max_input_length_ev = config["gnn_enc_ev_max_input"]
        self.max_input_length_ent = config["gnn_enc_ent_max_input"]

        # initialize LM
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config["gnn_encoder_lm"])
        self.model = transformers.AutoModel.from_pretrained(config["gnn_encoder_lm"])
        self.sep_token = self.tokenizer.sep_token

    def _batched_encodings(self, tokenized_input, batch_size=256):
        # move to cuda
        if torch.cuda.is_available():
            tokenized_input = tokenized_input.to(torch.device("cuda"))
        
        # run inference
        encodings = self.model(**tokenized_input).last_hidden_state
        
        # move to cuda
        if torch.cuda.is_available():
            encodings = encodings.cuda()
        return encodings