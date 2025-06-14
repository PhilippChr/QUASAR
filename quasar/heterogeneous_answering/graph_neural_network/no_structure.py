import time
import torch

from quasar.heterogeneous_answering.graph_neural_network.encoders.encoder_factory import EncoderFactory
from quasar.heterogeneous_answering.graph_neural_network.answering.answering_factory import AnsweringFactory

torch.manual_seed(7)

class NoStructureModel(torch.nn.Module):
	""" Model for ablation study, that does not leverage any graph structure. """
	def __init__(self, config):
		super(NoStructureModel, self).__init__()

		self.config = config

		# load parameters
		self.num_layers = config["gnn_num_layers"]
		self.emb_dimension = config["gnn_emb_dimension"]
		self.dropout = config["gnn_dropout"]

		# encoder
		self.encoder = EncoderFactory.get_encoder(config)
		
		# answering
		self.answering = AnsweringFactory.get_answering(config)

		# move layers to cuda
		if torch.cuda.is_available():
			self.cuda()

	def forward(self, batch, train=False):
		"""
		Forward step.
		"""
		# get data
		srs = batch["sr"]
		entities = batch["entities"]
		evidences = batch["evidences"]
		ev_to_ent = batch["ev_to_ent"] # size: batch_size x num_ev x num_ent

		## encoding
		sr_vec = self.encoder.encode_srs_batch(srs)
		evidences_mat = self.encoder.encode_evidences_batch(evidences, srs) # size: batch_size x num_ev x emb
		entities_mat = self.encoder.encode_entities_batch(entities, srs, evidences_mat, ev_to_ent, sr_vec, train) # size: batch_size x num_ent x emb

		## obtain answer probabilities, loss, qa-metrics
		start_time = time.time()
		res = self.answering(batch, train, entities_mat, sr_vec, evidences_mat)
		return res


