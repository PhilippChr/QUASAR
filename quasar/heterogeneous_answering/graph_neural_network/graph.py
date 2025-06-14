import time
import itertools
import networkx as nx

WIKIDATA_ENTITIES_SEP = "<BR>" + 5*"&nbsp;"
 
class Graph:
	def __init__(self):
		"""Create a new empty graph."""
		self.nx_graph = nx.Graph()
		self.nodes_dict = dict()
		self.ev_to_ent_dict = dict()
		self.ent_to_score = dict()

	def _add_entity(self, entity):
		g_ent_id = f'ent{entity["g_id"]}'
		self.nx_graph.add_node(
			g_ent_id,
			type='entity',
			entity_type=entity["type"] if "type" in entity and entity["type"] else "None",
			label=entity["label"],
			wikidata_id=entity["id"],
			is_question_entity="is_question_entity" in entity,
			is_answer="is_answer" in entity and entity["is_answer"],
			is_predicted_answer=False
		)

	def _add_evidence(self, evidence):
		g_ev_id = f'ev{evidence["g_id"]}'
		self.nx_graph.add_node(
			g_ev_id,
			type='evidence',
			label=evidence["evidence_text"],
			source=evidence["source"],
			wikidata_entities=WIKIDATA_ENTITIES_SEP.join([f'"{e["label"]}" => {e["id"]}' for e in evidence["wikidata_entities"]]),
			retrieved_for_entity=str(evidence["retrieved_for_entity"]),
			is_answering_evidence="is_answering_evidence" in evidence and evidence["is_answering_evidence"]
		)

	def from_instance(self, instance):
		""" Create a new graph from the given dataset instance. """
		# add entity nodes
		entities = instance["entities"]
		for entity in entities:
			if not "g_id" in entity:
				continue
			self._add_entity(entity)

		# add evidence nodes
		evidences = instance["evidences"]
		for evidence in evidences:
			if not "g_id" in evidence:
				continue
			self._add_evidence(evidence)

		# add edges
		ent_to_ev = instance["ent_to_ev"]
		for i, entity in enumerate(entities):
			if not "g_id" in entity:
				continue
			g_ent_id = f'ent{entity["g_id"]}'
			connected_ev_ids = ent_to_ev[i, :]
			for j, val in enumerate(connected_ev_ids):
				if val > 0:
					g_ev_id = f'ev{j}'
					self.nx_graph.add_edge(g_ent_id, g_ev_id)
		return self

	def from_scoring_output(self, scored_evidences, scored_entities, ent_to_ev, ques_id=None):
		""" Create an evidence-only graph from the outputs of the scoring phase. """

		# DEV: save data for quick development
		# torch.save(tensor, 'file.pt') and torch.load('file.pt')
		# if ques_id:
		# 	with open(f"tmp_data/tmp_data_{ques_id}.json", "w") as fp:
		# 		json.dump([scored_evidences, scored_entities], fp)
		# 	torch.save(ent_to_ev, f'tmp_data/tmp_ent_to_ev_{ques_id}.pt')

		for entity in scored_entities:
			self.ent_to_score[entity["id"]] = entity["score"]

		# add evidence nodes
		for evidence in scored_evidences:
			if not "g_id" in evidence: # padded evidence
				continue
			self._add_evidence(evidence)
			node_id = f'ev{evidence["g_id"]}'
			self.nodes_dict[node_id] = evidence
			self.ev_to_ent_dict[node_id] = [entity["id"] for entity in evidence["wikidata_entities"] if entity["id"] in self.ent_to_score]

		for i, evidence1 in enumerate(scored_evidences):
			if not "g_id" in evidence1: # padded evidence
				continue
			for j, evidence2 in enumerate(scored_evidences):
				# avoid duplicate checks or checks with same item
				if i >= j or not "g_id" in evidence2:
					continue 
				# derive set of entities
				entities1 = set([entity["id"] for entity in evidence1["wikidata_entities"]])
				entities2 = set([entity["id"] for entity in evidence2["wikidata_entities"]])

				# if shared entity, there is a connection
				if entities1 & entities2:
					g_ev_id1 = f'ev{evidence1["g_id"]}'
					g_ev_id2 = f'ev{evidence2["g_id"]}'

					# add edge
					self.nx_graph.add_edge(g_ev_id1, g_ev_id2)
		return self

	def _get_connected_subgraph(self, scored_evidences, scored_entities, max_evidences):
		"""
		From the given graph, get a connected subgraph that has 
		at most `max_evidences`. The output will be the set of
		evidences within this subgraph.
		"""
		start_time = time.time()
		top_evidences = scored_evidences[:max_evidences] if max_evidences < len(scored_evidences) else None
		# num evidences given as input should be higher than max output size
		if top_evidences is None:
			return scored_evidences
		top_evidences_nodes = [f'ev{evidence["g_id"]}' for evidence in top_evidences]
		top_evidences_subgraph = self.nx_graph.subgraph(top_evidences_nodes).copy()
		top_evidences_score = self._get_score_of_subgraph(top_evidences_nodes)
		# sum([evidence["score"] for evidence in top_evidences])

		# in case the top-5 nodes are connected, these obviously induce the subgraph with highest score
		if nx.is_connected(top_evidences_subgraph):
			return top_evidences

		print("Num nodes in top_evidences subgraph", len(top_evidences_subgraph.nodes))
		print("Num edges in top_evidences subgraph", len(top_evidences_subgraph.edges))
		print("Total score of top_evidences subgraph", top_evidences_score)

		# loose initial lower bound: max score of individual node (top-1 score)
		max_score = scored_evidences[0]["score"]
		max_nodes = {f'ev{scored_evidences[0]["g_id"]}'}

		# go through components
		components_counter  = 0
		for component in nx.connected_components(self.nx_graph):
			components_counter += 1
			print(f"Component number {components_counter}")
			if time.time() - start_time > 60:
				print("Time limit reached!")
				break

			# check if total score higher than current max
			max_score_component = self._get_score_of_subgraph(component)
			if max_score_component <= max_score:
				print("drop component")
				continue

			combination_size = min(len(component), max_evidences)
			# enumerate all possible combinations of n nodes
			combination_counter = 0
			for nodes in itertools.combinations(component, combination_size):
				combination_counter += 1
				print(f"Combination number {combination_counter}")
				if time.time() - start_time > 60:
					print("Time limit reached!")
					break

				# check if the nodes are connected
				if nx.is_connected(self.nx_graph.subgraph(nodes)):
					# calculate the aggregate score for the set of nodes
					score = self._get_score_of_subgraph(nodes)
					if score > max_score:
						max_score = score
						max_nodes = nodes

		top_evidences = [self.nodes_dict[node] for node in max_nodes]
		top_evidences.sort(key=lambda x: x["score"], reverse=True)

		top_evidences_subgraph = self.nx_graph.subgraph(max_nodes).copy()
		top_evidences_score = max_score
		# sum([evidence["score"] for evidence in top_evidences])

		print("Num nodes in updated_top_evidences subgraph", len(top_evidences_subgraph.nodes))
		print("Num edges in updated_top_evidences subgraph", len(top_evidences_subgraph.edges))
		print("Total score of updated_top_evidences subgraph", top_evidences_score)
		print("len(top_evidences)", len(top_evidences))
		return top_evidences

	def _get_score_of_subgraph(self, evidence_nodes):
		"""Compute the score of the subgraph defined by the evidence nodes."""
		evidences_score = sum(self.nodes_dict[node]["score"] for node in evidence_nodes)
		seen_entities = set()
		entities_score = 0
		for node in evidence_nodes:
			for entity_id in self.ev_to_ent_dict[node]:
				if not entity_id in seen_entities:
					entities_score += self.ent_to_score[entity_id]
					seen_entities.add(entity_id)
		return evidences_score + entities_score

	def _get_diversified_evidences(self, scored_evidences, scored_entities, max_evidences):
		max_evidences_per_component = int(max_evidences/3)
		# identify components
		components = nx.connected_components(self.nx_graph)
		components = sorted(components, key=lambda c: self._get_score_of_subgraph(c), reverse=True)
		
		# add evidences
		top_evidences = list()
		for component in components:
			c_top_evidences = [self.nodes_dict[node] for node in component]
			c_top_evidences.sort(key=lambda x: x["aggregated_score"], reverse=True)
			evidences_to_add = max_evidences - len(top_evidences)
			if evidences_to_add:
				evidences_to_add = min(max_evidences_per_component, evidences_to_add)
				c_top_evidences = c_top_evidences[:evidences_to_add]
				top_evidences += c_top_evidences

		# fill-up remaining space
		evidences_ids = set(ev["g_id"] for ev in top_evidences)
		for ev in scored_evidences:
			if len(top_evidences) == max_evidences:
				break
			if not ev["g_id"] in evidences_ids:
				top_evidences.append(ev)
				evidences_ids.add(ev["g_id"])
		return top_evidences

	def from_nx_graph(self, nx_graph):
		"""Create an instance of "Graph" from a given nx graph."""
		self.nx_graph = nx_graph

	def write_to_file(self, file_path="/home/pchristm/public_html/graph.gexf"):
		"""Write the graph to file."""
		xml = self.to_string()
		with open(file_path, "w") as fp:
			fp.write(xml)

	def to_string(self):
		"""Write the graph to String."""
		xml_lines = nx.generate_gexf(self.nx_graph)
		xml = "\n".join(xml_lines)
		for i in range(20):
			# fix attributes in xml string
			i_str = str(i)
			if f'id="{i_str}" title="' in xml:
				title = xml.split(f'id="{i_str}" title="', 1)[1].split('"', 1)[0]
				xml = xml.replace(f'id="{i_str}" title="', f'id="{title}" title="')
				xml = xml.replace(f'for="{i_str}"', f'for="{title}"')
		xml = "<?xml version='1.0' encoding='utf-8'?>\n" + xml
		return xml

	def get_answer_neighborhood(self, answer_entity):
		""" Get the 2-hop neighborhood of the answer in a graph (surrounding evidences->entities."""
		graph = Graph()
		if not "g_id" in answer_entity:
			return self
		g_ent_id = f'ent{answer_entity["g_id"]}'
		self.nx_graph.nodes[g_ent_id]["is_predicted_answer"] = True
		graph.from_nx_graph(nx.ego_graph(self.nx_graph, g_ent_id, radius=2))
		return graph

