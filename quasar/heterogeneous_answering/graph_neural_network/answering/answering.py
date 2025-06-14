import numpy as np
import torch
import warnings
from sklearn.metrics import balanced_accuracy_score

import quasar.evaluation as evaluation
from quasar.library.utils import get_logger
from quasar.heterogeneous_answering.graph_neural_network.graph import Graph


SOURCES = [["kb"], ["text"], ["table", "info"]]


class Answering(torch.nn.Module):
    def __init__(self, config):
        super(Answering, self).__init__()

        self.config = config
        self.logger = get_logger(__name__, config)

    def forward(self, *args):
        raise Exception(
            "This is an abstract function which should be overwritten in a derived class!"
        )

    def evaluate(self, batch, answer_predictions, evidence_predictions=None):
        """Evaluate the outputs, using standard QA metrics."""
        # iterate through batch
        results = list()
        for idx in range(len(batch["gold_answers"])):
            # load
            gold_answers = batch["gold_answers"][idx]
            ranked_answers = answer_predictions[idx]["ranked_answers"]

            # eval answers
            result = dict()
            p_at_1 = evaluation.precision_at_1(ranked_answers, gold_answers)
            result["p_at_1"] = p_at_1
            mrr = evaluation.mrr_score(ranked_answers, gold_answers)
            result["mrr"] = mrr
            h_at_5 = evaluation.hit_at_5(ranked_answers, gold_answers)
            result["h_at_5"] = h_at_5

            # eval evidence predictions
            if evidence_predictions:
                top_evidences = evidence_predictions[idx]["top_evidences"]
                ans_pres, _ = evaluation.answer_presence(top_evidences, gold_answers, relaxed=True)
                result["answer_presence"] = 1 if ans_pres else 0
                if "top30_evidences" in evidence_predictions[idx]:
                    ans_pres, _ = evaluation.answer_presence(evidence_predictions[idx]["top30_evidences"], gold_answers, relaxed=True)
                    result["answer_presence@30"] = 1 if ans_pres else 0
                result["mrr_evidences"] = evaluation.mrr_score_evidences(top_evidences, gold_answers, relaxed=True)
                result["num_evidences"] = len(top_evidences)
            else:
                result["answer_presence"] = 0
            results.append(result)

        return results

    def add_ranked_answers(self, batch, answer_logits):
        """Add the ranked answers (and predictions) to the output."""
        logits_cp = answer_logits.clone().detach()
        if logits_cp.size(dim=2) == 1:
            predictions = logits_cp[:, :, 0]
        else:
            predictions = logits_cp[:, :, 1]

        # iterate through batch
        results = list()
        for idx in range(len(batch["id_to_entity"])):
            question = batch["question"][idx]
            id_to_entity = batch["id_to_entity"][idx]
            predictions_for_q = predictions[idx]

            # get ranked answers
            ranked_answers = self.get_ranked_answers(question, predictions_for_q, id_to_entity)

            # add predictions
            result = dict()
            result["ranked_answers"] = ranked_answers

            # append
            results.append(result)
        return results

    def get_ranked_answers(self, question, predictions_for_q, id_to_entity):
        """For the predictions by the GNN, get the ranked answers."""
        # check if existential (special treatment)
        if evaluation.question_is_existential(question):
            ranked_answers = [
                {"answer": {"id": "yes", "label": "yes"}, "score": 1.0, "rank": 1},
                {"answer": {"id": "no", "label": "no"}, "score": 0.5, "rank": 2},
            ]
            return ranked_answers

        # sort answers
        top_candidate_ids = torch.sort(predictions_for_q, dim=0, descending=True)
        # filter out padded entities
        top_candidate_indices = [
            idx for idx in top_candidate_ids.indices if not id_to_entity[idx] == 0
        ]
        # reduce to max num of answers
        top_candidate_indices = top_candidate_indices[: self.config["ha_max_answers"]]

        ranked_answers = [
            {
                "answer": id_to_entity[idx],
                "score": top_candidate_ids.values[i].item(),
                "rank": (i + 1),
                "local_id": idx.item(),
            }
            for i, idx in enumerate(top_candidate_indices)
        ]
        return ranked_answers

    def add_top_evidences(self, batch, evidence_logits, answer_logits, answer_predictions):
        """Add the top ranked evidences to the output."""
        logits_cp = evidence_logits.clone().detach()
        if logits_cp.size(dim=2) == 1:
            ev_predictions = logits_cp[:, :, 0]
        else:
            ev_predictions = logits_cp[:, :, 1]

        logits_cp = answer_logits.clone().detach()
        if logits_cp.size(dim=2) == 1:
            ans_predictions = logits_cp[:, :, 0]
        else:
            ans_predictions = logits_cp[:, :, 1]

        # iterate through batch
        results = list()
        for idx in range(len(batch["id_to_evidence"])):

            # get scored evidences
            id_to_evidence = batch["id_to_evidence"][idx]
            ev_predictions_for_q = ev_predictions[idx]
            top_candidate_ids = torch.sort(ev_predictions_for_q, dim=0, descending=True)
            scored_evidences = [
                {
                    "evidence_text": id_to_evidence[candidate_idx]["evidence_text"],
                    "retrieved_for_entity": id_to_evidence[candidate_idx]["retrieved_for_entity"],
                    "wikidata_entities": id_to_evidence[candidate_idx]["wikidata_entities"],
                    "disambiguations": id_to_evidence[candidate_idx]["disambiguations"],
                    "source": id_to_evidence[candidate_idx]["source"],
                    "score": ev_predictions_for_q[candidate_idx].item(),
                    "g_id": id_to_evidence[candidate_idx]["g_id"],
                    "idx": candidate_idx.item(),
                    "is_answering_evidence": id_to_evidence[candidate_idx]["is_answering_evidence"],
                    "wikipedia_path": id_to_evidence[candidate_idx]["wikipedia_path"] if "wikipedia_path" in id_to_evidence[candidate_idx] else None
                }
                for candidate_idx in top_candidate_ids.indices
                if not id_to_evidence[candidate_idx] == 0
            ]

            # get scored entities
            id_to_entity = batch["id_to_entity"][idx]
            ans_predictions_for_q = ans_predictions[idx]
            top_candidate_ids = torch.sort(ans_predictions_for_q, dim=0, descending=True)
            scored_entities = [
                {
                    "id": id_to_entity[candidate_idx]["id"],
                    "label": id_to_entity[candidate_idx]["label"],
                    "type": id_to_entity[candidate_idx]["type"]
                    if "type" in id_to_entity[candidate_idx]
                    else None,
                    "score": ans_predictions_for_q[candidate_idx].item(),
                    "g_id": id_to_entity[candidate_idx]["g_id"],
                    "idx": candidate_idx.item(),
                    "is_answer": id_to_entity[candidate_idx]["is_answer"],
                }
                for candidate_idx in top_candidate_ids.indices
                if not id_to_entity[candidate_idx] == 0
            ]

            # get top-k evidences
            ent_to_ev = batch["ent_to_ev"][idx]
            scored_evidences = self._filter_evidences_by_sources(scored_evidences)
            top_evidences = self._compute_reduced_graph(self.config["gnn_max_output_evidences"], scored_evidences, scored_entities, ent_to_ev)
            top30_evidences = self._compute_reduced_graph(30, scored_evidences, scored_entities, ent_to_ev)

            # DEV: print missed answers
            # question = batch["question"][idx]
            # gold_answers = batch["gold_answers"][idx]
            # if not evaluation.answer_presence(top_evidences, gold_answers)[0]:
            #     ans_pres, answering_evidences = evaluation.answer_presence(scored_evidences, gold_answers)
            #     if ans_pres and len(answering_evidences) <= 15:
            #         print(f"\n\n\nFound an instance: question={question}, answers={gold_answers}")
            #         print(f"Top-5 answer predictions: {answer_predictions[idx]["ranked_answers"][:5]}")
            #         print(f"Top-5 evidences: {top_evidences[:5]}")
            #         for e in answering_evidences:
            #             print(f"    Missed evidence={e}")
            #     elif ans_pres:
            #         print(f"\n\nSkipping instance (question={question}, answers={gold_answers}) with {len(answering_evidences)} answering evidences.")

            # add predictions
            result = dict()
            result["top_evidences"] = top_evidences
            result["top30_evidences"] = top30_evidences
            result["ev_predictions"] = ev_predictions_for_q
            result["ans_predictions"] = ans_predictions_for_q

            # append
            results.append(result)
        return results
    
    def _filter_evidences_by_sources(self, scored_evidences):
        """
        Used for pruning evidence from specific sources at GNN.
        """
        if self.config.get("gnn_evidence_sources"):
            sources = self.config["gnn_evidence_sources"]
            scored_evidences = [ev for ev in scored_evidences if ev["source"] in sources]
        return scored_evidences

    def _compute_reduced_graph(self, max_evidences, scored_evidences, scored_entities, ent_to_ev, ques_id="0"):
        """
        Given the scores for evidences and entities in the current iteration,
        compute the input graph for the next iteration.
        """
        strategy = self.config["gnn_subgraph_strategy"] if "gnn_subgraph_strategy" in self.config else "naive"

        if strategy == "naive":
            top_evidences = scored_evidences[:max_evidences]
        elif strategy == "connected":
            ers_graph = Graph().from_scoring_output(scored_evidences, scored_entities, ent_to_ev, ques_id=ques_id)
            top_evidences = ers_graph._get_connected_subgraph(scored_evidences, scored_entities, max_evidences)
        elif strategy == "distributed":
            top_evidences = self._get_distributed_evidences(scored_evidences, max_evidences)
        elif strategy == "aggregate":
            top_evidences = self._get_highest_scoring_evidences(scored_evidences, scored_entities, max_evidences)
        elif strategy == "diverse":
            ent_to_score = {entity["id"]: entity["score"] for entity in scored_entities}
            for evidence in scored_evidences:
                ev_score = evidence["score"]
                ev_entities = [entity["id"] for entity in evidence["wikidata_entities"] if entity["id"] in ent_to_score]
                ent_scores = sum(ent_to_score[ent] for ent in ev_entities)
                evidence["aggregated_score"] = ev_score + ent_scores
                evidence["ent_scores"] = {ent: ent_to_score[ent] for ent in ev_entities}
            ers_graph = Graph().from_scoring_output(scored_evidences, scored_entities, ent_to_ev, ques_id=ques_id)
            top_evidences = ers_graph._get_diversified_evidences(scored_evidences, scored_entities, max_evidences)
            top_evidences = sorted(top_evidences, key=lambda ev: ev["aggregated_score"], reverse=True)
        else:
            raise Exception(f"Strategy {strategy} for computing the reduced subgraph not available!")
        return top_evidences

    def _get_highest_scoring_evidences(self, scored_evidences, scored_entities, max_evidences):
        ent_to_score = {entity["id"]: entity["score"] for entity in scored_entities}
        for evidence in scored_evidences:
            ev_score = evidence["score"]
            ev_entities = [entity["id"] for entity in evidence["wikidata_entities"] if entity["id"] in ent_to_score]
            ent_scores = sum(ent_to_score[ent] for ent in ev_entities)
            evidence["aggregated_score"] = ev_score + ent_scores
            evidence["ent_scores"] = {ent: ent_to_score[ent] for ent in ev_entities}
        top_evidences = sorted(scored_evidences, key=lambda ev: ev["aggregated_score"], reverse=True)[:max_evidences]
        return top_evidences
    
    def _get_distributed_evidences(self, scored_evidences, max_evidences):
        max_evidences_per_source = int(max_evidences/3)
        evidence_tuples = list()
        for sources in SOURCES:
            evidences_src = [(idx, ev) for idx, ev in enumerate(scored_evidences) if ev["source"] in sources][:max_evidences_per_source]
            evidence_tuples += evidences_src
        taken_evidences = set(idx for idx, _ in evidence_tuples)
        evidences = list(ev for _, ev in evidence_tuples)
        if len(evidences) < max_evidences:
            evidences_to_add = max_evidences - len(evidences)
            for idx, ev in enumerate(scored_evidences):
                if evidences_to_add == 0:
                    break
                if idx not in taken_evidences:
                    evidences_to_add -= 1
                    evidences.append(ev)
        return evidences

    def compute_balanced_accuracy(self, outputs, batch, labels_key="entity_labels"):
        """
        Method to compute balanced accuracy for predictions (for batch).
        Takes care of the class imbalance in the graph (few answer nodes, many candidates).
        """
        labels = batch[labels_key]

        # check if numpy or not
        if type(outputs).__module__ == np.__name__:
            logits_cp = outputs
        else:
            logits_cp = outputs.clone().detach()

        # predict classes
        if torch.cuda.is_available():
            predictions = np.argmax(logits_cp.cpu(), axis=-1)
            labels = labels.cpu()
        else:
            predictions = np.argmax(logits_cp, axis=-1)

        # balanced accuracies
        with warnings.catch_warnings(record=True) as _:
            balanced_accs = list()
            for i, _ in enumerate(predictions):
                res = balanced_accuracy_score(labels[i], predictions[i])
                balanced_accs.append(res)
        return balanced_accs
