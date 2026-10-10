"""Joint NER/RE network; independent of the Label Studio service."""
import os
import json
import logging
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from transformers import AutoModel
from config import config

logger = logging.getLogger(__name__)


class ModernBERTForNERRE(nn.Module):
    """
    ModernBERT NER+RE 联合模型
    - 共享 ModernBERT Encoder
    - NER 分类头：每个 token 预测实体标签
    - RE 分类头：基于实体对的关系分类
    """

    def __init__(
            self,
            model_path: str,
            num_ner_labels: int,
            num_re_labels: int,
            hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True
        )
        self.bert = self.bert.to(torch.float32)
        self.hidden_size = self.bert.config.hidden_size

        # NER 分类头
        self.ner_classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, num_ner_labels),
        )

        # RE 分类头（实体1表示 + 实体2表示 + 距离嵌入）
        self.re_classifier = nn.Sequential(
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(self.hidden_size, num_re_labels),
        )

        # 距离嵌入
        self.max_distance = 128
        self.distance_embedding = nn.Embedding(
            self.max_distance * 2 + 1, self.hidden_size
        )

        self.register_buffer("ner_label_weights", None)
        self.register_buffer("re_label_weights", None)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            ner_labels: Optional[torch.Tensor] = None,
            entity_pairs: Optional[List[List[Tuple[int, int]]]] = None,
            re_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state

        # NER 预测
        ner_logits = self.ner_classifier(sequence_output)

        # RE 预测
        re_logits = None
        if entity_pairs is not None:
            re_logits = self._compute_re_logits(sequence_output, entity_pairs)

        # 计算损失
        total_loss = None
        if ner_labels is not None:
            if self.ner_label_weights is not None:
                ner_weights = self.ner_label_weights.to(ner_logits.device)
            else:
                ner_weights = torch.ones(ner_logits.size(-1), device=ner_logits.device)
                ner_weights[0] = 0.15
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=ner_weights)
            ner_loss = loss_fct(
                ner_logits.view(-1, ner_logits.size(-1)), ner_labels.view(-1)
            )
            total_loss = config.ner_loss_weight * ner_loss

        if re_labels is not None and re_logits is not None:
            if self.re_label_weights is not None:
                re_weights = self.re_label_weights.to(re_logits.device)
            else:
                re_weights = torch.ones(re_logits.size(-1), device=re_logits.device)
                re_weights[0] = 0.15
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, weight=re_weights)
            re_loss = loss_fct(
                re_logits.view(-1, re_logits.size(-1)), re_labels.view(-1)
            )
            total_loss = (
                (total_loss + config.re_loss_weight * re_loss) if total_loss is not None else config.re_loss_weight * re_loss
            )

        return {
            "loss": total_loss,
            "ner_logits": ner_logits,
            "re_logits": re_logits,
            "sequence_output": sequence_output,
        }

    def _compute_re_logits(
            self,
            sequence_output: torch.Tensor,
            entity_pairs: List[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        all_pair_representations = []
        for batch_idx, pairs in enumerate(entity_pairs):
            if len(pairs) == 0:
                continue
            for (start1, end1), (start2, end2) in pairs:
                entity1_repr = sequence_output[batch_idx, start1:end1].mean(dim=0)
                entity2_repr = sequence_output[batch_idx, start2:end2].mean(dim=0)
                distance = min(abs(start2 - end1), self.max_distance)
                if start2 < start1:
                    distance = -distance
                distance_idx = distance + self.max_distance
                distance_repr = self.distance_embedding(
                    torch.tensor(distance_idx, device=sequence_output.device)
                )
                pair_repr = torch.cat(
                    [entity1_repr, entity2_repr, distance_repr], dim=0
                )
                all_pair_representations.append(pair_repr)

        if len(all_pair_representations) == 0:
            return torch.zeros(
                0, self.re_classifier[-1].out_features, device=sequence_output.device
            )

        pair_reprs = torch.stack(all_pair_representations)
        return self.re_classifier(pair_reprs)

    def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            id2ner_label: Dict[int, str],
            id2re_label: Dict[int, str],
            ner_threshold: float = 0.5,
            re_threshold: float = 0.5,
    ) -> List[Dict]:
        self.eval()
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids, attention_mask=attention_mask
            )
            sequence_output = outputs.last_hidden_state

            ner_logits = self.ner_classifier(sequence_output)
            ner_probs = torch.softmax(ner_logits, dim=-1)
            ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()

            results = []
            for batch_idx in range(input_ids.size(0)):
                entities = self._extract_entities(
                    ner_preds[batch_idx],
                    id2ner_label,
                    attention_mask[batch_idx].cpu().numpy(),
                )

                # 计算实体置信度
                for ent in entities:
                    token_scores = []
                    for t in range(ent["start"], ent["end"]):
                        pred_label_id = int(ner_preds[batch_idx][t])
                        token_scores.append(
                            float(ner_probs[batch_idx, t, pred_label_id])
                        )
                    ent["score"] = (
                        round(sum(token_scores) / len(token_scores), 4)
                        if token_scores
                        else 0.5
                    )

                entities = [
                    e for e in entities if e.get("score", 1.0) >= ner_threshold
                ]
                for i, ent in enumerate(entities):
                    ent["id"] = f"entity_{i}"

                if entities:
                    ent_summary = ", ".join(
                        f"{e['label']}({e['score']:.2f})" for e in entities
                    )
                    logger.info(
                        f"[Predict-NER] 识别 {len(entities)} 个实体: {ent_summary}"
                    )
                else:
                    logger.info("[Predict-NER] 未识别到实体")

                # RE 预测
                relations = []
                if len(entities) >= 2:
                    entity_pairs = []
                    pair_indices = []
                    for i, ent1 in enumerate(entities):
                        for j, ent2 in enumerate(entities):
                            if i != j:
                                if ent1["label"] == ent2["label"]:
                                    continue
                                entity_pairs.append(
                                    [
                                        (ent1["start"], ent1["end"]),
                                        (ent2["start"], ent2["end"]),
                                    ]
                                )
                                pair_indices.append((i, j))

                    if entity_pairs:
                        re_logits = self._compute_re_logits(
                            sequence_output[batch_idx: batch_idx + 1],
                            [entity_pairs],
                        )
                        re_probs = torch.softmax(re_logits, dim=-1)
                        re_preds = torch.argmax(re_probs, dim=-1).cpu().numpy()

                        for (i, j), pred_idx, probs in zip(
                                pair_indices, re_preds, re_probs.cpu().numpy()
                        ):
                            ent1_label = entities[i]['label']
                            ent2_label = entities[j]['label']
                            pred_name = id2re_label.get(pred_idx, "?")
                            if pred_idx != 0:
                                confidence = float(probs[pred_idx])
                                logger.info(
                                    f"[Predict-RE] {ent1_label} -> {ent2_label}: "
                                    f"{pred_name}(conf={confidence:.4f}) "
                                    f"{'✓ 输出' if confidence >= re_threshold else '✗ 低于阈值'}"
                                )
                                if confidence >= re_threshold:
                                    relations.append(
                                        {
                                            "from_id": entities[i].get("id", i),
                                            "to_id": entities[j].get("id", j),
                                            "type": id2re_label[pred_idx],
                                            "confidence": confidence,
                                        }
                                    )

                results.append({"entities": entities, "relations": relations})
            return results

    def _extract_entities(
            self,
            ner_preds: List[int],
            id2ner_label: Dict[int, str],
            attention_mask: List[int],
    ) -> List[Dict]:
        entities = []
        current_entity = None

        for idx, pred_idx in enumerate(ner_preds):
            if attention_mask[idx] == 0:
                continue
            label = id2ner_label.get(pred_idx, "O")

            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "id": f"entity_{len(entities)}",
                    "start": idx,
                    "end": idx + 1,
                    "label": label[2:],
                }
            elif (
                    label.startswith("I-")
                    and current_entity
                    and current_entity["label"] == label[2:]
            ):
                current_entity["end"] = idx + 1
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        entities = self._merge_adjacent_entities(entities)
        entities = [e for e in entities if e["end"] - e["start"] >= 1]
        for i, ent in enumerate(entities):
            ent["id"] = f"entity_{i}"
        return entities

    def _merge_adjacent_entities(self, entities: List[Dict]) -> List[Dict]:
        if not entities:
            return entities
        entities = sorted(entities, key=lambda x: (x["start"], x["end"]))
        merged = []
        current = entities[0].copy()
        for ent in entities[1:]:
            if ent["label"] == current["label"] and ent["start"] <= current["end"] + 1:
                current["end"] = max(current["end"], ent["end"])
            else:
                merged.append(current)
                current = ent.copy()
        merged.append(current)
        return merged

    def save_pretrained(self, save_path: str):
        os.makedirs(save_path, exist_ok=True)
        torch.save(
            self.state_dict(), os.path.join(save_path, "pytorch_model.bin")
        )
        cfg = {
            "hidden_size": self.hidden_size,
            "num_ner_labels": self.ner_classifier[-1].out_features,
            "num_re_labels": self.re_classifier[-1].out_features,
        }
        with open(
                os.path.join(save_path, "model_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_pretrained(
            cls,
            model_path: str,
            num_ner_labels: int = None,
            num_re_labels: int = None,
    ):
        config_path = os.path.join(model_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                saved_cfg = json.load(f)
            num_ner_labels = num_ner_labels or saved_cfg.get("num_ner_labels", 1)
            num_re_labels = num_re_labels or saved_cfg.get("num_re_labels", 1)

        model = cls(model_path, num_ner_labels, num_re_labels)
        state_dict_path = os.path.join(model_path, "pytorch_model.bin")
        if os.path.exists(state_dict_path):
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model
