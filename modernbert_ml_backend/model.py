"""
ModernBERT NER+RE ML Backend
基于 ModernBERT 的命名实体识别与关系抽取联合模型
"""
import os
import json
import hashlib
import logging
import threading
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from transformers import AutoModel

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_sdk.label_interface.objects import PredictionValue

from config import config

logger = logging.getLogger(__name__)


# ==================== NER + RE 联合模型 ====================

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


# ==================== 预测辅助函数 ====================

def predict_text(
        model: ModernBERTForNERRE,
        tokenizer,
        text: str,
        ner_id2label: Dict[int, str],
        re_id2label: Dict[int, str],
        device: str,
        max_length: int = 8192,
        ner_threshold: float = 0.5,
        re_threshold: float = 0.5,
) -> Dict:
    """对单条文本进行 NER+RE 预测，返回字符级位置结果"""
    logger.debug(f"[Predict] 输入文本: {text[:80]}{'...' if len(text) > 80 else ''}")

    encoding = tokenizer(
        text,
        max_length=max_length,
        padding=False,  # 预测时不填充，只使用实际长度（避免浪费计算资源）
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    offset_mapping = encoding["offset_mapping"][0].cpu().numpy()
    logger.info(
        "[Predict] text_chars=%s, tokens=%s, max_length=%s, device=%s",
        len(text),
        input_ids.shape[1],
        max_length,
        device,
    )

    result = model.predict(
        input_ids,
        attention_mask,
        ner_id2label,
        re_id2label,
        ner_threshold,
        re_threshold,
    )[0]

    # token 索引转换为字符位置
    for ent in result.get("entities", []):
        ts = ent["start"]
        te = ent["end"]
        if ts < len(offset_mapping):
            ent["start"] = int(offset_mapping[ts][0])
        if te - 1 < len(offset_mapping):
            ent["end"] = int(offset_mapping[te - 1][1])

    entities = result.get("entities", [])

    # 根据字符距离过滤远距离关系
    relations_before_filter = result.get("relations", [])
    if config.re_max_distance > 0:
        ent_map = {ent.get("id", f"entity_{i}"): ent for i, ent in enumerate(entities)}
        filtered_relations = []
        for rel in relations_before_filter:
            from_ent = ent_map.get(rel.get("from_id"))
            to_ent = ent_map.get(rel.get("to_id"))
            if from_ent and to_ent:
                s1, e1 = from_ent["start"], from_ent["end"]
                s2, e2 = to_ent["start"], to_ent["end"]
                gap = max(s1, s2) - min(e1, e2)
                if gap <= config.re_max_distance:
                    filtered_relations.append(rel)
                else:
                    logger.debug(
                        f"[Distance-Filter] 过滤远距离关系: "
                        f"{from_ent['label']}({from_ent['start']}-{from_ent['end']}) -> "
                        f"{to_ent['label']}({to_ent['start']}-{to_ent['end']}), "
                        f"字符间距={gap} > {config.re_max_distance}"
                    )
            else:
                filtered_relations.append(rel)
        result["relations"] = filtered_relations
    logger.info(
        f"[Predict] 关系过滤: {len(result.get('relations', []))}/{len(relations_before_filter)} "
        f"(re_max_distance={config.re_max_distance})"
    )

    relations = result.get("relations", [])
    logger.debug(
        f"[Predict] 提取实体 {len(entities)} 个: "
        f"{[(e['label'], text[e['start']:e['end']], f'{e['score']:.3f}') for e in entities[:5]]}"
    )
    logger.debug(
        f"[Predict] 提取关系 {len(relations)} 个: "
        f"{[(r['type'], r.get('confidence', 0)) for r in relations[:5]]}"
    )

    return result


class NERREPredictor:
    """预测器封装"""

    def __init__(
            self,
            model: ModernBERTForNERRE,
            tokenizer,
            ner_id2label: Dict[int, str],
            re_id2label: Dict[int, str],
            device: str,
            max_length: int = 8192,
            ner_threshold: float = 0.5,
            re_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.ner_id2label = ner_id2label
        self.re_id2label = re_id2label
        self.device = device
        self.max_length = max_length
        self.ner_threshold = ner_threshold
        self.re_threshold = re_threshold

    def predict(self, text: str) -> Dict:
        return predict_text(
            self.model,
            self.tokenizer,
            text,
            self.ner_id2label,
            self.re_id2label,
            self.device,
            self.max_length,
            self.ner_threshold,
            self.re_threshold,
        )


# ==================== Label Studio ML Backend ====================

class ModernBERTModel(LabelStudioMLBase):
    """
    ModernBERT NER+RE ML Backend
    支持从 Label Studio 标签配置动态获取实体和关系标签
    """
    _MODEL_CACHE = OrderedDict()
    _MODEL_CACHE_LOCK = threading.Lock()

    def setup(self):
        self._sync_labels_from_label_studio()
        self.text_field_candidates = self._resolve_text_field_candidates()
        self.label_schema_id = self._build_label_schema_id()
        config.update_model_paths(model_scope=self.label_schema_id)
        model_version = f"ModernBERT-{config.base_model_name}-NERRE-v1.0"
        if self.label_schema_id:
            model_version = f"{model_version}-{self.label_schema_id}"
        self.set("model_version", model_version)

        # 路径配置
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = config.model_path
        self.output_dir = config.output_dir
        self.best_model_path = config.best_model_path

        # 设备
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # 使用 config 中的共享 tokenizer（避免重复加载）
        self.tokenizer = config.tokenizer

        # 使用从 Label Studio 动态同步后的标签映射
        self.ner_labels = config.ner_labels
        self.re_labels = config.re_labels
        self.ner_label2id = config.ner_label2id
        self.ner_id2label = config.ner_id2label
        self.num_ner_labels = config.num_ner_labels
        self.re_label2id = config.re_label2id
        self.re_id2label = config.re_id2label
        self.num_re_labels = config.num_re_labels

        # 加载模型
        self._model = None
        self._load_model()

        logger.info("ModernBERTModel 初始化完成")
        logger.info(f"  基础模型: {config.base_model_name}")
        logger.info(f"  标签 schema: {self.label_schema_id or 'default'}")
        logger.info(f"  NER 标签数: {self.num_ner_labels} ({self.ner_labels[:5]}{'...' if len(self.ner_labels) > 5 else ''})")
        logger.info(f"  RE 标签数: {self.num_re_labels} ({self.re_labels[:5]}{'...' if len(self.re_labels) > 5 else ''})")
        logger.info(f"  文本字段候选: {self.text_field_candidates}")
        logger.info(f"  设备: {self.device}")
        has_pytorch = os.path.exists(os.path.join(self.best_model_path, "pytorch_model.bin"))
        has_lora = os.path.exists(os.path.join(self.best_model_path, "adapter_config.json"))
        has_heads = os.path.exists(os.path.join(self.best_model_path, "classification_heads.pt"))
        model_status = "全参数模型" if has_pytorch else ("LoRA 模型" if (has_lora and has_heads) else "无")
        logger.info(f"  微调模型路径: {self.best_model_path}")
        logger.info(f"  微调模型状态: {model_status} (pytorch={has_pytorch}, lora={has_lora}, heads={has_heads})")

    def _sync_labels_from_label_studio(self):
        ner_labels, re_labels = self._extract_labels_from_label_interface()
        if not ner_labels or not re_labels:
            xml_ner_labels, xml_re_labels = self._extract_labels_from_label_config_xml()
            if not ner_labels:
                ner_labels = xml_ner_labels
            if not re_labels:
                re_labels = xml_re_labels

        if not ner_labels:
            logger.warning("未能从 Label Studio label_config 中获取 NER 标签，已清空动态标签配置")
            config.update_labels([], [])
            return

        config.update_labels(ner_labels, re_labels)
        logger.info(
            f"已从 Label Studio 动态同步标签: NER={len(config.ner_labels)}, RE={len(config.re_labels)}"
        )

    def _extract_labels_from_label_interface(self) -> Tuple[List[str], List[str]]:
        label_interface = getattr(self, "label_interface", None)
        if label_interface is None:
            return [], []

        ner_labels = []
        re_labels = []
        for control in getattr(label_interface, "controls", []) or []:
            control_tag = getattr(control, "tag", "")
            control_labels = self._normalize_control_labels(getattr(control, "labels", []))
            if control_tag == "Labels":
                ner_labels.extend(control_labels)
            elif control_tag in ("Relation", "Relations"):
                re_labels.extend(control_labels)
        return self._unique_labels(ner_labels), self._unique_labels(re_labels)

    def _extract_labels_from_label_config_xml(self) -> Tuple[List[str], List[str]]:
        label_config = self.label_config
        if not label_config:
            return [], []

        try:
            root = ET.fromstring(label_config)
        except ET.ParseError as exc:
            logger.warning(f"解析 Label Studio label_config 失败: {exc}")
            return [], []

        ner_labels = []
        re_labels = []
        for elem in root.iter():
            tag = elem.tag.rsplit("}", 1)[-1]
            if tag == "Labels":
                for child in elem:
                    child_tag = child.tag.rsplit("}", 1)[-1]
                    if child_tag == "Label" and child.attrib.get("value"):
                        ner_labels.append(child.attrib["value"])
            elif tag == "Relation" and elem.attrib.get("value"):
                re_labels.append(elem.attrib["value"])
        return self._unique_labels(ner_labels), self._unique_labels(re_labels)

    @staticmethod
    def _normalize_text_field_name(value) -> str:
        field_name = str(value or "").strip()
        if field_name.startswith("$"):
            if field_name.count("$") != 1 or any(
                    character.isspace() for character in field_name
            ):
                return ""
            return field_name[1:]
        if "$" in field_name:
            return ""
        return field_name

    def _resolve_text_field_candidates(self) -> List[str]:
        """Resolve task data keys from the Labels -> Text binding."""
        candidates = []
        label_interface = getattr(self, "label_interface", None)
        if label_interface is not None:
            try:
                for control in getattr(label_interface, "controls", []) or []:
                    if getattr(control, "tag", "") != "Labels":
                        continue
                    for object_tag in getattr(control, "objects", []) or []:
                        if getattr(object_tag, "tag", "") == "Text":
                            candidates.append(
                                self._normalize_text_field_name(
                                    getattr(object_tag, "value", "")
                                )
                            )
                            break
                    if any(candidates):
                        break
            except Exception as exc:
                logger.debug("从 Label Interface 解析文本字段失败: %s", exc)

        if not any(candidates):
            label_config = getattr(self, "label_config", None)
            if label_config:
                try:
                    root = ET.fromstring(label_config)
                    labels_to_names = []
                    text_fields = {}
                    first_text_field = ""

                    for elem in root.iter():
                        tag = elem.tag.rsplit("}", 1)[-1]
                        if tag == "Labels":
                            labels_to_names.extend(
                                name.strip()
                                for name in elem.attrib.get("toName", "").split(",")
                                if name.strip()
                            )
                        elif tag == "Text":
                            field_name = self._normalize_text_field_name(
                                elem.attrib.get("value")
                            )
                            if not field_name:
                                continue
                            text_name = elem.attrib.get("name", "")
                            if text_name:
                                text_fields[text_name] = field_name
                            if not first_text_field:
                                first_text_field = field_name

                    for to_name in labels_to_names:
                        if to_name in text_fields:
                            candidates.append(text_fields[to_name])
                            break
                    if not any(candidates) and first_text_field:
                        candidates.append(first_text_field)
                except ET.ParseError as exc:
                    logger.debug("从 label_config XML 解析文本字段失败: %s", exc)

        candidates.extend(("text", "label_studio_text"))
        return list(dict.fromkeys(field for field in candidates if field))

    def _extract_task_text(self, task_data: Dict) -> str:
        if not isinstance(task_data, dict):
            try:
                task_data = dict(task_data)
            except (TypeError, ValueError):
                return ""

        candidates = getattr(self, "text_field_candidates", None)
        if not candidates:
            candidates = self._resolve_text_field_candidates()

        for text_source in candidates:
            text = task_data.get(text_source)
            if text:
                text_for_log = str(text)
                logger.info(
                    "[TextParser] source=%s, chars=%s, first_50=%r, last_50=%r",
                    text_source,
                    len(text_for_log),
                    text_for_log[:50],
                    text_for_log[-50:],
                )
                return text
        logger.warning(
            "[TextParser] 未解析到文本，候选字段=%s，任务字段=%s",
            candidates,
            list(task_data.keys()),
        )
        return ""

    @staticmethod
    def _normalize_control_labels(labels) -> List[str]:
        if labels is None:
            return []
        if isinstance(labels, dict):
            values = labels.keys()
        else:
            values = labels

        normalized = []
        for label in values:
            value = getattr(label, "value", label)
            if value:
                normalized.append(str(value))
        return normalized

    @staticmethod
    def _unique_labels(labels: List[str]) -> List[str]:
        return list(dict.fromkeys(label for label in labels if label and label != "无关系"))

    def _build_label_schema_id(self) -> Optional[str]:
        if not config.ner_labels:
            return None
        schema_payload = {
            "ner_labels": config.ner_labels,
            "re_labels": config.re_labels,
        }
        schema_json = json.dumps(
            schema_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return f"schema_{hashlib.sha256(schema_json.encode('utf-8')).hexdigest()[:12]}"

    def _model_artifact_signature(self) -> Tuple[Tuple[str, int, int], ...]:
        paths = [
            os.path.join(self.best_model_path, "pytorch_model.bin"),
            os.path.join(self.best_model_path, "adapter_config.json"),
            os.path.join(self.best_model_path, "adapter_model.safetensors"),
            os.path.join(self.best_model_path, "classification_heads.pt"),
        ]
        signature = []
        for path in paths:
            if os.path.exists(path):
                stat = os.stat(path)
                signature.append((os.path.basename(path), stat.st_size, stat.st_mtime_ns))
        return tuple(signature)

    def _model_cache_key(self) -> Tuple:
        return (
            config.base_model_name,
            os.path.abspath(self.model_path),
            os.path.abspath(self.best_model_path),
            tuple(self.ner_labels),
            tuple(self.re_labels),
            self.num_ner_labels,
            self.num_re_labels,
            str(self.device),
            self._model_artifact_signature(),
        )

    def _load_model(self):
        """加载模型（支持全参数模型和 LoRA 模型）"""
        cache_key = self._model_cache_key()
        with self._MODEL_CACHE_LOCK:
            cached_model = self._MODEL_CACHE.get(cache_key)
            if cached_model is not None:
                self._MODEL_CACHE.move_to_end(cache_key)
                self._model = cached_model
                self._model.eval()
                logger.info(
                    f"复用已缓存模型: {config.base_model_name} "
                    f"(cache={len(self._MODEL_CACHE)}/{config.model_cache_max_size})"
                )
                return

        pytorch_path = os.path.join(self.best_model_path, "pytorch_model.bin")
        lora_config_path = os.path.join(self.best_model_path, "adapter_config.json")
        heads_path = os.path.join(self.best_model_path, "classification_heads.pt")

        has_full_model = os.path.exists(pytorch_path)
        has_lora_model = os.path.exists(lora_config_path) and os.path.exists(heads_path)
        if not self.label_schema_id:
            has_full_model = False
            has_lora_model = False
            logger.info("未检测到标签 schema，跳过默认微调模型，仅加载基础模型")

        if has_full_model:
            logger.info(f"加载全参数微调模型: {self.best_model_path}")
            self._model = ModernBERTForNERRE.from_pretrained(
                self.best_model_path,
                num_ner_labels=self.num_ner_labels,
                num_re_labels=self.num_re_labels,
            )
        elif has_lora_model:
            logger.info(f"加载 LoRA 微调模型: {self.best_model_path}")
            self._model = ModernBERTForNERRE(
                model_path=self.model_path,
                num_ner_labels=self.num_ner_labels,
                num_re_labels=self.num_re_labels,
            )
            from peft import PeftModel
            self._model.bert = PeftModel.from_pretrained(
                self._model.bert, self.best_model_path
            )
            heads_state = torch.load(heads_path, map_location="cpu")
            self._model.ner_classifier.load_state_dict(
                heads_state["ner_classifier"]
            )
            self._model.re_classifier.load_state_dict(
                heads_state["re_classifier"]
            )
            self._model.distance_embedding.load_state_dict(
                heads_state["distance_embedding"]
            )
            logger.info("LoRA adapter 和分类头加载完成")
        else:
            logger.info(f"加载基础模型: {self.model_path}")
            self._model = ModernBERTForNERRE(
                model_path=self.model_path,
                num_ner_labels=self.num_ner_labels,
                num_re_labels=self.num_re_labels,
            )
        self._model.to(self.device)
        self._model.eval()
        with self._MODEL_CACHE_LOCK:
            if config.model_cache_max_size > 0:
                self._MODEL_CACHE[cache_key] = self._model
                self._MODEL_CACHE.move_to_end(cache_key)
                while len(self._MODEL_CACHE) > config.model_cache_max_size:
                    evicted_key, _ = self._MODEL_CACHE.popitem(last=False)
                    logger.info(f"模型缓存已淘汰最久未使用项: {evicted_key[2]}")
                logger.info(
                    f"模型已写入缓存: {config.base_model_name} "
                    f"(cache={len(self._MODEL_CACHE)}/{config.model_cache_max_size})"
                )
            else:
                self._MODEL_CACHE.clear()
                logger.info("模型缓存已禁用，当前模型不会写入缓存")

    def predict(self, tasks, context=None, **kwargs):
        """
        对输入任务进行 NER 和 RE 预测
        返回 Label Studio 格式的预测结果
        """
        logger.info(f"[MLBackend] 收到预测请求，任务数: {len(tasks)}")

        # 获取 Labels 标签名称映射
        label_to_from_name = {}
        for control in self.label_interface.controls:
            if control.tag == "Labels":
                for label in control.labels:
                    label_to_from_name[label] = control.name

        # 获取 Text 控件名称；文本值统一通过动态字段候选读取。
        try:
            _, to_name, _ = self.label_interface.get_first_tag_occurence(
                "Labels", "Text"
            )
        except Exception:
            to_name = "text"

        predictions = []
        for task in tasks:
            text = self._extract_task_text(task.get("data", {}))
            if not text:
                continue

            result = predict_text(
                self._model,
                self.tokenizer,
                text,
                self.ner_id2label,
                self.re_id2label,
                self.device,
                max_length=config.max_length,
                ner_threshold=config.ner_threshold,
                re_threshold=config.re_threshold,
            )

            ls_result = []
            entity_ls_id_map = {}

            # 添加实体
            for idx, ent in enumerate(result.get("entities", [])):
                ent_id = f"ent_{idx}"
                entity_ls_id_map[ent.get("id", str(idx))] = ent_id
                ent_text = text[ent["start"]: ent["end"]]

                ls_result.append(
                    {
                        "id": ent_id,
                        "from_name": label_to_from_name.get(
                            ent["label"], "label"
                        ),
                        "to_name": to_name,
                        "type": "labels",
                        "value": {
                            "start": ent["start"],
                            "end": ent["end"],
                            "text": ent_text,
                            "labels": [ent["label"]],
                        },
                        "score": round(ent.get("score", 0.5), 4),
                    }
                )
                logger.info(
                    f"[MLBackend] 实体 #{idx}: [{ent['label']}] \"{ent_text}\" "
                    f"(pos={ent['start']}-{ent['end']}, score={ent.get('score', 0):.2f})"
                )

            # 添加关系
            for rel in result.get("relations", []):
                from_id_raw = rel.get("from_id", "")
                to_id_raw = rel.get("to_id", "")

                from_id = entity_ls_id_map.get(from_id_raw)
                to_id = entity_ls_id_map.get(to_id_raw)

                if from_id and to_id:
                    ls_result.append(
                        {
                            "type": "relation",
                            "from_id": from_id,
                            "to_id": to_id,
                            "labels": [rel["type"]],
                            "score": round(rel.get("confidence", 0.5), 4),
                        }
                    )
                    ent_list = result.get("entities", [])
                    from_ent = next((e for e in ent_list if e.get("id") == from_id_raw), None)
                    to_ent = next((e for e in ent_list if e.get("id") == to_id_raw), None)
                    from_label = from_ent["label"] if from_ent else "?"
                    to_label = to_ent["label"] if to_ent else "?"
                    logger.info(
                        f"[MLBackend] 关系: {from_label} -> {to_label} "
                        f"[{rel['type']}] (conf={rel.get('confidence', 0):.4f})"
                    )

            score = (
                min([p["score"] for p in ls_result])
                if ls_result
                else 2.0
            )
            total_ents = sum(1 for r in ls_result if r.get("type") == "labels")
            total_rels = sum(1 for r in ls_result if r.get("type") == "relation")
            logger.info(
                f"[MLBackend] 任务结果: 实体={total_ents} 个, 关系={total_rels} 个, "
                f"score={score:.4f}"
            )
            predictions.append(
                PredictionValue(
                    result=ls_result,
                    score=score,
                    model_version=self.get("model_version"),
                )
            )

        logger.info(f"[MLBackend] 预测完成，返回 {len(predictions)} 个结果")
        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        """训练接口"""
        if event not in (
                "START_TRAINING"
        ):
            logger.info(f"跳过训练: 事件 {event} 不支持")
            return

        try:
            training_data = self._load_training_data()
        except Exception as e:
            logger.error(f"获取训练数据失败: {e}")
            return

        if not training_data:
            logger.warning("没有可用的训练数据")
            return

        logger.info(f"准备训练，共 {len(training_data)} 条样本")

        if len(training_data) <= 20:
            logger.warning(
                f"数据量极少 ({len(training_data)} 条)，建议至少 50 条以上"
            )
        elif len(training_data) <= 200:
            logger.info(f"少样本场景 ({len(training_data)} 条)，自动调整超参数")

        try:
            from train_ner_re import train_from_data

            model, best_f1 = train_from_data(
                training_data=training_data,
                train_ratio=0.8,
            )
            logger.info(f"训练完成，最佳综合 F1: {best_f1:.4f}")

            self._model = None
            with self._MODEL_CACHE_LOCK:
                self._MODEL_CACHE.clear()
            self._load_model()
            logger.info("模型已更新并重新加载")

        except Exception as e:
            logger.exception(f"训练过程出错: {e}")

    def _load_training_data(self) -> List[Dict]:
        """从 Label Studio 下载已标注数据并解析为训练格式"""
        ls_host = self.get_label_studio_url() or config.label_studio_url
        ls_api_key = self.get_label_studio_access_token()

        if not ls_api_key:
            logger.warning(
                "项目 %s 尚未从 Label Studio /setup 接收到动态 access_token，"
                "请在 Label Studio 中重新保存或刷新 ML Backend 连接",
                self.project_id,
            )
            return []

        try:
            ls = self.get_label_studio_client()
            if not ls:
                logger.warning(
                    "无法为项目 %s 创建 Label Studio SDK 客户端，地址=%s",
                    self.project_id,
                    ls_host,
                )
                return []
            all_tasks = list(
                ls.tasks.list(
                    project=self.project_id, only_annotated=True, fields="all"
                )
            )

            tasks = [
                t
                for t in all_tasks
                if any(
                    not getattr(ann, "was_cancelled", False)
                    for ann in (getattr(t, "annotations", None) or [])
                )
            ]
            logger.info(f"从 Label Studio 获取 {len(tasks)} 条已标注任务")

        except Exception as e:
            logger.warning(f"通过 SDK 获取数据失败: {e}")
            return []

        formatted_data = []
        for task in tasks:
            try:
                data = self._parse_task_to_training_format(task)
                if data and data.get("entities"):
                    formatted_data.append(data)
            except Exception as e:
                logger.warning(f"解析任务失败: {e}")
                continue

        logger.info(f"解析成功 {len(formatted_data)} 条训练样本")
        return formatted_data

    def _parse_task_to_training_format(self, task) -> Dict:
        """将 Label Studio task 解析为 {text, entities, relations} 格式"""
        task_data = task.data if hasattr(task, "data") else task.get("data", {})
        if hasattr(task_data, "__iter__") and not isinstance(task_data, dict):
            task_data = dict(task_data)

        text = self._extract_task_text(task_data)
        if not text:
            return {}

        annotations = (
            task.annotations if hasattr(task, "annotations") else task.get("annotations", [])
        )
        result = []
        for ann in annotations:
            if getattr(ann, "was_cancelled", False):
                continue
            result = ann.result if hasattr(ann, "result") else ann.get("result", [])
            if result:
                break

        if not result:
            return {}

        entities = []
        relations = []
        entity_id_map = {}

        for ann in result:
            if ann.get("type") == "labels":
                value = ann.get("value", {})
                labels = value.get("labels", [])
                if not labels:
                    continue

                ent_label = labels[0]
                ent = {
                    "id": ann.get("id", f"ent_{len(entities)}"),
                    "start": value.get("start", 0),
                    "end": value.get("end", 0),
                    "text": value.get("text", ""),
                    "label": ent_label,
                }
                if ent_label not in config.ner_label2id and f"B-{ent_label}" not in config.ner_label2id:
                    logger.warning(
                        f"Label Studio 实体标签 '{ent_label}' 不在 config.ner_labels 中"
                    )
                entity_id_map[ent["id"]] = ent
                entities.append(ent)

        for ann in result:
            if ann.get("type") == "relation":
                from_id = ann.get("from_id")
                to_id = ann.get("to_id")
                labels = ann.get("labels", [])

                if from_id and to_id and labels:
                    rel_type = labels[0]
                    from_ent = entity_id_map.get(from_id)
                    to_ent = entity_id_map.get(to_id)
                    if from_ent and to_ent:
                        if rel_type not in config.re_label2id:
                            logger.warning(
                                f"Label Studio 关系标签 '{rel_type}' 不在 config.re_labels 中"
                            )
                        relations.append({
                            "from_id": from_id,
                            "to_id": to_id,
                            "type": rel_type,
                        })

        return {
            "text": text,
            "entities": entities,
            "relations": relations,
            "has_relations": len(relations) > 0,
        }
