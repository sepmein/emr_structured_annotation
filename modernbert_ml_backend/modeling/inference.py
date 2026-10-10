"""Text decoding and prediction wrapper shared by training and service."""
import logging
from typing import Dict
from config import config
from modeling.network import ModernBERTForNERRE

logger = logging.getLogger(__name__)


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
