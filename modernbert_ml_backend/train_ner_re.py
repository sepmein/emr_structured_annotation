"""
NER + RE 联合训练脚本
支持 LoRA 微调、断点续训、dynamic padding 加速
"""
import os
import json
import logging
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter
import math

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm

from config import config
from model import ModernBERTForNERRE

logger = logging.getLogger(__name__)


# ==================== 数据集 ====================

class NERREDataset(Dataset):
    """NER+RE 数据集（不在此处 padding，由 collate_fn 动态填充）"""

    def __init__(
            self,
            data: List[Dict],
            tokenizer,
            max_length: int,
            ner_label2id: Dict,
            re_label2id: Dict,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ner_label2id = ner_label2id
        self.re_label2id = re_label2id
        self.processed_data = self._preprocess_data()

    def _preprocess_data(self) -> List[Dict]:
        processed = []
        for sample in self.data:
            text = sample["text"]
            entities = sample.get("entities", [])
            relations = sample.get("relations", [])

            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

            ner_labels = self._generate_ner_labels(entities, offset_mapping)
            re_labels, re_entity_pairs = self._generate_re_labels(
                entities, relations, offset_mapping
            )

            processed.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "ner_labels": ner_labels,
                "re_labels": re_labels if re_labels else None,
                "re_entity_pairs": re_entity_pairs if re_entity_pairs else None,
            })
        return processed

    def _generate_ner_labels(
            self, entities: List[Dict], offset_mapping: List[List[int]]
    ) -> List[int]:
        labels = [0] * len(offset_mapping)
        for entity in entities:
            ent_start, ent_end = entity["start"], entity["end"]
            ent_label = entity["label"]
            token_positions = []
            for token_idx, (ts, te) in enumerate(offset_mapping):
                if ts == 0 and te == 0:
                    continue                # 部分重叠匹配：token 与实体有交集即可
                # 修复原代码 ts >= ent_start and te <= ent_end 导致边界 token 丢失
                if ts < ent_end and te > ent_start:
                    token_positions.append(token_idx)
            if token_positions:
                b_label = f"B-{ent_label}"
                if b_label in self.ner_label2id:
                    labels[token_positions[0]] = self.ner_label2id[b_label]
                else:
                    logger.warning(f"NER 标签 '{b_label}' 不在 label2id 中，实体将被忽略")
                i_label = f"I-{ent_label}"
                if i_label in self.ner_label2id:
                    for pos in token_positions[1:]:
                        labels[pos] = self.ner_label2id[i_label]
        return labels

    def _generate_re_labels(
            self,
            entities: List[Dict],
            relations: List[Dict],
            offset_mapping: List[List[int]],
    ) -> Tuple[Optional[List[int]], Optional[List[Tuple[Tuple[int, int], Tuple[int, int]]]]]:
        entity_token_map = {}
        for entity in entities:
            ent_start, ent_end = entity["start"], entity["end"]
            ent_id = entity.get("id", "")
            token_positions = []
            for token_idx, (ts, te) in enumerate(offset_mapping):
                if ts == 0 and te == 0:
                    continue
                # 与 _generate_ner_labels 保持一致：部分重叠匹配
                if ts < ent_end and te > ent_start:
                    token_positions.append(token_idx)
            if token_positions and ent_id:
                entity_token_map[ent_id] = (
                    min(token_positions),
                    max(token_positions) + 1,
                )

        if len(entity_token_map) < 2:
            return None, None

        re_labels = []
        entity_pairs = []
        entity_ids = list(entity_token_map.keys())

        relation_map = {}
        for rel in relations or []:
            key = (rel.get("from_id"), rel.get("to_id"))
            relation_map[key] = rel.get("type", "无关系")

        sampled_pairs = set()

        def add_pair(from_id: str, to_id: str, label_id: int):
            key = (from_id, to_id)
            if key in sampled_pairs:
                return
            if from_id not in entity_token_map or to_id not in entity_token_map:
                return
            entity_pairs.append((entity_token_map[from_id], entity_token_map[to_id]))
            re_labels.append(label_id)
            sampled_pairs.add(key)

        # 策略：正样本 + 充足负采样（包含同类型实体对）
        # 1. 先添加所有正样本（有标注关系的实体对）
        positive_pairs = set()
        for (from_id, to_id), rel_type in relation_map.items():
            add_pair(from_id, to_id, self.re_label2id.get(rel_type, 0))
            if from_id in entity_token_map and to_id in entity_token_map:
                positive_pairs.add((from_id, to_id))

        # 2. 负采样：对每个实体，随机选多个其他实体作为负样本
        # 包含同类型实体对（模型也需要学会判断同类型实体间无关系）
        neg_per_entity = 3
        for from_id in entity_ids:
            candidates = [
                eid for eid in entity_ids
                if eid != from_id and (from_id, eid) not in positive_pairs
            ]
            if candidates:
                sampled = random.sample(candidates, min(neg_per_entity, len(candidates)))
                for neg_to_id in sampled:
                    add_pair(from_id, neg_to_id, 0)

        # 3. 如果前面没有生成任何实体对，兜底添加少量负样本
        if not entity_pairs and len(entity_ids) >= 2:
            all_pairs = []
            for from_id in entity_ids:
                for to_id in entity_ids:
                    if from_id != to_id:
                        all_pairs.append((from_id, to_id))
            sampled = random.sample(all_pairs, min(5, len(all_pairs)))
            for from_id, to_id in sampled:
                add_pair(from_id, to_id, 0)

        if not entity_pairs:
            return None, None

        return re_labels, entity_pairs

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """Dynamic padding：只填充到批次中的最大长度"""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    ner_labels = []
    re_labels = []
    re_entity_pairs = []

    from config import config
    pad_token_id = config.tokenizer.pad_token_id or 0

    for item in batch:
        seq_len = len(item["input_ids"])
        pad_len = max_len - seq_len

        input_ids.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        ner_labels.append(
            torch.cat([torch.tensor(item["ner_labels"], dtype=torch.long),
                       torch.full((pad_len,), -100, dtype=torch.long)])
        )
        re_labels.append(item.get("re_labels"))
        re_entity_pairs.append(item.get("re_entity_pairs"))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "ner_labels": torch.stack(ner_labels),
        "re_labels": re_labels,
        "re_entity_pairs": re_entity_pairs,
    }


# ==================== 标签权重计算 ====================

def compute_label_weights(data: List[Dict], method: str = "sqrt") -> torch.Tensor:
    """计算 NER 标签权重，缓解类别不平衡"""
    label_counts = Counter()
    for sample in data:
        for ent in sample.get("entities", []):
            label = ent.get("label", "O")
            label_counts[f"B-{label}"] += 1
            label_counts[f"I-{label}"] += 1

    weights = {"O": 0.15}
    entity_labels = {k: v for k, v in label_counts.items() if k != "O"}
    if entity_labels:
        avg_count = sum(entity_labels.values()) / len(entity_labels)
        for label, count in entity_labels.items():
            freq_ratio = count / avg_count if avg_count > 0 else 1.0
            if method == "sqrt":
                weight = 1.0 / math.sqrt(freq_ratio)
            elif method == "log":
                weight = 1.0 / math.log(1 + freq_ratio)
            else:
                weight = 1.0 / freq_ratio
            weights[label] = max(0.5, min(5.0, weight))

    weight_list = []
    for i in range(config.num_ner_labels):
        label_name = config.ner_id2label.get(i, "O")
        weight_list.append(weights.get(label_name, 1.0))

    return torch.tensor(weight_list, dtype=torch.float32)


# ==================== 训练与评估 ====================

def train_epoch(
        model: ModernBERTForNERRE,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        epoch: int,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    use_amp = scaler is not None
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ner_labels = batch["ner_labels"].to(device)

        re_labels_list = batch["re_labels"]
        re_entity_pairs_list = batch["re_entity_pairs"]

        batch_entity_pairs = []
        batch_re_labels = []
        for rel_labels, rel_pairs in zip(re_labels_list, re_entity_pairs_list):
            if rel_labels is not None and len(rel_labels) > 0:
                batch_entity_pairs.append(rel_pairs)
                batch_re_labels.extend(rel_labels)
            else:
                batch_entity_pairs.append([])

        with torch.cuda.amp.autocast(enabled=use_amp):
            if batch_re_labels:
                re_labels_tensor = torch.tensor(batch_re_labels, dtype=torch.long).to(device)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                    entity_pairs=batch_entity_pairs,
                    re_labels=re_labels_tensor,
                )
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ner_labels=ner_labels,
                )

            loss = outputs["loss"]

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        num_batches += 1
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "avg": f"{total_loss / num_batches:.4f}",
        })
    return {"loss": total_loss / num_batches if num_batches > 0 else 0}


def evaluate(
        model: ModernBERTForNERRE,
        dataloader: DataLoader,
        device: torch.device,
        use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    all_ner_preds, all_ner_labels = [], []
    all_re_preds, all_re_labels = [], []
    total_loss, num_batches = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            ner_labels = batch["ner_labels"].to(device)

            re_labels_list = batch["re_labels"]
            re_entity_pairs_list = batch["re_entity_pairs"]

            batch_entity_pairs = []
            batch_re_labels = []
            for rel_labels, rel_pairs in zip(re_labels_list, re_entity_pairs_list):
                if rel_labels is not None and len(rel_labels) > 0:
                    batch_entity_pairs.append(rel_pairs)
                    batch_re_labels.extend(rel_labels)
                else:
                    batch_entity_pairs.append([])

            with torch.cuda.amp.autocast(enabled=use_amp):
                if batch_re_labels:
                    re_labels_tensor = torch.tensor(batch_re_labels, dtype=torch.long).to(device)
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ner_labels=ner_labels,
                        entity_pairs=batch_entity_pairs,
                        re_labels=re_labels_tensor,
                    )
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        ner_labels=ner_labels,
                    )

            loss = outputs["loss"]
            total_loss += loss.item()
            num_batches += 1

            # NER 指标
            ner_logits = outputs["ner_logits"]
            ner_preds = torch.argmax(ner_logits, dim=-1).cpu().numpy()
            ner_labels_cpu = ner_labels.cpu().numpy()
            for i in range(len(ner_preds)):
                mask = batch["attention_mask"][i].cpu().numpy()
                for j in range(len(ner_preds[i])):
                    if mask[j] == 1 and ner_labels_cpu[i][j] != -100:
                        all_ner_preds.append(ner_preds[i][j])
                        all_ner_labels.append(ner_labels_cpu[i][j])

            # RE 指标（基于标注实体对评估，避免 NER 误差干扰 RE 指标）
            sequence_output = outputs["sequence_output"]

            batch_size = input_ids.size(0)
            for i in range(batch_size):
                gt_re_labels_i = re_labels_list[i]
                gt_re_pairs_i = re_entity_pairs_list[i]

                # 没有标注关系时跳过
                if gt_re_labels_i is None or len(gt_re_labels_i) == 0:
                    continue

                # 用标注实体对做 RE 预测（而非预测实体），确保评估的公平性
                re_logits = model._compute_re_logits(
                    sequence_output[i:i + 1], [gt_re_pairs_i]
                )
                re_preds_i = torch.argmax(re_logits, dim=-1).cpu().numpy().tolist()

                all_re_preds.extend(re_preds_i)
                all_re_labels.extend(gt_re_labels_i)

    metrics = {"loss": total_loss / num_batches if num_batches > 0 else 0}

    if all_ner_labels:
        p, r, f1, _ = precision_recall_fscore_support(
            all_ner_labels, all_ner_preds, average="macro", zero_division=0
        )
        metrics.update({"ner_precision": p, "ner_recall": r, "ner_f1": f1})

    # RE 指标
    if all_re_labels:
        re_p, re_r, re_f1, _ = precision_recall_fscore_support(
            all_re_labels, all_re_preds, average="macro", zero_division=0
        )
        # 额外输出正样本（有关系的）指标
        re_preds_pos = [p for p, l in zip(all_re_preds, all_re_labels) if l != 0]
        re_labels_pos = [l for l in all_re_labels if l != 0]
        if re_labels_pos:
            pos_p, pos_r, pos_f1, _ = precision_recall_fscore_support(
                re_labels_pos, re_preds_pos, average="macro", zero_division=0
            )
        else:
            pos_p, pos_r, pos_f1 = 0, 0, 0
        # 负样本准确率
        re_preds_neg = [p for p, l in zip(all_re_preds, all_re_labels) if l == 0]
        re_labels_neg = [l for l in all_re_labels if l == 0]
        neg_acc = sum(1 for p, l in zip(re_preds_neg, re_labels_neg) if p == l) / len(re_labels_neg) if re_labels_neg else 1.0

        n_pos = len(re_labels_pos)
        n_neg = len(re_labels_neg)
        metrics.update({
            "re_precision": re_p,
            "re_recall": re_r,
            "re_f1": re_f1,
            "re_pos_f1": pos_f1,
            "re_neg_acc": neg_acc,
            "re_pos_count": n_pos,
            "re_neg_count": n_neg,
        })
        logger.info(
            f"RE 详细: 总F1={re_f1:.4f}, 正样本F1={pos_f1:.4f}({n_pos}个), "
            f"负样本acc={neg_acc:.4f}({n_neg}个)"
        )

    return metrics


# ==================== 模型保存/加载 ====================

def _save_model(model: ModernBERTForNERRE, save_path: str, is_lora: bool = False,
               best_f1: float = 0.0, optimizer=None, scheduler=None,
               epoch: int = 0, epochs_trained: int = 0):
    os.makedirs(save_path, exist_ok=True)
    if is_lora:
        from peft import PeftModel
        if isinstance(model.bert, PeftModel):
            model.bert.save_pretrained(save_path)
        torch.save({
            "ner_classifier": model.ner_classifier.state_dict(),
            "re_classifier": model.re_classifier.state_dict(),
            "distance_embedding": model.distance_embedding.state_dict(),
            "num_ner_labels": config.num_ner_labels,
            "num_re_labels": config.num_re_labels,
        }, os.path.join(save_path, "classification_heads.pt"))
    else:
        model.save_pretrained(save_path)
    checkpoint = {
        "best_f1": best_f1,
        "num_ner_labels": config.num_ner_labels,
        "num_re_labels": config.num_re_labels,
        "epoch": epoch,
        "epochs_trained": epochs_trained,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))


def _load_model_if_exists(
        model_path: str,
        num_ner_labels: int,
        num_re_labels: int,
        device: torch.device,
) -> Optional[ModernBERTForNERRE]:
    """加载已有模型（支持全参数和 LoRA）"""
    pytorch_path = os.path.join(model_path, "pytorch_model.bin")
    lora_config_path = os.path.join(model_path, "adapter_config.json")
    heads_path = os.path.join(model_path, "classification_heads.pt")

    has_full = os.path.exists(pytorch_path)
    has_lora = os.path.exists(lora_config_path) and os.path.exists(heads_path)

    if not has_full and not has_lora:
        return None

    logger.info(f"加载已训练模型: {model_path}")
    if has_full:
        model = ModernBERTForNERRE.from_pretrained(
            model_path, num_ner_labels=num_ner_labels, num_re_labels=num_re_labels
        )
    else:
        model = ModernBERTForNERRE(
            model_path=config.model_path,
            num_ner_labels=num_ner_labels,
            num_re_labels=num_re_labels,
        )
        from peft import PeftModel
        model.bert = PeftModel.from_pretrained(model.bert, model_path)
        heads_state = torch.load(heads_path, map_location="cpu")
        model.ner_classifier.load_state_dict(heads_state["ner_classifier"])
        model.re_classifier.load_state_dict(heads_state["re_classifier"])
        model.distance_embedding.load_state_dict(heads_state["distance_embedding"])
    model.to(device)
    return model


# ==================== 主训练函数 ====================

def train_model(
        train_loader: DataLoader,
        dev_loader: Optional[DataLoader],
        device: torch.device,
        ner_label_weights: Optional[torch.Tensor] = None,
        metrics_callback=None,
        save_first_validation: bool = False,
) -> Tuple[ModernBERTForNERRE, float]:
    best_f1 = 0.0

    model = ModernBERTForNERRE(
        model_path=config.model_path,
        num_ner_labels=config.num_ner_labels,
        num_re_labels=config.num_re_labels,
        hidden_dropout_prob=config.hidden_dropout_prob,
    )

    if ner_label_weights is not None:
        model.register_buffer("ner_label_weights", ner_label_weights)

    if config.use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            lora_cfg = LoraConfig(
                task_type=TaskType.TOKEN_CLS,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            model.bert = get_peft_model(model.bert, lora_cfg)
            logger.info(
                f"LoRA 已应用: r={config.lora_r}, alpha={config.lora_alpha}, "
                f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
            )
        except ImportError:
            logger.warning("peft 未安装，跳过 LoRA")

    model.to(device)

    existing = _load_model_if_exists(
        config.best_model_path, config.num_ner_labels, config.num_re_labels, device
    )
    # 续训时保存的 checkpoint 信息
    saved_ckpt = None
    epochs_already_trained = 0
    if existing is not None:
        model = existing
        if ner_label_weights is not None:
            model.register_buffer("ner_label_weights", ner_label_weights)
        checkpoint_path = os.path.join(config.best_model_path, "checkpoint.pt")
        if os.path.exists(checkpoint_path):
            saved_ckpt = torch.load(checkpoint_path, map_location="cpu")
            best_f1 = saved_ckpt.get("best_f1", 0.0)
            epochs_already_trained = saved_ckpt.get("epochs_trained", 0)
            logger.info(
                f"已加载现有模型，继续训练（历史最佳 F1: {best_f1:.4f}, "
                f"已训练 {epochs_already_trained} 轮，本次再训练 {config.epochs} 轮）"
            )

    # 混合精度训练
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    if use_amp:
        logger.info("已启用混合精度训练（AMP）")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # 差异化学习率：分类头使用更大学习率
    if config.classifier_lr != config.learning_rate:
        classifier_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(key in name for key in ["ner_classifier", "re_classifier", "distance_embedding"]):
                classifier_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": config.learning_rate, "weight_decay": config.weight_decay},
            {"params": classifier_params, "lr": config.classifier_lr, "weight_decay": config.weight_decay},
        ])
        logger.info(
            f"差异化学习率: encoder/LoRA={config.learning_rate}, classifier={config.classifier_lr}"
        )
    total_epochs = config.epochs  # 本次训练的轮数
    total_steps = len(train_loader) * total_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 恢复 optimizer 和 scheduler 状态（避免续训时学习率重置）
    if saved_ckpt is not None:
        if "optimizer_state_dict" in saved_ckpt:
            saved_optim_state = saved_ckpt["optimizer_state_dict"]
            # 检查参数组数量是否匹配（差异化学习率会改变参数组数量）
            if len(saved_optim_state["param_groups"]) == len(optimizer.state_dict()["param_groups"]):
                try:
                    optimizer.load_state_dict(saved_optim_state)
                    logger.info("已恢复 optimizer 状态")
                except Exception as e:
                    logger.warning(f"恢复 optimizer 状态失败: {e}")
            else:
                # 参数组数量不匹配时，只恢复参数的 momentum/variance 状态，保留新的学习率
                try:
                    new_state = optimizer.state_dict()
                    # 复制参数状态（momentum, variance 等），但保留新的 param_groups
                    for param_id, param_state in saved_optim_state["state"].items():
                        if param_id in new_state["state"]:
                            for key, value in param_state.items():
                                new_state["state"][param_id][key] = value
                    optimizer.load_state_dict(new_state)
                    logger.info(
                        f"已恢复 optimizer 参数状态（参数组已更新: "
                        f"旧={len(saved_optim_state['param_groups'])}组 -> "
                        f"新={len(optimizer.state_dict()['param_groups'])}组）"
                    )
                except Exception as e:
                    logger.warning(f"恢复 optimizer 状态失败: {e}")
        if "scheduler_state_dict" in saved_ckpt:
            # scheduler 状态与 optimizer 参数组强绑定，参数组变化时也不恢复
            saved_scheduler = saved_ckpt["scheduler_state_dict"]
            if len(saved_optim_state["param_groups"]) == len(optimizer.state_dict()["param_groups"]):
                try:
                    scheduler.load_state_dict(saved_scheduler)
                    logger.info("已恢复 scheduler 状态")
                except Exception as e:
                    logger.warning(f"恢复 scheduler 状态失败: {e}")
            else:
                logger.info("参数组数量变化，scheduler 从头初始化（使用新学习率调度）")

    os.makedirs(config.output_dir, exist_ok=True)

    # 早停计数器
    patience_counter = 0
    best_f1_with_delta = best_f1

    for epoch in range(1, total_epochs + 1):
        global_epoch = epochs_already_trained + epoch
        logger.info(f"\n{'=' * 50} Epoch {epoch}/{total_epochs} (全局第 {global_epoch} 轮) {'=' * 50}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler=scaler
        )
        logger.info(f"训练 Loss: {train_metrics['loss']:.4f}")

        if dev_loader is not None and len(dev_loader.dataset) > 0:
            val_metrics = evaluate(model, dev_loader, device, use_amp=use_amp)
            logger.info(
                f"验证 Loss: {val_metrics['loss']:.4f} | "
                f"NER F1: {val_metrics.get('ner_f1', 0):.4f} | "
                f"RE F1: {val_metrics.get('re_f1', 0):.4f} "
                f"(正F1={val_metrics.get('re_pos_f1', 0):.4f}, "
                f"neg_acc={val_metrics.get('re_neg_acc', 0):.4f})"
            )
            current_f1 = val_metrics.get("ner_f1", 0) + val_metrics.get("re_f1", 0)
            select_checkpoint = (save_first_validation and epoch == 1 and existing is None) or current_f1 > best_f1_with_delta + config.early_stopping_min_delta
            if metrics_callback is not None:
                metrics_callback({"epoch": epoch, "global_epoch": global_epoch,
                                  "train": train_metrics, "validation": val_metrics,
                                  "selection_score_ner_plus_re": current_f1,
                                  "checkpoint_selected": select_checkpoint})
            if select_checkpoint:
                # 独立入口保存首轮；后续仍按既有改进阈值选择。
                best_f1 = current_f1
                best_f1_with_delta = current_f1
                patience_counter = 0
                _save_model(model, config.best_model_path, is_lora=config.use_lora,
                            best_f1=best_f1, optimizer=optimizer, scheduler=scheduler,
                            epoch=epoch, epochs_trained=global_epoch)
                logger.info(f"★ 新的最佳模型！综合 F1: {best_f1:.4f}")
            else:
                # 指标无实质提升
                if current_f1 > best_f1:
                    best_f1 = current_f1
                patience_counter += 1
                logger.info(
                    f"验证指标无实质提升 (patience={patience_counter}/{config.early_stopping_patience})"
                )
                if patience_counter >= config.early_stopping_patience:
                    logger.info(
                        f"早停触发！连续 {config.early_stopping_patience} 轮验证指标无提升，停止训练"
                    )
                    break
        else:
            if epoch % 5 == 0 or epoch == total_epochs:
                _save_model(
                    model, config.best_model_path, is_lora=config.use_lora,
                    best_f1=best_f1, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch, epochs_trained=global_epoch
                )
                logger.info(f"Epoch {epoch} 定期保存（无验证集）")

    logger.info(f"\n训练完成！最佳综合 F1: {best_f1:.4f}")
    return model, best_f1


# ==================== 简化接口（供 fit() 调用）====================

def train_from_splits(train_data: List[Dict], dev_data: List[Dict], seed: int = 42,
                      metrics_callback=None) -> Tuple[ModernBERTForNERRE, float]:
    """Fresh experiment from explicit splits; no resplitting or oversampling.

    Freeze XML mappings and validate patient separation before calling this.
    The existing fit()/train_from_data path keeps its original behavior.
    """
    if not train_data or not dev_data:
        raise ValueError("Explicit nonempty train and validation sets are required")
    if not config.ner_labels:
        raise ValueError("Freeze XML label mappings before training explicit splits")
    if os.path.exists(config.best_model_path):
        raise ValueError("Explicit-split training requires a fresh checkpoint path")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_dataset = NERREDataset(train_data, config.tokenizer, config.max_length,
                                 config.ner_label2id, config.re_label2id)
    dev_dataset = NERREDataset(dev_data, config.tokenizer, config.max_length,
                               config.ner_label2id, config.re_label2id)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    return train_model(train_loader, dev_loader, torch.device(config.device),
                       ner_label_weights=compute_label_weights(train_data),
                       metrics_callback=metrics_callback, save_first_validation=True)


def train_from_data(
        training_data: List[Dict],
        train_ratio: float = 0.8,
        ner_label_weights: Optional[torch.Tensor] = None,
        seed: int = 42,
) -> Tuple[ModernBERTForNERRE, float]:
    """从内存数据直接训练"""
    logger.info(f"开始训练，数据条数: {len(training_data)}")
    logger.info(f"基础模型: {config.base_model_name}")

    device = torch.device(config.device)

    # 固定随机种子，确保每次训练的数据划分一致
    random.seed(seed)
    torch.manual_seed(seed)
    random.shuffle(training_data)
    if len(training_data) <= 20:
        train_data = training_data
        dev_data = []
        logger.info(f"数据量仅 {len(training_data)} 条，全部用于训练")
    else:
        split_idx = int(len(training_data) * train_ratio)
        train_data = training_data[:split_idx]
        dev_data = training_data[split_idx:]
        logger.info(f"训练集: {len(train_data)}, 验证集: {len(dev_data)}")

    if len(train_data) < 30:
        original_len = len(train_data)
        repeats = (30 // original_len) + 1
        # 打乱后复制，避免连续重复同一样本导致过拟合
        oversampled = []
        for _ in range(repeats):
            shuffled = train_data.copy()
            random.shuffle(shuffled)
            oversampled.extend(shuffled)
        train_data = oversampled[:30]
        logger.info(f"过采样: {original_len} -> {len(train_data)} 条（已打乱）")

    if not config.ner_labels or not config.re_labels:
        _extract_labels_from_data(training_data)

    train_dataset = NERREDataset(
        train_data, config.tokenizer, config.max_length,
        config.ner_label2id, config.re_label2id,
    )
    dev_dataset = NERREDataset(
        dev_data, config.tokenizer, config.max_length,
        config.ner_label2id, config.re_label2id,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn
    ) if dev_data else None

    logger.info(
        f"数据加载完成: 训练集={len(train_dataset)}条({len(train_loader)}batch), "
        f"验证集={len(dev_dataset)}条"
    )

    if ner_label_weights is None:
        ner_label_weights = compute_label_weights(train_data)

    return train_model(
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        ner_label_weights=ner_label_weights,
    )


def _extract_labels_from_data(data: List[Dict]):
    """从训练数据中提取标签并更新配置"""
    ner_labels = set()
    re_labels = set()
    for sample in data:
        for ent in sample.get("entities", []):
            ner_labels.add(ent.get("label", ""))
        for rel in sample.get("relations", []):
            re_labels.add(rel.get("type", ""))
    ner_labels.discard("")
    re_labels.discard("")
    config.update_labels(sorted(ner_labels), sorted(re_labels))
    logger.info(f"从数据提取标签 - NER: {sorted(ner_labels)}, RE: {sorted(re_labels)}")


def main():
    """从文件加载数据训练"""
    import argparse
    parser = argparse.ArgumentParser(description='ModernBERT NER+RE Training')
    parser.add_argument('--train-file', type=str, required=True, help='训练数据文件路径')
    parser.add_argument('--base-model', type=str, default=None, help='基础模型名称')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    args = parser.parse_args()

    if args.base_model:
        config.update_model_paths(args.base_model, config.model_scope)
    if args.epochs:
        config.epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size

    device = torch.device(config.device)
    logger.info(f"使用设备: {device}")
    logger.info(f"基础模型: {config.base_model_name}")

    if os.path.exists(args.train_file):
        with open(args.train_file, "r", encoding="utf-8") as f:
            training_data = json.load(f)
        train_from_data(training_data, train_ratio=0.9)
    else:
        logger.error(f"训练数据不存在: {args.train_file}")


if __name__ == "__main__":
    main()
