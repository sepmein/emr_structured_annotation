import logging
import os
import json
import pathlib
from typing import Any, Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase
from .prompts import GLINER2_LABELS, LABEL_PROMPTS, PATHOGEN_GROUPS, RELATION_SCHEMA

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

THRESHOLD = float(os.getenv("GLINER_THRESHOLD", "0.4"))
MODEL_NAME = os.getenv("GLINER_MODEL", "fastino/gliner2-base-v1")

class PneumoniaNERModel(LabelStudioMLBase):
    """
    GLiNER2-based ML backend for pneumonia NER annotation.
    Uses class-level singleton pattern to share model across instances.
    """
    # 类变量：所有实例共享同一个模型
    _gliner = None
    _schema = None
    _initialized = False

    def __init__(self, **kwargs):
        # 先初始化属性，因为父类 __init__ 会调用 setup()
        self.MODEL_DIR = kwargs.get('model_dir') or os.getenv('MODEL_DIR', './models')
        self.finetuned_model_path = os.getenv('FINETUNED_MODEL_PATH', 'finetuned_model')
        self.threshold = float(os.getenv('THRESHOLD', THRESHOLD))
        super(PneumoniaNERModel, self).__init__(**kwargs)

    def _lazy_init(self):
        """延迟初始化模型，确保在预测前加载。只加载一次。"""
        # 用类变量判断是否已初始化
        if self.__class__._initialized and self.__class__._gliner is not None:
            return

        from gliner2 import GLiNER2

        # 尝试加载 fine-tuned 模型，失败则回退到预训练模型
        model_path = pathlib.Path(self.MODEL_DIR, self.finetuned_model_path)

        # 检查本地路径是否存在，避免尝试从 HuggingFace 下载不存在的模型
        if model_path.exists() and (model_path / "config.json").exists():
            try:
                logger.info(f"正在从 {model_path} 加载 fine-tuned GLiNER2 模型")
                self.__class__._gliner = GLiNER2.from_pretrained(str(model_path))
                logger.info("成功加载 fine-tuned 模型")
            except Exception as e:
                logger.warning(f"Failed to load fine-tuned model from {model_path}: {e}")
                logger.info(f"回退到预训练模型: {MODEL_NAME}")
                # 设置离线模式，避免 HuggingFace 超时（使用本地缓存）
                os.environ['HF_HUB_OFFLINE'] = '1'
                os.environ['TRANSFORMERS_OFFLINE'] = '1'
                self.__class__._gliner = GLiNER2.from_pretrained(MODEL_NAME)
        else:
            logger.info(f"Fine-tuned 模型路径不存在: {model_path}")
            logger.info(f"直接加载预训练模型: {MODEL_NAME}")
            # 设置离线模式，避免 HuggingFace 超时（使用本地缓存）
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            self.__class__._gliner = GLiNER2.from_pretrained(MODEL_NAME)

        # 预构建 schema：实体 + 关系
        schema = self.__class__._gliner.create_schema().entities({
            label: {"description": desc, "threshold": self.threshold}
            for label, desc in GLINER2_LABELS.items()
        })

        # 添加关系定义
        for rel_name, rel_cfg in RELATION_SCHEMA.items():
            schema = schema.relations({
                rel_name: {
                    "description": rel_cfg["description"],
                    "threshold": rel_cfg["threshold"],
                }
            })

        self.__class__._schema = schema
        self.__class__._initialized = True
        logger.info("模型和 Schema 已就绪（实体 + 关系）")

    def setup(self):
        """Label Studio 启动时尝试初始化。"""
        self._lazy_init()

    def predict(
        self,
        tasks: List[Dict],
        context: Optional[Dict] = None,
        **kwargs,
    ) -> List[Dict]:
        self._lazy_init()
        predictions = []

        for task in tasks:
            task_id = task.get('id', 'unknown')
            text = self._extract_text(task)

            # DEBUG 1: 打印输入给模型的原始文本
            logger.debug(f"\n[TASK {task_id}] 输入模型文本 (前500字):\n{text[:500]}...\n")

            if not text:
                predictions.append({"result": [], "score": 0.0})
                continue

            # 执行提取
            raw = self.__class__._gliner.extract(
                text,
                schema=self.__class__._schema,
                threshold=self.threshold,
                include_confidence=True,
                include_spans=True,
            )

            # DEBUG 2: 打印 GLiNER2 原始返回的详细 JSON
            logger.debug(f"[TASK {task_id}] GLiNER2 原始识别 JSON:\n{json.dumps(raw, ensure_ascii=False, indent=2)}")

            # ── 第一步：转换实体，同时建立 (start, end, label) -> result_id 索引 ──
            result = []
            span_index: Dict[tuple, str] = {}  # (start, end, label) -> id
            total_score = 0.0
            entities_dict = raw.get("entities", {})

            for label_value, spans in entities_dict.items():
                if label_value not in LABEL_PROMPTS:
                    logger.warning(f"跳过未知标签: {label_value}")
                    continue

                from_name = LABEL_PROMPTS[label_value][1]
                to_name = "chief_complaint_text" # 统一使用这个目标

                for span in spans:
                    entity_text = span.get("text", "")
                    score = float(span.get("confidence", 0.0))
                    start, end = span["start"], span["end"]

                    # 生成稳定唯一 ID，供关系对象引用
                    result_id = f"{from_name}_{start}_{end}"
                    span_index[(start, end, label_value)] = result_id

                    logger.debug(f"  实体: [{entity_text}] -> {label_value} (id={result_id}, 置信度={score:.4f})")

                    total_score += score
                    result.append({
                        "id": result_id,
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "score": score,
                        "value": {
                            "start":  start,
                            "end":    end,
                            "text":   entity_text,
                            "labels": [label_value],
                        },
                    })

            # ── 第二步：转换关系，引用已生成的实体 ID ──
            relations_dict = raw.get("relations", {})
            rel_count = 0
            for rel_name, rel_instances in relations_dict.items():
                ls_relation = RELATION_SCHEMA.get(rel_name, {}).get("ls_relation", rel_name)
                for inst in rel_instances:
                    head = inst.get("head")
                    tail = inst.get("tail")
                    if not head or not tail:
                        continue

                    # 在 span_index 中查找匹配的实体 ID
                    head_id = self._find_span_id(span_index, head["start"], head["end"])
                    tail_id = self._find_span_id(span_index, tail["start"], tail["end"])

                    if not head_id or not tail_id:
                        # 关系端点没有对应实体 span，跳过
                        logger.debug(
                            f"  关系 [{rel_name}] 端点未找到对应实体: "
                            f"head={head.get('text')}({head['start']}:{head['end']}) "
                            f"tail={tail.get('text')}({tail['start']}:{tail['end']})"
                        )
                        continue

                    logger.debug(
                        f"  关系: {head.get('text')} --[{ls_relation}]--> {tail.get('text')} "
                        f"(from_id={head_id}, to_id={tail_id})"
                    )
                    result.append({
                        "type": "relation",
                        "from_id": head_id,
                        "to_id": tail_id,
                        "direction": "right",
                        "labels": [ls_relation],
                    })
                    rel_count += 1

            avg_score = total_score / len([r for r in result if r.get("type") == "labels"]) if result else 0.0
            predictions.append({"result": result, "score": avg_score})
            logger.info(
                f"[TASK {task_id}] 预测结束: "
                f"{len(result) - rel_count} 个实体, "
                f"{rel_count} 个关系, "
                f"平均置信度 {avg_score:.3f}"
            )

        return predictions

    def _find_span_id(self, span_index: Dict[tuple, str], start: int, end: int) -> Optional[str]:
        """在 span_index 中查找 (start, end) 匹配的任意标签的 ID。"""
        for (s, e, _label), sid in span_index.items():
            if s == start and e == end:
                return sid
        return None

    def _extract_text(self, task: Dict) -> str:
        """
        [IMPORTANT] 精确还原 XML 模板渲染出的字符串。
        偏差 1 个字符都会导致标注错位。

        支持三种数据格式：
        1. 预渲染的 chief_complaint_text 字段（直接返回）
        2. 扁平字段：activity_time / chief_complaint / present_illness_his /
           physical_examination / studies_summary_result（单次就诊）
        3. emr_activity_info 数组（多次就诊，最多 7 次）
        """
        data: Dict[str, Any] = task.get("data", {})

        # 格式 1：预渲染文本
        if "chief_complaint_text" in data and isinstance(data["chief_complaint_text"], str):
            return data["chief_complaint_text"]

        # 格式 2：扁平单次就诊字段
        # XML 模板: value="就诊时间：$activity_time&#10;&#10;主诉：$chief_complaint&#10;&#10;..."
        # &#10; 渲染为 \n，加上 XML 文本中的字面换行，每个 &#10; 后产生 \n\n
        if "activity_time" in data or "chief_complaint" in data:
            def flat_val(field):
                v = data.get(field)
                return str(v) if v is not None else ""

            t   = flat_val("activity_time")
            cc  = flat_val("chief_complaint")
            pih = flat_val("present_illness_his")
            pe  = flat_val("physical_examination")
            ssr = flat_val("studies_summary_result")
            return f"就诊时间：{t}\n\n主诉：{cc}\n\n现病史：{pih}\n\n检查：{pe}\n\n检验：{ssr}"

        # 格式 3：多次就诊 emr_activity_info 数组
        activities: List[Dict] = data.get("emr_activity_info", [])
        def get_val(idx, field):
            if idx < len(activities):
                v = activities[idx].get(field)
                return str(v) if v is not None else ""
            return ""

        parts = []
        for i in range(7):
            t = get_val(i, "activity_time")
            cc = get_val(i, "chief_complaint")
            pih = get_val(i, "present_illness_his")

            # XML 模板中每个 &#10; 会渲染为 \n，且其后还有 XML 文本中的字面换行符，
            # 因此 activity_time 和 chief_complaint 行各产生双换行 \n\n，
            # 而 present_illness_his 行只有一个字面换行 \n（无 &#10;）。
            part = (
                f"          就诊时间：{t} \n\n"
                f"          主诉：{cc}  \n\n"
                f"          现病史：{pih}\n"
            )
            parts.append(part)

        # 分隔符来自 XML 中 "          &#10;---------&#10;\n"
        # 渲染后为：10空格 + \n + --------- + \n + 字面\n = "          \n---------\n\n"
        full_text = "          \n---------\n\n".join(parts)
        return "\n" + full_text + "        "
