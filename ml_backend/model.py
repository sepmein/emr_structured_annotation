import logging
import os
import json
from typing import Any, Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase
from .prompts import GLINER2_LABELS, LABEL_PROMPTS, PATHOGEN_GROUPS

# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

THRESHOLD = float(os.getenv("GLINER_THRESHOLD", "0.4"))
MODEL_NAME = os.getenv("GLINER_MODEL", "fastino/gliner2-base-v1")

class PneumoniaNERModel(LabelStudioMLBase):
    """
    [DEBUG VERSION] GLiNER2-based ML backend for pneumonia NER annotation.
    """

    def __init__(self, **kwargs):
        super(PneumoniaNERModel, self).__init__(**kwargs)
        self.model_dir = kwargs.get('model_dir') or os.getenv('MODEL_DIR', './models')
        self.gliner = None
        self._schema = None

    def _lazy_init(self):
        """延迟初始化模型，确保在预测前加载。"""
        if self.gliner is not None:
            return

        from gliner2 import GLiNER2
        logger.info(f"--- [DEBUG] 正在从 {MODEL_NAME} 加载 GLiNER2 模型 ---")
        self.gliner = GLiNER2.from_pretrained(MODEL_NAME)
        
        # 预构建 schema
        self._schema = self.gliner.create_schema().entities({
            label: {"description": desc, "threshold": THRESHOLD}
            for label, desc in GLINER2_LABELS.items()
        })
        logger.info("--- [DEBUG] 模型和 Schema 已就绪 ---")

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
            raw = self.gliner.extract(
                text,
                self._schema,
                include_confidence=True,
                include_spans=True,
            )

            # DEBUG 2: 打印 GLiNER2 原始返回的详细 JSON
            logger.debug(f"[TASK {task_id}] GLiNER2 原始识别 JSON:\n{json.dumps(raw, ensure_ascii=False, indent=2)}")

            result = []
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
                    logger.debug(f"  识别成功: [{entity_text}] -> {label_value} (置信度: {score:.4f})")

                    total_score += score
                    result.append({
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "score": score,
                        "value": {
                            "start":  span["start"],
                            "end":    span["end"],
                            "text":   entity_text,
                            "labels": [label_value],
                        },
                    })

            avg_score = total_score / len(result) if result else 0.0
            predictions.append({"result": result, "score": avg_score})
            logger.info(f"[TASK {task_id}] 预测结束, 识别出 {len(result)} 个实体, 平均置信度 {avg_score:.3f}")

        return predictions

    def _extract_text(self, task: Dict) -> str:
        """
        [IMPORTANT] 精确还原 XML 模板渲染出的字符串。
        偏差 1 个字符都会导致标注错位。
        """
        data: Dict[str, Any] = task.get("data", {})
        if "chief_complaint_text" in data and isinstance(data["chief_complaint_text"], str):
            return data["chief_complaint_text"]

        activities: List[Dict] = data.get("emr_activity_info", [])
        def get_val(idx, field):
            if idx < len(activities):
                v = activities[idx].get(field)
                return str(v) if v is not None else ""
            return "undefined"

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
