import logging
import os
from typing import Any, Dict, List, Optional

from label_studio_ml.model import LabelStudioMLBase

from .prompts import GLINER2_LABELS, LABEL_PROMPTS, PATHOGEN_GROUPS

logger = logging.getLogger(__name__)

# 置信度阈值：中文医疗文本建议 0.35~0.45，比默认 0.5 更宽松避免漏召回
THRESHOLD = float(os.getenv("GLINER_THRESHOLD", "0.4"))
# 模型名：GLiNER2 默认推荐 fastino/gliner2-base-v1
MODEL_NAME = os.getenv("GLINER_MODEL", "fastino/gliner2-base-v1")


class PneumoniaNERModel(LabelStudioMLBase):
    """
    GLiNER2-based ML backend for pneumonia NER annotation.
    对应 pneumonia.xml 的完整标签体系。
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
        logger.info(f"Loading GLiNER2 model: {MODEL_NAME}")
        self.gliner = GLiNER2.from_pretrained(MODEL_NAME)
        
        # 预构建 schema
        self._schema = self.gliner.create_schema().entities({
            label: {"description": desc, "threshold": THRESHOLD}
            for label, desc in GLINER2_LABELS.items()
        })
        logger.info("Model and schema loaded.")

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
            text = self._extract_text(task)
            if not text:
                predictions.append({"result": [], "score": 0.0})
                continue

            # 使用 GLiNER2 extract API，指定 include_spans=True 获取 offset
            raw = self.gliner.extract(
                text,
                self._schema,
                include_confidence=True,
                include_spans=True,
            )

            result = []
            total_score = 0.0
            
            entities_dict = raw.get("entities", {})

            for label_value, spans in entities_dict.items():
                if label_value not in LABEL_PROMPTS:
                    continue

                from_name = LABEL_PROMPTS[label_value][1]
                to_name = "text" if from_name in PATHOGEN_GROUPS else "chief_complaint_text"
                
                for span in spans:
                    score = float(span.get("confidence", 0.0))
                    total_score += score

                    result.append({
                        "from_name": from_name,
                        "to_name": to_name,
                        "type": "labels",
                        "score": score,
                        "value": {
                            "start":  span["start"],
                            "end":    span["end"],
                            "text":   span["text"],
                            "labels": [label_value],
                        },
                    })

            avg_score = total_score / len(result) if result else 0.0
            predictions.append({"result": result, "score": avg_score})
            logger.debug(f"Task {task.get('id')}: {len(result)} entities, avg_score={avg_score:.3f}")

        return predictions

    # ── helpers ────────────────────────────────────────────────────

    def _extract_text(self, task: Dict) -> str:
        """
        从 task data 里拼出完整文本。
        Label Studio 把模板变量渲染后的纯文本放在 task['data'] 里，
        但多次就诊记录是数组，需要手动拼接。
        """
        data: Dict[str, Any] = task.get("data", {})

        # 优先取已渲染的纯文本字段
        if "chief_complaint_text" in data:
            return str(data["chief_complaint_text"])

        # 否则从 emr_activity_info 数组拼接
        activities: List[Dict] = data.get("emr_activity_info", [])
        parts = []
        for act in activities:
            time_str = act.get("activity_time", "")
            cc = act.get("chief_complaint", "")
            pih = act.get("present_illness_his", "")
            parts.append(f"就诊时间：{time_str}\n主诉：{cc}\n现病史：{pih}")
        return "\n---------\n".join(parts)
