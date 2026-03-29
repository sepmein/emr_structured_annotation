import logging
import os
import pathlib
from math import floor
from typing import Dict, List, Optional

import label_studio_sdk
from label_studio_sdk.label_interface.objects import PredictionValue

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse

logger = logging.getLogger(__name__)

GLINER_MODEL_NAME = os.getenv("GLINER_MODEL_NAME", "fastino/gliner2-base-v1")

# ── 提示词词典 + per-entity threshold ────────────────────────────────────────
# 格式：{ "XML标签值": {"description": "提示词", "threshold": 0.x, "from_name": "..."} }
# description 就是 GLiNER2 的提示词，越精确召回率越高
# threshold 按标签类型差异化设置：断言词要高精度，症状词宽松一些

LABEL_SCHEMA: Dict[str, Dict] = {

    # ── 症状 ──────────────────────────────────────────────────────────────
    "发热": {
        "description": "发热症状，包括体温升高、发烧、高热、低热、热退 / fever, high temperature, pyrexia",
        "threshold": 0.35,
        "from_name": "symptons_labels",
    },
    "气促": {
        "description": "呼吸急促、气促、呼吸增快 / shortness of breath, tachypnea",
        "threshold": 0.35,
        "from_name": "symptons_labels",
    },
    "鼻翼煽动": {
        "description": "鼻翼煽动、鼻翼扇动，吸气时鼻孔扩张 / nasal flaring",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },
    "三凹征": {
        "description": "三凹征，吸气时胸骨上窝、锁骨上窝、肋间隙凹陷 / intercostal retractions",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },
    "喘鸣或喘息": {
        "description": "喘鸣、喘息、哮鸣音、呼吸时有哨音 / wheezing, stridor",
        "threshold": 0.35,
        "from_name": "symptons_labels",
    },
    "意识障碍": {
        "description": "意识障碍、昏迷、嗜睡、谵妄、烦躁、神志不清 / altered consciousness, confusion, coma",
        "threshold": 0.35,
        "from_name": "symptons_labels",
    },
    "惊厥": {
        "description": "惊厥、抽搐、癫痫发作、抽风 / seizure, convulsion",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },
    "拒食或喂养困难": {
        "description": "拒食、拒奶、喂养困难、不吃奶、进食差 / poor feeding, feeding difficulty",
        "threshold": 0.35,
        "from_name": "symptons_labels",
    },
    "呼吸衰竭": {
        "description": "呼吸衰竭、I型呼吸衰竭、II型呼吸衰竭、呼衰 / respiratory failure",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },
    "休克": {
        "description": "休克、感染性休克、脓毒性休克、循环衰竭 / shock, septic shock",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },
    "器官衰竭": {
        "description": "器官衰竭、多器官功能障碍、MODS / organ failure, multi-organ dysfunction",
        "threshold": 0.4,
        "from_name": "symptons_labels",
    },

    # ── 诊断 ──────────────────────────────────────────────────────────────
    "肺实质异常影": {
        "description": "肺实质异常影像，肺部阴影、渗出影、实变影、磨玻璃影 / pulmonary infiltrate, consolidation, ground-glass opacity",
        "threshold": 0.4,
        "from_name": "diagnosis_labels",
    },
    "肺组织炎症": {
        "description": "肺组织炎症、肺炎、肺部炎性改变 / pulmonary inflammation, pneumonia",
        "threshold": 0.4,
        "from_name": "diagnosis_labels",
    },
    "肺炎诊断": {
        "description": "肺炎诊断，社区获得性肺炎、医院获得性肺炎、CAP、HAP / pneumonia diagnosis",
        "threshold": 0.4,
        "from_name": "diagnosis_labels",
    },
    "肺部感染": {
        "description": "肺部感染、下呼吸道感染 / lung infection, lower respiratory tract infection",
        "threshold": 0.4,
        "from_name": "diagnosis_labels",
    },

    # ── 治疗 ──────────────────────────────────────────────────────────────
    "机械通气": {
        "description": "机械通气、有创通气、无创通气、气管插管、呼吸机、CPAP、BiPAP / mechanical ventilation, intubation",
        "threshold": 0.4,
        "from_name": "treatment_labels",
    },

    # ── 流行病学 ──────────────────────────────────────────────────────────
    "医务人员": {
        "description": "医务人员、医生、护士、医护人员 / healthcare worker, medical staff",
        "threshold": 0.45,
        "from_name": "epidemics_labels",
    },
    "禽、畜类从业人员": {
        "description": "禽类从业人员、畜牧业人员、养殖场工人 / poultry worker, livestock worker",
        "threshold": 0.45,
        "from_name": "epidemics_labels",
    },
    "农牧民等野外作业人员": {
        "description": "农民、牧民、野外作业人员 / farmer, agricultural worker, outdoor worker",
        "threshold": 0.45,
        "from_name": "epidemics_labels",
    },
    "病原微生物实验室检测人员": {
        "description": "实验室人员、微生物检测人员 / laboratory worker, microbiologist",
        "threshold": 0.45,
        "from_name": "epidemics_labels",
    },
    "可疑动物或动物制品接触史": {
        "description": "动物接触史、野生动物接触、动物制品接触 / animal contact history, wildlife exposure",
        "threshold": 0.4,
        "from_name": "epidemics_labels",
    },
    "可疑环境暴露史": {
        "description": "环境暴露史、疫区暴露、高风险环境暴露 / environmental exposure, epidemic area exposure",
        "threshold": 0.4,
        "from_name": "epidemics_labels",
    },
    "外出外来史": {
        "description": "外出史、旅行史、外来人员、流行地区旅居史 / travel history, migration history",
        "threshold": 0.4,
        "from_name": "epidemics_labels",
    },

    # ── 时间 ──────────────────────────────────────────────────────────────
    "当前": {
        "description": "当前时间状态词：目前、现在、此次、本次 / current, present, this episode",
        "threshold": 0.5,
        "from_name": "time_labels",
    },
    "既往": {
        "description": "既往时间状态词：既往、以前、曾经、过去、历史上 / previous, past, history of, prior",
        "threshold": 0.5,
        "from_name": "time_labels",
    },
    "进行性加重": {
        "description": "进行性加重：进行性、逐渐加重、持续恶化、进展性 / progressive, worsening, deteriorating",
        "threshold": 0.45,
        "from_name": "time_labels",
    },
    "持续": {
        "description": "持续时间描述：持续、一直、连续、不间断 / persistent, continuous, ongoing",
        "threshold": 0.5,
        "from_name": "time_labels",
    },

    # ── 断言（高精度，threshold 较高）────────────────────────────────────
    "肯定": {
        "description": "肯定断言词：确有、明确、确实、已证实 / confirmed, definite, positive finding",
        "threshold": 0.6,
        "from_name": "status_labels",
    },
    "否定": {
        "description": "否定断言词：无、未、否认、不伴、未见、排除、没有 / no, without, denied, absent, negative, ruled out",
        "threshold": 0.55,
        "from_name": "status_labels",
    },
    "可疑": {
        "description": "可疑断言词：疑似、可能、不除外、考虑、待排、疑诊 / suspected, possible, cannot exclude",
        "threshold": 0.55,
        "from_name": "status_labels",
    },
    "条件性": {
        "description": "条件性断言词：若、如果、当…时、在…情况下 / if, when, conditional",
        "threshold": 0.6,
        "from_name": "status_labels",
    },
    "假设性": {
        "description": "假设性断言词：假设、推测、估计、预计、可能为 / hypothetical, estimated, presumed",
        "threshold": 0.6,
        "from_name": "status_labels",
    },
    "与患者本人无关": {
        "description": "与患者本人无关：家族史、家属、父母、兄弟姐妹 / family history, relative, not the patient",
        "threshold": 0.6,
        "from_name": "status_labels",
    },

    # ── 测量实体 ──────────────────────────────────────────────────────────
    "呼吸频率": {
        "description": "呼吸频率指标：呼吸频率、呼吸次数、RR / respiratory rate, RR",
        "threshold": 0.45,
        "from_name": "measure_entities",
    },
    "指氧饱和度": {
        "description": "指氧饱和度指标：指氧饱和度、血氧饱和度、SpO2 / oxygen saturation, SpO2",
        "threshold": 0.45,
        "from_name": "measure_entities",
    },
    "动脉氧分压(PaO2)": {
        "description": "动脉氧分压指标：动脉氧分压、PaO2 / arterial oxygen partial pressure, PaO2",
        "threshold": 0.45,
        "from_name": "measure_entities",
    },
    "吸氧浓度(FiO2)": {
        "description": "吸氧浓度指标：吸氧浓度、FiO2 / fraction of inspired oxygen, FiO2",
        "threshold": 0.45,
        "from_name": "measure_entities",
    },

    # ── 测量属性 ──────────────────────────────────────────────────────────
    "数值": {
        "description": "测量数值，具体数字读数：38.5、120、96 / numeric value, measurement reading",
        "threshold": 0.4,
        "from_name": "measure_labels",
    },
    "单位": {
        "description": "测量单位：℃、次/分、mmHg、% / unit of measurement",
        "threshold": 0.5,
        "from_name": "measure_labels",
    },
    "比较符": {
        "description": "比较符号：大于、小于、≥、≤、高于、低于 / greater than, less than, above, below",
        "threshold": 0.55,
        "from_name": "measure_labels",
    },
    "阈值判断": {
        "description": "阈值判断：超过正常范围、低于正常值、异常升高、明显下降 / above normal, below normal, abnormal",
        "threshold": 0.5,
        "from_name": "measure_labels",
    },

    # ── 病毒 ──────────────────────────────────────────────────────────────
    "新冠病毒": {
        "description": "新冠病毒、SARS-CoV-2、COVID-19 / COVID-19, SARS-CoV-2",
        "threshold": 0.5,
        "from_name": "pathogen",
    },
    "流感病毒": {
        "description": "流感病毒、甲型流感、乙型流感、H1N1、H3N2 / influenza virus, flu",
        "threshold": 0.5,
        "from_name": "pathogen",
    },
    "呼吸道合胞病毒": {
        "description": "呼吸道合胞病毒、RSV / respiratory syncytial virus, RSV",
        "threshold": 0.5,
        "from_name": "pathogen",
    },
    "腺病毒":     {"description": "腺病毒 / adenovirus", "threshold": 0.5, "from_name": "pathogen"},
    "人偏肺病毒": {"description": "人偏肺病毒、hMPV / human metapneumovirus", "threshold": 0.5, "from_name": "pathogen"},
    "副流感病毒": {"description": "副流感病毒 / parainfluenza virus", "threshold": 0.5, "from_name": "pathogen"},
    "普通冠状病毒": {"description": "普通冠状病毒、季节性冠状病毒（非新冠）/ seasonal coronavirus", "threshold": 0.5, "from_name": "pathogen"},
    "博卡病毒":   {"description": "博卡病毒、HBoV / bocavirus", "threshold": 0.5, "from_name": "pathogen"},
    "鼻病毒":     {"description": "鼻病毒 / rhinovirus", "threshold": 0.5, "from_name": "pathogen"},
    "肠道病毒":   {"description": "肠道病毒、EV71、柯萨奇病毒 / enterovirus, EV71", "threshold": 0.5, "from_name": "pathogen"},

    # ── 细菌 ──────────────────────────────────────────────────────────────
    "A族链球菌":    {"description": "A族链球菌、化脓性链球菌 / group A streptococcus", "threshold": 0.5, "from_name": "bacteria"},
    "百日咳鲍特菌": {"description": "百日咳鲍特菌 / Bordetella pertussis", "threshold": 0.5, "from_name": "bacteria"},
    "肺炎链球菌":   {"description": "肺炎链球菌、肺炎球菌 / Streptococcus pneumoniae", "threshold": 0.5, "from_name": "bacteria"},
    "流感嗜血杆菌": {"description": "流感嗜血杆菌 / Haemophilus influenzae", "threshold": 0.5, "from_name": "bacteria"},
    "军团菌":       {"description": "军团菌、嗜肺军团菌 / Legionella pneumophila", "threshold": 0.5, "from_name": "bacteria"},
    "肺炎克雷伯菌": {"description": "肺炎克雷伯菌 / Klebsiella pneumoniae", "threshold": 0.5, "from_name": "bacteria"},

    # ── 其他病原体 ────────────────────────────────────────────────────────
    "肺炎支原体":   {"description": "肺炎支原体、MP / Mycoplasma pneumoniae", "threshold": 0.5, "from_name": "other_pathogen"},
    "曲霉菌":       {"description": "曲霉菌、侵袭性曲霉病 / Aspergillus", "threshold": 0.5, "from_name": "other_pathogen"},
    "隐球菌":       {"description": "隐球菌 / Cryptococcus", "threshold": 0.5, "from_name": "other_pathogen"},
    "鹦鹉热衣原体": {"description": "鹦鹉热衣原体 / Chlamydia psittaci", "threshold": 0.5, "from_name": "other_pathogen"},
    "肺炎衣原体":   {"description": "肺炎衣原体 / Chlamydia pneumoniae", "threshold": 0.5, "from_name": "other_pathogen"},
}


class GLiNERModel(LabelStudioMLBase):
    """
    GLiNER2 ML Backend for pneumonia NER annotation.
    使用 GLiNER2 schema API：description 作为提示词，per-entity threshold。
    """

    def setup(self):
        self.LABEL_STUDIO_HOST = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
        self.LABEL_STUDIO_API_KEY = os.getenv("LABEL_STUDIO_API_KEY")
        self.MODEL_DIR = os.getenv("MODEL_DIR", "/data/models")
        self.finetuned_model_path = os.getenv("FINETUNED_MODEL_PATH", "finetuned_model")
        self.default_threshold = float(os.getenv("THRESHOLD", 0.4))
        self.extractor = None

    def lazy_init(self):
        if not self.extractor:
            from gliner2 import GLiNER2
            ckpt = str(pathlib.Path(self.MODEL_DIR, self.finetuned_model_path))
            try:
                logger.info(f"Loading finetuned model from {ckpt}")
                self.extractor = GLiNER2.from_pretrained(ckpt, local_files_only=True)
                self.set("model_version", f"{self.__class__.__name__}-finetuned")
            except Exception:
                logger.info(f"No finetuned model found. Loading {GLINER_MODEL_NAME}")
                self.extractor = GLiNER2.from_pretrained(GLINER_MODEL_NAME)
                self.set("model_version", f"{self.__class__.__name__}-pretrained")

            # 预构建 schema（启动时一次性构建，避免每次预测重建）
            self._schema = self._build_schema()

    def _build_schema(self):
        """构建 GLiNER2 schema，每个标签带 description 和 per-entity threshold。"""
        schema_def = {
            label: {
                "description": cfg["description"],
                "dtype": "list",
                "threshold": cfg["threshold"],
            }
            for label, cfg in LABEL_SCHEMA.items()
        }
        return self.extractor.create_schema().entities(schema_def)

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        self.lazy_init()

        # 从 XML 配置自动读取 to_name（Text 组件的 name 属性）
        _, to_name, value = self.label_interface.get_first_tag_occurence("Labels", "Text")

        predictions = []
        for task in tasks:
            text = task["data"].get(value, "")
            if not text:
                text = self._build_text_from_activities(task["data"])

            raw = self.extractor.extract(
                text,
                self._schema,
                include_confidence=True,
                include_spans=True,
            )

            result = self._convert_to_ls_format(raw, to_name)
            score = min(r["score"] for r in result) if result else 0.0
            predictions.append(PredictionValue(
                result=result,
                score=score,
                model_version=self.get("model_version"),
            ))

        return ModelResponse(predictions=predictions)

    def _convert_to_ls_format(self, raw: Dict, to_name: str) -> List[Dict]:
        """
        把 GLiNER2 输出转换为 Label Studio result 格式。
        raw 格式：{"entities": {"标签名": [{"text": ..., "confidence": ..., "start": ..., "end": ...}]}}
        """
        result = []
        entities_dict = raw.get("entities", {})

        for label_value, items in entities_dict.items():
            cfg = LABEL_SCHEMA.get(label_value)
            if not cfg:
                continue
            from_name = cfg["from_name"]

            for item in items:
                if not isinstance(item, dict):
                    continue
                result.append({
                    "from_name": from_name,
                    "to_name": to_name,
                    "type": "labels",
                    "score": round(float(item.get("confidence", 0.0)), 4),
                    "value": {
                        "start":  item["start"],
                        "end":    item["end"],
                        "text":   item["text"],
                        "labels": [label_value],
                    },
                })
        return result

    def _build_text_from_activities(self, data: Dict) -> str:
        activities = data.get("emr_activity_info", [])
        parts = []
        for act in activities:
            parts.append(
                f"就诊时间：{act.get('activity_time', '')}\n"
                f"主诉：{act.get('chief_complaint', '')}\n"
                f"现病史：{act.get('present_illness_his', '')}"
            )
        return "\n---------\n".join(parts)

    def fit(self, event, data, **kwargs):
        """训练触发入口，从 Label Studio 下载标注数据后微调模型。"""
        self.lazy_init()
        if event != "START_TRAINING":
            logger.info("Training not triggered")
            return

        logger.info("Starting training...")
        ls = label_studio_sdk.Client(self.LABEL_STUDIO_HOST, self.LABEL_STUDIO_API_KEY)
        project = ls.get_project(id=self.project_id)
        tasks = project.get_labeled_tasks()
        logger.info(f"Downloaded {len(tasks)} labeled tasks")

        # GLiNER2 微调需要原版 GLiNER 格式，此处留作扩展
        # 参考：https://github.com/fastino-ai/GLiNER2
        logger.info("Fine-tuning with GLiNER2 not yet implemented — skipping")
