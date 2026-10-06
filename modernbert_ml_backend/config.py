"""
ModernBERT NER+RE ML Backend 配置
"""
import os
import torch
from transformers import AutoTokenizer

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_project_env():
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if load_dotenv:
        load_dotenv(env_path)
        return
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


_load_project_env()

if os.getenv("DISABLE_MKLDNN", "0").lower() in {"1", "true", "yes"}:
    torch.backends.mkldnn.enabled = False


class Config:
    """全局配置类"""

    def __init__(self, base_model_name: str = None):
        # 基础模型名称
        self.base_model_name = base_model_name or os.environ.get(
            "BASE_MODEL_NAME", "chinese-modernbert-large-wwm"
        )

        # 项目根目录
        self.root_dir = PROJECT_ROOT

        # 模型路径。Label Studio 请求会按标签 schema 切换 model_scope。
        self.model_scope = None
        self.update_model_paths()

        # 模型配置
        self.max_length = max(1, self._get_int_env("MAX_LENGTH", 1024))
        self.hidden_dropout_prob = 0.1

        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 标签配置由 Label Studio 项目的 label_config 动态同步
        self.ner_labels = []
        self.re_labels = []
        self._build_label_mappings()

        # 训练配置
        self.epochs = 20
        self.batch_size = 4
        self.learning_rate = 5e-5
        self.classifier_lr = 5e-4  # 分类头学习率（比 LoRA 大 10 倍）
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # 损失权重（NER loss 和 RE loss 的平衡）
        self.ner_loss_weight = 1.0
        self.re_loss_weight = 2.0  # RE 样本少，需要更大权重

        # 早停配置
        self.early_stopping_patience = 5  # 连续 N 轮验证指标无提升则停止
        self.early_stopping_min_delta = 0.001  # 最小提升阈值

        # LoRA 配置（ModernBERT 目标模块）
        self.use_lora = True
        self.lora_r = 16  # 增大 LoRA rank，提升容量
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.lora_target_modules = ["Wqkv", "Wo", "Wi"]  # attention + FFN 层

        # 推理阈值与距离限制
        self.ner_threshold = float(os.getenv("NER_THRESHOLD", "0.5"))
        self.re_threshold = float(os.getenv("RE_THRESHOLD", "0.5"))
        self.re_max_distance = int(os.getenv("RE_MAX_DISTANCE", "50"))

        # Label Studio 连接配置
        self.label_studio_url = os.getenv("LABEL_STUDIO_URL", "http://localhost:8080")
        self.model_cache_max_size = max(
            0, self._get_int_env("MODEL_CACHE_MAX_SIZE", 1)
        )

        # Tokenizer（延迟加载）
        self._tokenizer = None

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                # Label Studio annotations use offsets in the original text.
                # Length-changing tokenizer preprocessing would make offsets drift.
                do_text_preprocessing=False,
            )
        return self._tokenizer

    def update_model_paths(self, base_model_name: str = None, model_scope: str = None):
        """更新基础模型路径与输出路径。

        model_scope 为空时沿用 output/{base_model}/best_model；
        传入 schema_xxx 时使用 output/{base_model}/schema_xxx/best_model。
        """
        if base_model_name and base_model_name != self.base_model_name:
            self.base_model_name = base_model_name
            self._tokenizer = None
        self.model_scope = model_scope
        self.model_path = os.path.join(
            self.root_dir, "bert-base-model", self.base_model_name
        )
        if self.model_scope:
            self.output_dir = os.path.join(
                self.root_dir, "output", self.base_model_name, self.model_scope
            )
        else:
            self.output_dir = os.path.join(self.root_dir, "output", self.base_model_name)
        self.best_model_path = os.path.join(self.output_dir, "best_model")

    @staticmethod
    def _get_int_env(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except ValueError:
            return default

    def _build_label_mappings(self):
        """根据 ner_labels / re_labels 构建标签映射"""
        # NER 标签映射：O + B-XXX + I-XXX
        self.ner_label2id = {"O": 0}
        idx = 1
        for label in self.ner_labels:
            self.ner_label2id[f"B-{label}"] = idx
            idx += 1
            self.ner_label2id[f"I-{label}"] = idx
            idx += 1
        self.ner_id2label = {v: k for k, v in self.ner_label2id.items()}
        self.num_ner_labels = len(self.ner_label2id)

        # RE 标签映射：无关系 + 关系类型
        self.re_label2id = {"无关系": 0}
        for idx, label in enumerate(self.re_labels, start=1):
            self.re_label2id[label] = idx
        self.re_id2label = {v: k for k, v in self.re_label2id.items()}
        self.num_re_labels = len(self.re_label2id)

    def update_labels(self, ner_labels: list, re_labels: list):
        """更新标签配置并重建映射"""
        self.ner_labels = list(dict.fromkeys(label for label in ner_labels if label))
        self.re_labels = list(dict.fromkeys(label for label in re_labels if label and label != "无关系"))
        self._build_label_mappings()


# 全局配置实例
config = Config()
