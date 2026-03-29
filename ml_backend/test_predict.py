"""
Debug 专用测试脚本：深入检查模型置信度和文本偏移。
用法：uv run python -m ml_backend.test_predict
"""
import json
from ml_backend.model import PneumoniaNERModel
from ml_backend.prompts import GLINER2_LABELS

# 构造一个真实的、包含多个待识别实体的测试用例
SAMPLE_TASK = {
    "id": "DEBUG_001",
    "data": {
        "chief_complaint_text": (
            "患儿3天前无明显诱因出现发热，体温最高39.2℃，"
            "伴咳嗽，有痰，气促明显，无惊厥，无意识障碍。"
            "既往有哮喘病史。否认结核接触史。"
            "疑似肺炎支原体感染，不除外流感病毒感染。"
        )
    },
}

def debug_run():
    print("=== [DEBUG MODE] 开始初始化模型 ===")
    model = PneumoniaNERModel()
    model.setup()

    # 1. 正常预测流程
    print("\n--- 1. 标准预测流程测试 (当前阈值) ---")
    results = model.predict([SAMPLE_TASK])
    
    # 2. 深度探测模式：使用极低阈值探测模型“犹豫”的标签
    print("\n--- 2. 深度探测模式 (探测阈值=0.1) ---")
    print("正在检查那些置信度较低但模型其实识别出来的实体...")
    
    # 临时创建一个低阈值的 schema
    low_threshold_schema = model.gliner.create_schema().entities({
        label: {"description": desc, "threshold": 0.1} 
        for label, desc in GLINER2_LABELS.items()
    })
    
    text = SAMPLE_TASK["data"]["chief_complaint_text"]
    raw_low = model.gliner.extract(
        text, 
        low_threshold_schema, 
        include_confidence=True, 
        include_spans=True
    )
    
    entities = raw_low.get("entities", {})
    
    # 打印表头
    print(f"\n  {'状态':<8} | {'标签':<12} | {'分数':<8} | {'文本':<20}")
    print("  " + "-" * 55)

    for label, spans in entities.items():
        # 获取该标签在模型中的正式阈值（用于对比）
        formal_threshold = 0.4 
        
        for span in spans:
            conf = span.get('confidence', 0)
            status = "✅ [通过]" if conf >= formal_threshold else "❌ [被过滤]"
            print(f"  {status:<8} | {label:<12} | {conf:.4f} | '{span['text']}'")

    print("\n=== [DEBUG] 检查完毕 ===")

if __name__ == "__main__":
    debug_run()
