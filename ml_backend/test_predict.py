"""快速本地测试：不启动 HTTP server，直接调用 predict()"""
from ml_backend.model import PneumoniaNERModel

# 模拟一条 Label Studio task
SAMPLE_TASK = {
    "id": 1,
    "data": {
        "emr_activity_info": [
            {
                "activity_time": "2024-01-10 09:00",
                "chief_complaint": "发热3天，伴咳嗽、气促",
                "present_illness_his": (
                    "患儿3天前无明显诱因出现发热，体温最高39.2℃，"
                    "伴咳嗽，有痰，气促明显，无惊厥，无意识障碍。"
                    "既往有哮喘病史。否认结核接触史。"
                    "疑似肺炎支原体感染，不除外流感病毒感染。"
                ),
            }
        ]
    },
}


def main():
    print("初始化模型（首次运行会下载模型，需要几分钟）...")
    model = PneumoniaNERModel()
    model.setup()

    print("\n直接调用 GLiNER2 查看原始返回...\n")
    text = (
        "患儿3天前无明显诱因出现发热，体温最高39.2℃，"
        "伴咳嗽，有痰，气促明显，无惊厥，无意识障碍。"
        "既往有哮喘病史。否认结核接触史。"
        "疑似肺炎支原体感染，不除外流感病毒感染。"
    )
    raw = model.gliner.extract(
        text,
        model._schema,
        include_confidence=True,
        include_spans=True,
    )
    entities_dict = raw.get("entities", {})
    count = sum(len(spans) for spans in entities_dict.values())
    print(f"原始实体数: {count}")
    for label, spans in entities_dict.items():
        for span in spans:
            print(f"  text={span['text']!r:20s}  score={span.get('confidence', 0):.3f}  label={label[:40]!r}")

    print("\n开始预测...\n")
    results = model.predict([SAMPLE_TASK])

    for pred in results:
        entities = pred.get("result", [])
        print(f"共识别 {len(entities)} 个实体，平均置信度 {pred.get('score', 0):.3f}\n")
        for ent in sorted(entities, key=lambda x: x["value"]["start"]):
            v = ent["value"]
            print(
                f"  [{v['start']:3d}:{v['end']:3d}]  "
                f"{v['text']:<20s}  "
                f"→ {v['labels'][0]:<20s}  "
                f"(from_name={ent['from_name']}, score={ent['score']:.3f})"
            )


if __name__ == "__main__":
    main()
