import json
from collections import Counter
from pathlib import Path
import openpyxl
SOURCE = Path("D:/OneDrive/03_Resource/Documents/Desktop/emr_data_1.xlsx")
OUT = Path("E:/emr_structured_annotation/output/emr_data_1_20261008")
OUT.mkdir(parents=True, exist_ok=True)
wb = openpyxl.load_workbook(SOURCE, read_only=True, data_only=True)
records, errors = [], []
for sheet in wb:
    for row_num, cells in enumerate(sheet.iter_rows(values_only=True), 1):
        raw = cells[0]
        if raw is None: continue
        try:
            obj = json.loads(raw)
            assert isinstance(obj, dict) and isinstance(obj.get("text"), str)
            records.append({"row": row_num, "sheet": sheet.title, "data": obj})
        except Exception as exc:
            errors.append({"row": row_num, "sheet": sheet.title, "cell_length": len(str(raw)), "error": str(exc), "raw": raw})
(OUT / "_parsed_records.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
(OUT / "invalid_source_rows.json").write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"valid":len(records),"invalid":[{k:v for k,v in x.items() if k != "raw"} for x in errors],"text_chars":sum(len(x["data"]["text"]) for x in records),"keys":Counter(k for x in records for k in x["data"]),"length_quantiles":sorted(len(x["data"]["text"]) for x in records)[::30],"unique_patients":len(set(x["data"].get("patient_id") for x in records)),"duplicate_text":len(records)-len(set(x["data"]["text"] for x in records))},ensure_ascii=False))

