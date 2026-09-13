import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath =
  process.argv[2] ??
  "/Users/sepmein/Nutstore Files/scdc/01 监测预警所/01-预算/01-市级财政/2027/一上经费申请/1-1-公共卫生综合监测预警项目2027年项目申报书（含绩效目标申报表）-20260611.xls";

let workbook;
try {
  const input = await FileBlob.load(sourcePath);
  workbook = await SpreadsheetFile.importXlsx(input);
} catch (error) {
  console.error("IMPORT_ERROR");
  console.error(error?.message ?? String(error));
  process.exit(1);
}

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 8000,
});
console.log("SHEETS");
console.log(sheets.ndjson);

const terms = ["科普", "宣传", "视频", "健康", "传播"];
for (const term of terms) {
  const hits = await workbook.inspect({
    kind: "match",
    searchTerm: term,
    options: { useRegex: false, maxResults: 200 },
    summary: `Matches for ${term}`,
    maxChars: 20000,
  });
  console.log(`MATCH:${term}`);
  console.log(hits.ndjson);
}
