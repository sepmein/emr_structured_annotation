import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const sourcePath = process.argv[2];
const input = await FileBlob.load(sourcePath);
const workbook = await SpreadsheetFile.importXlsx(input);

for (const [sheetId, range] of [
  ["2027年预算", "A1:AC9"],
  ["2027年预算", "A46:AC55"],
  ["绩效目标申报表", "A11:J18"],
]) {
  const result = await workbook.inspect({
    kind: "table",
    sheetId,
    range,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 30,
    tableMaxCellChars: 1000,
    maxChars: 40000,
  });
  console.log(`RANGE:${sheetId}!${range}`);
  console.log(result.ndjson);
}
