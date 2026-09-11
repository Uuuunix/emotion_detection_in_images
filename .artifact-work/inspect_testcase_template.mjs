import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/Jiang/Desktop/??1?????????.xlsx";
const previewDir = "C:/Users/Jiang/AppData/Local/Temp/codex-artifact-testcases-jyh/previews";
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 12000,
  tableMaxRows: 15,
  tableMaxCols: 15,
  tableMaxCellChars: 120,
});
console.log(summary.ndjson);

const sheets = await workbook.inspect({
  kind: "sheet",
  include: "id,name",
  maxChars: 5000,
});
console.log("SHEETS");
console.log(sheets.ndjson);

await fs.mkdir(previewDir, { recursive: true });
const sheetNames = [];
for (const item of workbook.worksheets.items) {
  sheetNames.push(item.name);
  const preview = await workbook.render({
    sheetName: item.name,
    autoCrop: "all",
    scale: 1,
    format: "png",
  });
  const safeName = item.name.replace(/[\\/:*?"<>|]/g, "_");
  await fs.writeFile(
    `${previewDir}/${safeName}.png`,
    new Uint8Array(await preview.arrayBuffer()),
  );
}
console.log(JSON.stringify({ previewDir, sheetNames }));

const formulaCheck = await workbook.inspect({
  kind: "formula",
  sheetId: "Information????",
  range: "A1:K43",
  maxChars: 8000,
  options: { maxResults: 100 },
});
console.log("FORMULAS");
console.log(formulaCheck.ndjson);

for (const target of [
  { sheetId: "Information????", range: "A6:K43" },
  { sheetId: "Test Cases????", range: "A1:N9" },
]) {
  const styles = await workbook.inspect({
    kind: "computedStyle",
    sheetId: target.sheetId,
    range: target.range,
    maxChars: 10000,
  });
  console.log("STYLES", JSON.stringify(target));
  console.log(styles.ndjson);
}
