import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "C:/Users/Jiang/AppData/Local/Temp/codex-artifact-testcases-jyh/template.xlsx";
const outputDir = "//wsl.localhost/Ubuntu-24.04/home/jiang/project/emotion_detection_in_images/outputs/testcases_jyh_20260911";
const outputPath = `${outputDir}/test_case_list_completed.xlsx`;
const previewDir = `${outputDir}/final_previews`;
const input = await FileBlob.load(inputPath);
const workbook = await SpreadsheetFile.importXlsx(input);
const infoSheet = workbook.worksheets.getItemAt(0);
const caseSheet = workbook.worksheets.getItemAt(1);
caseSheet.getRange("A2:N5").copyTo(caseSheet.getRange("A6:N9"), "all");
caseSheet.getRange("N6:N9").clear({ applyTo: "contents" });
const rows = [
["WB-07","upload()","GET 请求直接渲染上传页面","Medium","是","是","Flask 应用和模型已成功加载","client.get('/upload')","1. 创建 Flask test_client\n2. GET /upload\n3. 检查响应正文","HTTP 200；显示上传表单；不包含 Detected Emotion 和 Base64 结果图","HTTP 200；上传表单正常显示；未出现情绪结果和结果图片","OK","判定覆盖：request.method == 'POST' 的假分支"],
["WB-08","upload()","空文件名触发重定向且不解码","Medium","是","是","应用已加载；mock cv2.imdecode","image=(空字节流, 空文件名)","1. POST /upload\n2. image 字段使用空文件名\n3. 检查状态码、Location 和 mock 调用","HTTP 302；Location 指向 /upload；cv2.imdecode 不应调用","HTTP 302；重定向至 /upload；cv2.imdecode 调用次数为 0","OK","判定覆盖：if not file 的真分支"],
["WB-09","upload()","合法图片完整处理并回显 Base64 JPEG","High","是","是","mock 图片解码、情绪检测和 JPEG 编码","文件名 face.jpg；合法模拟图片字节；预测结果 Happy","1. POST 合法图片\n2. 检查解码和检测调用\n3. 提取 data URI\n4. Base64 解码并检查 JPEG 魔数","HTTP 200；显示 Happy；回显内容可 Base64 解码且以 FF D8 FF 开头","HTTP 200；显示 Happy；解码结果以 FF D8 FF 开头；完整调用链符合预期","OK","基本路径覆盖：upload() 最长正常路径"],
["WB-10","upload()","请求缺少 image 字段时的当前异常路径","Medium","是","是","Flask 应用保持当前实现","client.post('/upload', data={'other':'1'})","1. POST /upload\n2. 表单不包含 image 字段\n3. 检查响应状态","当前实现返回 HTTP 400，用于确认 BadRequestKeyError 异常路径","返回 HTTP 400，成功复现 request.files['image'] 引发的异常响应","OK","异常路径测试；确认缺陷 D-02 的当前表现"],
["WB-11","upload() / detect_faces_and_emotions()","图片解码失败时的当前 500 异常路径","High","是","是","mock cv2.imdecode 返回 None；关闭异常传播以检查 HTTP 响应","内容 b'not an image'；文件名 fake.jpg","1. POST 无效图片\n2. imdecode 返回 None\n3. None 进入检测函数并抛出 cv2.error\n4. 检查响应","当前实现返回 HTTP 500，并确认检测函数收到 None","返回 HTTP 500；detect_faces_and_emotions(None) 被调用一次","OK","异常路径测试；确认缺陷 D-01 的当前表现"],
["WB-12","start_detection()","首次启动时创建摄像头并启用视频流","High","是","是","全局 camera=None；mock cv2.VideoCapture","client.post('/start')","1. POST /start\n2. 检查 VideoCapture 参数和次数\n3. 检查全局 camera 与页面内容","HTTP 200；VideoCapture(0) 调用一次；页面包含 video_feed 和 Stop Detection","HTTP 200；VideoCapture(0) 调用一次；camera 指向 mock；流页面元素存在","OK","判定覆盖：camera is None 的真分支"],
["D-02","upload()","缺少 image 字段应被友好处理","High","是","是","当前实现仍使用 request.files['image']","POST /upload，仅包含 other=1","1. 提交缺少 image 的表单\n2. 期望 200 或 302\n3. pytest 使用 strict=True 的 xfail 标记","不得返回默认 400；应返回友好提示页面或重定向到 /upload","pytest XFAIL：实际返回 HTTP 400，未对缺少 image 字段进行友好处理","NG","错误推测法；严格缺陷回归测试。修复后会触发 XPASS(strict)"],
["D-01","upload()","无效图片应在进入人脸检测前被拒绝","High","是","是","mock cv2.imdecode 返回 None，并监控检测函数调用","内容 b'not an image'；文件名 fake.jpg","1. POST 无效图片\n2. 期望 200 或 302\n3. 断言检测函数未调用\n4. pytest 使用 strict=True 的 xfail 标记","不得将 None 传入检测函数；不得返回 500；应提示图片无效","pytest XFAIL：None 被传入检测函数；真实处理链会抛 cv2.error 并返回 HTTP 500","NG","错误推测法；严格缺陷回归测试。修复后会触发 XPASS(strict)"],
];
caseSheet.getRange("A2:M9").values = rows;
caseSheet.getRange("A2:M9").format.wrapText = true;
caseSheet.getRange("A2:M9").format.verticalAlignment = "top";
caseSheet.getRange("A2:A9").format.horizontalAlignment = "center";
caseSheet.getRange("D2:F9").format.horizontalAlignment = "center";
caseSheet.getRange("L2:L9").format.horizontalAlignment = "center";
caseSheet.getRange("A2:N9").format.rowHeight = 78;
const widths={A:12,B:22,C:30,D:12,E:10,F:12,G:30,H:32,I:42,J:42,K:42,L:10,M:34};
for(const [column,width] of Object.entries(widths)){caseSheet.getRange(`${column}1:${column}9`).format.columnWidth=width;}
caseSheet.freezePanes.freezeRows(1);
infoSheet.getRange("E7").values=[["main"]];
infoSheet.getRange("J7").values=[["内部"]];
infoSheet.getRange("E8").values=[["emotion_detection_in_images"]];
infoSheet.getRange("J8").values=[["EDI-UT"]];
infoSheet.getRange("E9").values=[["Jiang"]];
infoSheet.getRange("J9").values=[[new Date("2026-09-11T00:00:00Z")]];
infoSheet.getRange("J9").format.numberFormat="yyyy-mm-dd";
infoSheet.getRange("B16").values=[["本清单记录 upload() 与 start_detection() 的白盒测试 WB-07 至 WB-12，以及用于暴露 D-01、D-02 的严格缺陷回归测试。执行环境：Python 3.12.3、pytest 9.1.1。"]];
infoSheet.getRange("B20").values=[[new Date("2026-09-11T00:00:00Z")]];
infoSheet.getRange("B20").format.numberFormat="yyyy-mm-dd";
infoSheet.getRange("C20").values=[["1.00"]];
infoSheet.getRange("D20").values=[["WB-07~WB-12 / D-01 / D-02"]];
infoSheet.getRange("E20").values=[["录入白盒测试、缺陷回归测试及 2026-09-11 执行结果"]];
infoSheet.getRange("K20").values=[["Jiang"]];
const caseName=caseSheet.name.replace(/'/g,"''");
infoSheet.getRange("E13").formulas=[[`=COUNTA('${caseName}'!A2:A9)`]];
infoSheet.getRange("C38").formulas=[[`=COUNTIF('${caseName}'!L2:L9,"OK")`]];
infoSheet.getRange("E38").formulas=[["=IF(E13=0,0,C38/E13)"]];
infoSheet.getRange("C39").formulas=[[`=COUNTIF('${caseName}'!L2:L9,"POK")`]];
infoSheet.getRange("E39").formulas=[["=IF(E13=0,0,C39/E13)"]];
infoSheet.getRange("C40").formulas=[[`=COUNTIF('${caseName}'!L2:L9,"NG")`]];
infoSheet.getRange("E40").formulas=[["=IF(E13=0,0,C40/E13)"]];
infoSheet.getRange("C41").formulas=[[`=COUNTIF('${caseName}'!L2:L9,"NT")`]];
infoSheet.getRange("E41").formulas=[["=IF(E13=0,0,C41/E13)"]];
infoSheet.getRange("C43").formulas=[["=E13"]];
infoSheet.getRange("E38:E41").format.numberFormat="0.0%";
const changed=await workbook.inspect({kind:"table",range:`${caseSheet.name}!A1:N9`,include:"values,formulas",tableMaxRows:12,tableMaxCols:14,maxChars:24000});
console.log("CHANGED_RANGE"); console.log(changed.ndjson);
const errors=await workbook.inspect({kind:"match",searchTerm:"#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",options:{useRegex:true,maxResults:300},summary:"final formula error scan"});
console.log("FORMULA_ERRORS"); console.log(errors.ndjson);
await fs.mkdir(outputDir,{recursive:true}); await fs.mkdir(previewDir,{recursive:true});
for(const sheet of workbook.worksheets.items){const preview=await workbook.render({sheetName:sheet.name,autoCrop:"all",scale:1,format:"png"});const safeName=sheet.name.replace(/[\\/:*?"<>|]/g,"_");await fs.writeFile(`${previewDir}/${safeName}.png`,new Uint8Array(await preview.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(workbook); await output.save(outputPath);
console.log(JSON.stringify({outputPath,previewDir}));
