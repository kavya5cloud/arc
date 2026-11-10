"""Build paper.pdf from paper.md. Fill in the three values below, then: python3 build.py"""
import pathlib, subprocess, markdown

AUTHORS = "Kavya Shree, Pandit Deendayal Energy University, Gujarat, India"
REPO = "github.com/kavya5cloud/arc"
KAGGLE = "[Kaggle code submission link]"
KAGGLE_RESULT = ("Results of the linked Kaggle submission are not included in this version. "
                 "All quantitative evidence in this paper comes from the controlled grid games in sections 4.2 to 4.5.")

HERE = pathlib.Path(__file__).parent
md = (HERE / "paper.md").read_text()
for k, v in {"{{AUTHORS}}": AUTHORS, "{{REPO}}": REPO, "{{KAGGLE}}": KAGGLE, "{{KAGGLE_RESULT}}": KAGGLE_RESULT}.items():
    md = md.replace(k, v)
body = markdown.markdown(md, extensions=["tables", "fenced_code"])
css = """
body { font-family: 'Times New Roman', Georgia, serif; font-size: 10.5pt; line-height: 1.4; color: #111; }
h1 { font-size: 17pt; text-align: center; margin: 0 0 6pt; }
h2 { font-size: 12.5pt; margin: 14pt 0 4pt; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
.authors { text-align: center; font-size: 11pt; margin: 0; }
.meta { text-align: center; font-size: 9pt; color: #444; margin: 2pt 0 10pt; }
.eq { text-align: center; font-size: 12pt; margin: 8pt 0; }
.caption { font-size: 9pt; color: #333; margin: 3pt 0 8pt; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 4pt 0; }
th, td { border-top: 0.5pt solid #999; border-bottom: 0.5pt solid #999; padding: 2.5pt 4pt; text-align: left; }
th { background: #f2f2f2; }
pre { background: #f6f6f6; padding: 6pt; font-size: 8pt; line-height: 1.3; white-space: pre-wrap; }
code { font-size: 8.5pt; }
img { width: 92%; display: block; margin: 6pt auto 0; }
li { margin: 2pt 0; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{body}</body></html>"
(HERE / "paper.html").write_text(html)


subprocess.run(["node", "-e", """
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage();
  await p.goto('file://' + process.argv[1] + '/paper.html');
  await p.pdf({ path: process.argv[1] + '/paper.pdf', format: 'Letter', printBackground: true,
    margin: { top: '0.8in', bottom: '0.8in', left: '0.85in', right: '0.85in' } });
  await b.close();
})();
""", str(HERE.resolve())], check=True)
print("wrote", HERE / "paper.pdf")
