"""Generate a self-contained HTML annotation sheet for the human-eval pilot.
Loads human_eval/items/*.json (blinded S1/S2 + source) and human_eval/silver/*.json
(Opus draft) and emits human_eval/annotation_sheet.html: each item rendered blind,
pre-filled with the silver labels, every field editable, with a 'Download JSON'
button that serializes the human verdicts. The human validates/corrects; nothing
reveals which system is which (truth stays in _truth.json).
"""
import json, html
from pathlib import Path

HE = Path("human_eval")
items = {p.stem: json.load(open(p)) for p in sorted((HE / "items").glob("*.json"))}
silver = {p.stem: json.load(open(p)) for p in sorted((HE / "silver").glob("*.json"))}
cells = [c for c in items if c in silver]

# embed only what the page needs
data = {}
for c in cells:
    it, sv = items[c], silver[c]
    data[c] = {
        "dataset": it["dataset"], "seed": it["seed"],
        "source": it["source_sample"],
        "S1": it["S1_sentences"], "S2": it["S2_sentences"],
        "silver": sv,
    }

PAYLOAD = json.dumps(data)

HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Vec2Summ human-eval annotation</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;max-width:1000px;margin:20px auto;padding:0 16px;color:#222}
 h1{font-size:20px} h2{font-size:16px;border-top:2px solid #ddd;padding-top:14px;margin-top:26px}
 .src{max-height:220px;overflow:auto;border:1px solid #ddd;background:#fafafa;padding:8px;font-size:12.5px}
 .src div{padding:2px 0;border-bottom:1px solid #eee}
 .sum{display:flex;gap:16px;margin:10px 0}
 .col{flex:1;border:1px solid #ccc;border-radius:6px;padding:10px;background:#fff}
 .sent{padding:4px 0;border-bottom:1px solid #f0f0f0}
 .sent label{margin-right:8px} .err{margin-left:6px}
 .q{margin:6px 0} .q b{display:inline-block;min-width:130px}
 .silverhint{color:#888;font-size:12px;font-style:italic}
 textarea{width:100%;box-sizing:border-box;min-height:42px}
 .bar{position:sticky;top:0;background:#fff;border-bottom:2px solid #ddd;padding:10px 0;z-index:10}
 button{font-size:14px;padding:8px 14px;background:#3f6fb0;color:#fff;border:0;border-radius:6px;cursor:pointer}
 .rate{color:#3f6fb0;font-weight:bold}
</style></head><body>
<div class="bar">
 <h1>Vec2Summ human-eval (blind) &mdash; expert panel</h1>
 Your name/ID: <input id="annot" placeholder="e.g. R1" style="font-size:14px;padding:3px">
 <button onclick="dl()">Download annotations (JSON)</button>
 <span id="prog"></span>
 <p class="silverhint">Pre-filled with a model draft (silver). Read the source, then keep or change each answer. S1/S2 are blinded.</p>
</div>
<div id="root"></div>
<script>
const DATA = __PAYLOAD__;
const ERRS = ["", "entity_aspect","sentiment","extrinsic","overgeneralization","other"];
const root = document.getElementById('root');

function radio(name,val,checked,label){
 return `<label><input type="radio" name="${name}" value="${val}" ${checked?'checked':''}>${label}</label>`;
}
function covRow(cell,key,silverVal){
 return `<div class="q"><b>${key}</b> `+
  ["S1","S2","tie"].map(v=>radio(`${cell}__${key}`,v,silverVal===v,v)).join(" ")+
  ` <span class="silverhint">(silver: ${silverVal})</span></div>`;
}
function sentBlock(cell,side,sents,sv){
 const ps = sv.per_sentence||[];
 return `<div class="col"><b>Summary ${side}</b> &mdash; supported by source? <span class="rate" id="rate_${cell}_${side}"></span>`+
  sents.map((s,i)=>{
   const p = ps[i]||{supported:true};
   const sup = p.supported!==false;
   return `<div class="sent">${i+1}. ${escapeHtml(s)}<br>`+
    radio(`${cell}_${side}_${i}__sup`,"yes",sup,"supported")+
    radio(`${cell}_${side}_${i}__sup`,"no",!sup,"NOT supported")+
    `</div>`;
  }).join("")+`</div>`;
}
function escapeHtml(t){const d=document.createElement('div');d.textContent=t;return d.innerHTML;}

Object.keys(DATA).forEach(cell=>{
 const d=DATA[cell], sv=d.silver;
 const src = d.source.map(x=>`<div>${escapeHtml(x)}</div>`).join("");
 const el=document.createElement('div');
 el.innerHTML = `<h2>${cell} &mdash; ${d.dataset} (seed ${d.seed})</h2>
  <details><summary>Source sample (${d.source.length} docs) &mdash; click to read</summary><div class="src">${src}</div></details>
  <div class="sum">${sentBlock(cell,'S1',d.S1,sv.S1_faithfulness)}${sentBlock(cell,'S2',d.S2,sv.S2_faithfulness)}</div>
  <h3 style="font-size:14px;margin:6px 0">Coverage &mdash; which summary better covers the source sample?</h3>
  ${covRow(cell,'coverage_winner',sv.coverage_winner)}
  <div class="q"><b>notes (optional)</b><br><textarea name="${cell}__notes"></textarea><div class="silverhint">model's reason: ${escapeHtml(sv.coverage_rationale||'')}</div></div>`;
 root.appendChild(el);
});

function rate(cell,side){
 const d=DATA[cell][side];let s=0,n=d.length;
 for(let i=0;i<n;i++){const v=document.querySelector(`input[name="${cell}_${side}_${i}__sup"]:checked`);if(v&&v.value==='yes')s++;}
 return n?s/n:0;
}
function refresh(){
 let done=0,tot=Object.keys(DATA).length;
 Object.keys(DATA).forEach(cell=>{
  ['S1','S2'].forEach(side=>{document.getElementById(`rate_${cell}_${side}`).textContent=rate(cell,side).toFixed(2);});
  if(document.querySelector(`input[name="${cell}__coverage_winner"]:checked`))done++;
 });
 document.getElementById('prog').textContent=`  ${done}/${tot} items have a coverage verdict`;
}
document.addEventListener('change',refresh); refresh();

function dl(){
 const out={};
 Object.keys(DATA).forEach(cell=>{
  const g=k=>{const v=document.querySelector(`input[name="${cell}__${k}"]:checked`);return v?v.value:null;};
  const fa=side=>({per_sentence:DATA[cell][side].map((_,i)=>({i,
     supported:(document.querySelector(`input[name="${cell}_${side}_${i}__sup"]:checked`)||{}).value==='yes'})),
     faithfulness_rate:rate(cell,side)});
  out[cell]={coverage_winner:g('coverage_winner'),
   notes:(document.querySelector(`textarea[name="${cell}__notes"]`)||{}).value||"",
   S1_faithfulness:fa('S1'),S2_faithfulness:fa('S2')};
 });
 const who=(document.getElementById('annot').value||'anon').replace(/[^A-Za-z0-9_-]/g,'');
 const payload={annotator:who,annotations:out};
 const b=new Blob([JSON.stringify(payload,null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`human_annotations_${who}.json`;a.click();
}
</script></body></html>"""

out = HE / "annotation_sheet.html"
out.write_text(HTML.replace("__PAYLOAD__", PAYLOAD))
print(f"wrote {out} with {len(cells)} items (blinded, pre-filled from silver)")
