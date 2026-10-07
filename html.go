package main

import (
	"encoding/json"
	"os"
	"strings"
)

type htmlData struct {
	*Summary
	Points [][4]float64 `json:"points"` // [offset_sec, ttft_ms, concurrency, ok]
}

func WriteHTML(path string, sum *Summary, results []*Result) error {
	step := 1
	if len(results) > 20000 {
		step = len(results)/20000 + 1
	}
	d := htmlData{Summary: sum}
	for i := 0; i < len(results); i += step {
		r := results[i]
		if r.ErrKind == "canceled" {
			continue
		}
		ok := 0.0
		y := r.TTFTms
		if r.OK {
			ok = 1
		} else {
			y = r.E2Ems
		}
		d.Points = append(d.Points, [4]float64{r.OffsetSec, y, float64(r.Concurrency), ok})
	}
	b, err := json.Marshal(d)
	if err != nil {
		return err
	}
	js := strings.ReplaceAll(string(b), "</", "<\\/")
	return os.WriteFile(path, []byte(strings.Replace(htmlTemplate, "__DATA__", js, 1)), 0o644)
}

const htmlTemplate = `<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Load Test</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2dc;--accent:#2f6fde;--bad:#c9412f}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--card:#21211f;--fg:#ecebe6;--muted:#9a9993;--line:#34332f;--accent:#6b9cf0;--bad:#e5735f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1280px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:32px 0 12px}
.muted{color:var(--muted)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:22px;font-variant-numeric:tabular-nums}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,560px),1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}
.card h3{margin:0 0 8px;font-size:14px}
.chart{position:relative;height:280px}
.tbl{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;white-space:nowrap}
th,td{padding:6px 10px;text-align:right;border-bottom:1px solid var(--line)}
th{font-weight:600;color:var(--muted);font-size:12px}td:first-child,th:first-child{text-align:left}
ul.notes{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 12px 12px 32px;margin:0}
ul.notes li{margin:4px 0}
</style></head><body><main>
<h1>Нагрузочный тест LLM чат-бота</h1>
<div class="muted" id="sub"></div>
<div class="kpis" id="kpis"></div>
<h2>Выводы</h2><ul class="notes" id="notes"></ul>
<h2>Графики</h2>
<div class="grid">
 <div class="card"><h3>Пропускная способность vs одновременные диалоги</h3><div class="chart"><canvas id="cTput"></canvas></div></div>
 <div class="card"><h3>TTFT, мс</h3><div class="chart"><canvas id="cTTFT"></canvas></div></div>
 <div class="card"><h3>TPOT, мс на токен</h3><div class="chart"><canvas id="cTPOT"></canvas></div></div>
 <div class="card"><h3>TTFT p50 по номеру хода (рост контекста)</h3><div class="chart"><canvas id="cTurn"></canvas></div></div>
 <div class="card"><h3>Timeline: TTFT каждого запроса (ошибки — красным)</h3><div class="chart"><canvas id="cTime"></canvas></div></div>
 <div class="card" id="srvCard"><h3>Сервер vLLM: KV-cache и очередь</h3><div class="chart"><canvas id="cSrv"></canvas></div></div>
</div>
<h2>Ступени</h2><div class="tbl"><table id="tStages"></table></div>
<h2>Сервер vLLM</h2><div class="tbl"><table id="tSrv"></table></div>
<h2>Конфигурация</h2><div class="tbl"><table id="tCfg"></table></div>
</main>
<script>
const D = __DATA__;
const S = D.stages || [];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const pal = ['#2f6fde','#e07b39','#2a9d74','#b8509b','#c9a227','#5b8bb5','#c9412f','#7a6fd0','#4aa3a0','#8a8a3c'];
const fmt = (v,d=0) => v==null||isNaN(v) ? '—' : Number(v).toLocaleString('ru-RU',{maximumFractionDigits:d,minimumFractionDigits:d});
const conc = S.map(s=>s.concurrency);
document.getElementById('sub').textContent = D.meta.model + ' · ' + D.meta.url + ' · ' + new Date(D.meta.started_at).toLocaleString('ru-RU') + (D.meta.stopped_reason ? ' · остановлен: '+D.meta.stopped_reason : '');
const tot = S.reduce((a,s)=>({r:a.r+s.requests,e:a.e+s.errors,o:a.o+s.total_output_tokens}),{r:0,e:0,o:0});
const A = D.analysis;
[['Запросов', fmt(tot.r)], ['Ошибок', fmt(tot.r?tot.e/tot.r*100:0,1)+'%'], ['Макс. throughput', fmt(A.best_output_tok_per_sec)+' tok/s'],
 ['при диалогах', A.best_throughput_concurrency], ['Предел по SLO', A.max_concurrency_within_slo||'—'], ['Насыщение', A.saturation_concurrency||'—']]
 .forEach(([k,v])=>{const e=document.createElement('div');e.className='kpi';e.innerHTML='<span class="muted">'+k+'</span><b>'+v+'</b>';document.getElementById('kpis').appendChild(e)});
(A.notes||[]).forEach(n=>{const li=document.createElement('li');li.textContent=n;document.getElementById('notes').appendChild(li)});

Chart.defaults.color = css('--muted'); Chart.defaults.borderColor = css('--line');
Chart.defaults.font.family = 'system-ui,-apple-system,sans-serif';
const line = (id, datasets, opts={}) => new Chart(document.getElementById(id), {type:'line',
  data:{labels:opts.labels||conc, datasets:datasets.map((d,i)=>Object.assign({borderColor:pal[i%pal.length],backgroundColor:pal[i%pal.length],tension:.25,pointRadius:3},d))},
  options:Object.assign({maintainAspectRatio:false,interaction:{mode:'index',intersect:false},scales:{x:{title:{display:true,text:opts.x||'одновременных диалогов'}}}},opts.o||{})});
line('cTput',[{label:'Выход tok/s',data:S.map(s=>s.output_tok_per_sec),yAxisID:'y'},{label:'RPS',data:S.map(s=>s.rps),yAxisID:'y1'}],
  {o:{scales:{x:{title:{display:true,text:'одновременных диалогов'}},y:{position:'left',title:{display:true,text:'tok/s'}},y1:{position:'right',grid:{drawOnChartArea:false},title:{display:true,text:'RPS'}}}}});
line('cTTFT',['p50','p90','p95','p99'].map(p=>({label:p,data:S.map(s=>s.ttft_ms[p])})));
line('cTPOT',['p50','p95','p99'].map(p=>({label:p,data:S.map(s=>s.tpot_ms[p])})));
const maxTurn = Math.max(0,...S.flatMap(s=>(s.by_turn||[]).map(t=>t.turn)));
const turns = Array.from({length:maxTurn},(_,i)=>i+1);
line('cTurn', S.map(s=>{const m={};(s.by_turn||[]).forEach(t=>m[t.turn]=t.requests>t.errors?t.ttft_p50_ms:null);return {label:s.concurrency+' дл.',data:turns.map(t=>m[t]??null),spanGaps:true}}),
  {labels:turns,x:'номер хода в диалоге'});
const P = D.points||[];
new Chart(document.getElementById('cTime'),{type:'scatter',data:{datasets:[
  {label:'OK',data:P.filter(p=>p[3]).map(p=>({x:p[0],y:p[1]})),pointRadius:1.5,backgroundColor:css('--accent')+'99'},
  {label:'Ошибка (E2E)',data:P.filter(p=>!p[3]).map(p=>({x:p[0],y:p[1]})),pointRadius:2.5,backgroundColor:css('--bad')}]},
  options:{maintainAspectRatio:false,animation:false,scales:{x:{min:0,title:{display:true,text:'секунды от старта'}},y:{title:{display:true,text:'мс'}}}}});
const T = D.server_timeline||[];
if (T.length) {
  new Chart(document.getElementById('cSrv'),{type:'line',data:{datasets:[
    {label:'KV-cache %',data:T.map(p=>({x:p.t,y:p.kv??null})),borderColor:pal[0],yAxisID:'y',pointRadius:0},
    {label:'running',data:T.map(p=>({x:p.t,y:p.running??null})),borderColor:pal[2],yAxisID:'y1',pointRadius:0},
    {label:'waiting',data:T.map(p=>({x:p.t,y:p.waiting??null})),borderColor:pal[6],yAxisID:'y1',pointRadius:0}]},
    options:{maintainAspectRatio:false,animation:false,parsing:false,scales:{x:{type:'linear',min:0,title:{display:true,text:'секунды от старта'}},
      y:{min:0,max:100,title:{display:true,text:'KV %'}},y1:{position:'right',min:0,grid:{drawOnChartArea:false},title:{display:true,text:'запросов'}}}}});
} else document.getElementById('srvCard').style.display='none';

const tbl = (id, head, rows) => { const t=document.getElementById(id);
  t.innerHTML='<thead><tr>'+head.map(h=>'<th>'+h+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(c=>'<td>'+c+'</td>').join('')+'</tr>').join('')+'</tbody>'; };
tbl('tStages',['Диалоги','Время, с','Запросов','Ошибки','RPS','Вход tok/s','Выход tok/s','tok/s/поток p50','Goodput','TTFT p50','p95','p99','TPOT p50','p95','ITL p99','E2E p50','p95','Prompt ср.','max','Ток/чанк','Макс ход'],
  S.map(s=>[s.concurrency,fmt(s.duration_sec),s.requests,fmt(s.error_rate_pct,1)+'%',fmt(s.rps,2),fmt(s.input_tok_per_sec),fmt(s.output_tok_per_sec),fmt(s.stream_tok_per_sec.p50,1),fmt(s.goodput_pct,1)+'%',
    fmt(s.ttft_ms.p50),fmt(s.ttft_ms.p95),fmt(s.ttft_ms.p99),fmt(s.tpot_ms.p50,1),fmt(s.tpot_ms.p95,1),fmt(s.itl_ms.p99,1),fmt(s.e2e_ms.p50),fmt(s.e2e_ms.p95),
    fmt(s.prompt_tokens.mean),fmt(s.prompt_tokens.max),fmt(s.tokens_per_chunk,2),s.max_turn]));
tbl('tSrv',['Диалоги','Running max','Waiting max','KV max','KV ср.','Preempt','Prefix hit','Spec accept','Accept len','Prompt tok/s','Gen tok/s'],
  S.map(s=>{const v=s.server||{};return [s.concurrency,fmt(v.running_max),fmt(v.waiting_max),fmt(v.kv_usage_max_pct,1),fmt(v.kv_usage_mean_pct,1),fmt(v.preemptions),
    fmt(v.prefix_cache_hit_rate_pct,1),fmt(v.spec_decode_acceptance_pct,1),fmt(v.spec_decode_mean_accept_len,2),fmt(v.server_prompt_tok_per_sec),fmt(v.server_gen_tok_per_sec)]}));
tbl('tCfg',['Параметр','Значение'],Object.entries(D.config||{}).map(([k,v])=>[k,String(v).replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))]));
</script></body></html>
`
