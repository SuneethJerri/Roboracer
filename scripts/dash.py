#!/usr/bin/env python3
"""R122: a local dashboard for AutoDRIVE runs, live or after the fact. Stdlib only.

Reads the runner's step logs (`scratch/o41/*_steps.jsonl`: one line per command, one per reset)
and, when there is one, the SPSA iterations CSV of the same run. Shows, per episode: the car on
the track coloured by speed, what the LIDAR saw with Follow-the-Gap's chosen gap and aim, the
commands against measured speed over time, the gains in force, and the search's iterations.

    .venv/bin/python scripts/dash.py [--port 8050]      # then open http://localhost:8050
"""
import argparse
import csv
import glob
import http.server
import json
import math
import os
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(ROOT, 'scratch', 'o41')
DATA = os.path.join(ROOT, 'data')


def runs():
    out = []
    for p in glob.glob(os.path.join(LOGS, '*_steps.jsonl')):
        st = os.stat(p)
        out.append({'run': os.path.basename(p)[:-len('_steps.jsonl')], 'size': st.st_size,
                    'mtime': st.st_mtime})
    return sorted(out, key=lambda r: -r['mtime'])


def safe(run):
    """A run name is a file stem in LOGS, nothing else."""
    if not run or '/' in run or '\\' in run or run.startswith('.'):
        raise ValueError(run)
    return run


def clean(o):
    """NaN and inf are not JSON a browser will parse: they go out as null."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def steps(run, start):
    """Complete lines appended since byte `start`, and where the next read starts."""
    with open(os.path.join(LOGS, safe(run) + '_steps.jsonl'), 'rb') as fh:
        fh.seek(start)
        buf = fh.read()
    end = buf.rfind(b'\n') + 1
    return {'next': start + end, 'lines': [clean(json.loads(x)) for x in buf[:end].splitlines() if x]}


def iters(run):
    p = os.path.join(LOGS, safe(run) + '_iters.csv')
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def centreline(run):
    slugs = [os.path.basename(p)[len('autodrive_centerline_'):-4]
             for p in glob.glob(os.path.join(DATA, 'autodrive_centerline_*.csv'))]
    hit = max([s for s in slugs if safe(run).startswith(s + '_')], key=len, default=None)
    if hit is None:
        return []
    with open(os.path.join(DATA, f'autodrive_centerline_{hit}.csv')) as fh:
        rows = list(csv.reader(fh))[1:]
    return [[float(r[0]), float(r[1])] for r in rows]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, kind='application/json'):
        data = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        try:
            if u.path == '/':
                return self.reply(PAGE, 'text/html; charset=utf-8')
            if u.path == '/api/runs':
                return self.reply(json.dumps(runs()))
            if u.path == '/api/steps':
                return self.reply(json.dumps(steps(q['run'], int(q.get('from', 0)))))
            if u.path == '/api/iters':
                return self.reply(json.dumps(iters(q['run'])))
            if u.path == '/api/centreline':
                return self.reply(json.dumps(centreline(q['run'])))
        except (KeyError, ValueError, OSError) as e:
            self.send_error(400, str(e))
            return
        self.send_error(404)


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>RoboRacer run dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#fafaf8;--panel:#fff;--ink:#1d1d1b;--quiet:#6b6b66;--rule:#e3e2dd;--acc:#2f6fdf;--good:#2e9d5b;--bad:#d1453b;--warn:#d68a16;--route:#8e44ad;color-scheme:light}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--panel:#1f1f1d;--ink:#ecebe6;--quiet:#9d9c95;--rule:#34332f;--acc:#6b9cff;--good:#4cc27e;--bad:#ff6b5e;--warn:#f0a83a;--route:#c08ae0;color-scheme:dark}}
*{box-sizing:border-box}body{margin:0;font:13px/1.4 system-ui,sans-serif;background:var(--bg);color:var(--ink);accent-color:var(--acc)}
::selection{background:color-mix(in srgb,var(--acc) 25%,transparent)}
:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
header{background:var(--bg);border-bottom:1px solid var(--rule);padding:8px 16px}
.bar{display:flex;gap:8px 16px;align-items:center;flex-wrap:wrap;min-height:34px}
header h1{font-size:15px;margin:0 8px 0 0}label{display:inline-flex;gap:6px;align-items:center}
select,button{font:inherit;color:inherit;background:var(--panel);border:1px solid var(--rule);border-radius:6px;padding:4px 8px;max-width:100%}
select{min-width:0;text-overflow:ellipsis}#run{max-width:min(440px,70vw)}
button{min-width:64px;cursor:pointer;transition:transform 160ms cubic-bezier(.23,1,.32,1)}button:active{transform:scale(.97)}
select,button{transition:border-color 150ms ease}@media (hover:hover) and (pointer:fine){button:hover,select:hover{border-color:var(--quiet)}}
@media (prefers-reduced-motion:reduce){button:active{transform:none}}
#scrub{flex:1;min-width:160px;margin:0}
#stamp{display:flex;gap:4px 14px;flex-wrap:wrap;font-variant-numeric:tabular-nums}#stamp>span{white-space:nowrap}
#stamp b{font-weight:600}.v{display:inline-block;min-width:5ch;text-align:right}
.flag{font-size:11px;font-weight:600;padding:1px 6px;border-radius:4px;border:1px solid}.flag.bad{color:var(--bad)}.flag.boxed{color:var(--quiet)}
.dot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--quiet);margin-right:5px;vertical-align:1px}.dot.on{background:var(--good)}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(400px,100%),1fr));gap:12px;padding:12px 16px}
section{background:var(--panel);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;min-width:0}
section h2{font-size:13px;margin:0 0 4px}canvas{width:100%;display:block}
.wide{grid-column:1/-1}.q{color:var(--quiet)}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
td,th{padding:3px 6px;border-bottom:1px solid var(--rule);text-align:right;white-space:nowrap}th{color:var(--quiet);font-weight:500;position:sticky;top:0;background:var(--panel)}td:first-child,th:first-child{text-align:left}
tr:last-child td{border-bottom:0}
.scroll{max-height:260px;overflow:auto}
.key{display:flex;flex-wrap:wrap;gap:2px 14px;margin:0 0 8px;font-size:12px}.key span{white-space:nowrap}
.key i{display:inline-block;width:12px;height:3px;margin-right:5px;vertical-align:middle;border-radius:1px}.key i.blk{height:10px;opacity:.35}.key i.dash{background:none!important;border-top:2px dashed;height:0}
.key i.ramp{width:56px;height:6px;background:linear-gradient(90deg,hsl(220 75% 50%),hsl(110 75% 50%),hsl(0 75% 50%))}
.key i.pt{width:6px;height:6px;border-radius:50%}
.empty{padding:24px 0;text-align:center}[hidden]{display:none!important}
@media (min-width:700px){header{position:sticky;top:0;z-index:1}}
</style></head><body>
<header><div class="bar"><h1>RoboRacer run dashboard</h1>
<label>Run <select id="run"></select></label>
<label>Episode <select id="ep"></select></label>
<label title="jump to the newest run and its latest episode, and keep following it"><input type="checkbox" id="live" checked> Follow live</label>
<span id="fresh" class="q"></span></div>
<div class="bar"><button id="play" aria-pressed="false">Play</button><input type="range" id="scrub" min="0" max="0" value="0" aria-label="Step in episode">
<output id="stamp" for="scrub"></output></div></header>
<main>
<section><h2>Track</h2><div class="key q"><span><i class="ramp"></i>speed 0 → 8 m/s</span><span><i style="background:var(--bad)"></i>wall contact (×)</span><span><i class="pt" style="background:var(--ink)"></i>car at cursor</span></div><canvas id="map" height="420"></canvas></section>
<section><h2>Where episodes stopped</h2><div class="key q"><span><i class="pt" style="background:var(--good)"></i>furthest point</span><span>× where it ended</span><span><i style="background:linear-gradient(90deg,hsl(0 70% 45%),hsl(120 70% 45%))"></i>progress 0 → 1</span><span>number = episode</span></div><canvas id="stops" height="420"></canvas></section>
<section><h2>What it sees <span class="q">LIDAR at the cursor</span></h2><div class="key q"><span><i class="blk" style="background:var(--quiet)"></i>FOV</span><span><i class="dash" style="color:var(--quiet)"></i>gap threshold <span id="thr"></span></span><span><i class="blk" style="background:var(--bad)"></i>safety bubble</span><span><i class="blk" style="background:var(--good)"></i>chosen gap</span><span><i style="background:var(--warn)"></i>aim</span><span><i style="background:var(--acc)"></i>steering sent</span><span><i class="dash" style="color:var(--route)"></i>route ahead</span></div><canvas id="lidar" height="420"></canvas></section>
<section class="wide"><h2>Inputs vs outputs over the episode</h2>
<div class="key q"><span><i style="background:var(--ink)"></i>speed m/s</span><span><i style="background:var(--warn)"></i>throttle target m/s (25 × throttle)</span><span><i style="background:var(--acc)"></i>steering × 5</span><span><i class="blk" style="background:var(--bad)"></i>reversing</span><span><i class="blk" style="background:var(--quiet)"></i>boxed in</span><span><i style="background:var(--bad)"></i>wall contact (top ticks)</span></div>
<canvas id="ts" height="220"></canvas></section>
<section><h2>Episode</h2><div id="sum"></div></section>
<section><h2>Gains in force</h2><div id="gn"></div></section>
<section><h2>SPSA iterations <span class="q">fine-tune runs</span></h2><canvas id="gains" height="180"></canvas><div class="scroll" id="iters"></div></section>
</main>
<script>
const $=id=>document.getElementById(id), css=v=>getComputedStyle(document.documentElement).getPropertyValue(v).trim();
let run=null, off=0, eps=[], cl=[], it=[], cur=0, playing=false;
function fit(c,H){// the backing store is w*r x h*r; the CSS size stays w x h, read once (it was re-read from the grown attribute)
 if(!c.dataset.h){c.dataset.h=c.getAttribute('height')}const r=devicePixelRatio||1, w=c.clientWidth, h=H?H(w):+c.dataset.h;c.style.height=h+'px';c.width=Math.round(w*r);c.height=Math.round(h*r);const g=c.getContext('2d');g.setTransform(r,0,0,r,0,0);g.clearRect(0,0,w,h);return [g,w,h]}
async function j(u){const r=await fetch(u);if(!r.ok)throw new Error(await r.text());return r.json()}
async function loadRuns(){const rs=(await j('/api/runs')).filter(r=>r.size>0);const s=$('run');const keep=s.value;s.innerHTML=rs.map(r=>`<option>${r.run}</option>`).join('');
 if(!rs.length){$('fresh').textContent='';$('sum').innerHTML='<p class="q empty">No step logs in scratch/o41 yet. Start a run and it appears here.</p>';return}
 if($('live').checked&&rs.length)s.value=rs[0].run;else if(keep&&rs.some(r=>r.run===keep))s.value=keep;
 const m=rs.find(r=>r.run===s.value),ago=m?Date.now()/1000-m.mtime:Infinity;
 $('fresh').innerHTML=ago<10?'<span class="dot on"></span>writing now':`<span class="dot"></span>last write ${ago<3600?Math.round(ago/60)+' min':new Date(m.mtime*1000).toLocaleString([],{dateStyle:'short',timeStyle:'short'})}${ago<3600?' ago':''}`;
 if(s.value!==run)await pick(s.value)}
async function pick(r){run=r;off=0;eps=[];cur=0;if(!r)return;cl=await j('/api/centreline?run='+encodeURIComponent(r));await poll();}
async function poll(){if(!run)return;const d=await j(`/api/steps?run=${encodeURIComponent(run)}&from=${off}`);off=d.next;
 for(const l of d.lines){if(l.gains){eps.push({t0:l.t,gains:l.gains,rows:[]});continue}if(l.x==null||l.y==null||l.v==null)continue;if(!eps.length)eps.push({t0:l.t,gains:{},rows:[]});eps.at(-1).rows.push(l)}
 if(d.lines.length){const s=$('ep'),keep=s.value;s.innerHTML=eps.map((e,i)=>`<option value="${i}">${i}: ${outcome(e)}</option>`).join('');
  if($('live').checked){s.value=eps.length-1}else if(keep!=='')s.value=keep}
 it=await j('/api/iters?run='+encodeURIComponent(run));draw()}
function ep(){return eps[+$('ep').value]||null}
function draw(){const e=ep();if(!e){return}const n=e.rows.length;$('scrub').max=Math.max(0,n-1);
 if($('live').checked||cur>=n)cur=n-1;$('scrub').value=cur;const r=e.rows[cur];if(!r)return;
 const v=(k,x,u)=>`<span><span class="q">${k}</span> <b class="v">${x}</b>${u?' '+u:''}</span>`;
 $('stamp').innerHTML=v('t',(r.t-e.t0).toFixed(1),'s')+v('speed',r.v.toFixed(2),'m/s')+v('steer',r.st.toFixed(2))+v('throttle',r.th.toFixed(3))+v('step',`${cur+1}/${n}`)
  +(r.rev?'<span class="flag bad">reversing</span>':'')+(r.ftg&&r.ftg.boxed?'<span class="flag boxed">boxed in</span>':'');
 drawMap(e,r);drawStops();drawLidar(r);drawTs(e);drawSum(e);drawIters()}
function best(e){const p=e.rows.filter(q=>q.prog!=null).map(q=>q.prog);return p.length?Math.max(...p):null}
const fmt=b=>b==null?'not logged':b.toFixed(3);
function outcome(e){const R=e.rows;if(!R.length)return 'no data';const laps=(R.at(-1).laps||0)-(R[0].laps||0),d=R.at(-1).t-e.t0,
 longest=Math.max(...eps.map(x=>x.rows.length?x.rows.at(-1).t-x.t0:0));
 return `furthest ${fmt(best(e))}, `+(laps>0?`${laps} lap(s)`:d>=longest-1.5?'ran to the time limit':'stopped early')}
function contacts(e){const out=[];e.rows.forEach((q,i)=>{if(i&&q.col!=null&&q.col>e.rows[i-1].col)out.push(q)});return out}
function drawMap(e,r){const [g,w,h]=fit($('map'));const pts=cl.concat(e.rows.map(q=>[q.x,q.y]));if(!pts.length)return;
 let xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const s=Math.min((w-20)/(x1-x0||1),(h-20)/(y1-y0||1)),X=x=>10+(x-x0)*s,Y=y=>h-10-(y-y0)*s;
 g.strokeStyle=css('--rule');g.lineWidth=6;g.beginPath();cl.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();
 g.lineWidth=2;for(let i=1;i<e.rows.length;i++){const a=e.rows[i-1],b=e.rows[i];if(Math.hypot(b.x-a.x,b.y-a.y)>0.5)continue;const f=Math.min(1,b.v/8);
  g.strokeStyle=`hsl(${220-220*f},75%,50%)`;g.beginPath();g.moveTo(X(a.x),Y(a.y));g.lineTo(X(b.x),Y(b.y));g.stroke()}
 g.strokeStyle=css('--bad');g.lineWidth=2;for(const q of contacts(e)){g.beginPath();g.moveTo(X(q.x)-5,Y(q.y)-5);g.lineTo(X(q.x)+5,Y(q.y)+5);g.moveTo(X(q.x)+5,Y(q.y)-5);g.lineTo(X(q.x)-5,Y(q.y)+5);g.stroke()}g.lineWidth=1;
 g.fillStyle=css('--ink');g.beginPath();g.arc(X(r.x),Y(r.y),5,0,7);g.fill();
 const k=Math.max(0,cur-6),p=e.rows[k];const hd=Math.atan2(r.y-p.y,r.x-p.x);if(Math.hypot(r.y-p.y,r.x-p.x)>0.02){g.strokeStyle=css('--ink');g.beginPath();g.moveTo(X(r.x),Y(r.y));g.lineTo(X(r.x)+18*Math.cos(hd),Y(r.y)-18*Math.sin(hd));g.stroke()}}
function drawStops(){const [g,w,h]=fit($('stops'));const pts=cl.slice();eps.forEach(e=>e.rows.forEach(q=>pts.push([q.x,q.y])));if(!pts.length)return;
 const xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]),x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const s=Math.min((w-20)/(x1-x0||1),(h-20)/(y1-y0||1)),X=x=>10+(x-x0)*s,Y=y=>h-10-(y-y0)*s;
 g.strokeStyle=css('--rule');g.lineWidth=6;g.beginPath();cl.forEach((p,i)=>i?g.lineTo(X(p[0]),Y(p[1])):g.moveTo(X(p[0]),Y(p[1])));g.stroke();g.lineWidth=1;
 const placed=[];g.font='11px system-ui,sans-serif';
 eps.forEach((e,i)=>{if(!e.rows.length)return;const b=best(e)||0,k=e.rows.reduce((m,q,j)=>(q.prog||0)>(e.rows[m].prog||0)?j:m,0),f=e.rows[k],l=e.rows.at(-1),col=`hsl(${120*Math.min(1,b)},70%,45%)`;
  g.strokeStyle=col;g.fillStyle=col;g.beginPath();g.arc(X(f.x),Y(f.y),5,0,7);g.fill();g.lineWidth=2;g.beginPath();g.moveTo(X(l.x)-4,Y(l.y)-4);g.lineTo(X(l.x)+4,Y(l.y)+4);g.moveTo(X(l.x)+4,Y(l.y)-4);g.lineTo(X(l.x)-4,Y(l.y)+4);g.stroke();g.lineWidth=1;
  let lx=X(f.x)+7,ly=Y(f.y)-5;while(placed.some(([px,py])=>Math.abs(px-lx)<16&&Math.abs(py-ly)<12))ly+=12;// ponytail: greedy downward nudge, fine for tens of episodes
  placed.push([lx,ly,i])});g.fillStyle=css('--ink');g.strokeStyle=css('--panel');g.lineWidth=3;for(const [x,y,i] of placed){g.strokeText(i,x,y);g.fillText(i,x,y)}g.lineWidth=1}
function drawLidar(r){const a0=r.amin,a1=r.amin+r.inc*Math.max(0,r.scan.length-1),back=Math.max(0,-Math.cos(a0),-Math.cos(a1));
 const [g,w,h]=fit($('lidar'),w=>Math.min(420,Math.round((w/2-12)*(1+back)+28)));const R=Math.min(w/2-12,(h-28)/(1+back)),cx=w/2,cy=12+R,sc=R/10;// fit the whole scan, rear beams included
 const P=(a,d)=>[cx-d*sc*Math.sin(a),cy-d*sc*Math.cos(a)];
 g.strokeStyle=css('--rule');for(const d of [2,4,6,8,10]){g.beginPath();g.arc(cx,cy,d*sc,-Math.PI/2-a1,-Math.PI/2-a0);g.stroke()}
 
 const f=r.ftg||{};if(f.fov){g.fillStyle=css('--rule');g.globalAlpha=.5;g.beginPath();g.moveTo(cx,cy);for(let a=-f.fov;a<=f.fov;a+=0.02)g.lineTo(...P(a,10));g.closePath();g.fill();g.globalAlpha=1}
 if(f.thresh&&f.fov){g.strokeStyle=css('--quiet');g.setLineDash([4,4]);g.lineWidth=1.5;g.beginPath();for(let a=-f.fov;a<=f.fov+1e-9;a+=0.02){const [x,y]=P(a,f.thresh);a>-f.fov?g.lineTo(x,y):g.moveTo(x,y)}g.stroke();g.setLineDash([]);g.lineWidth=1}
 $('thr').textContent=f.thresh?f.thresh.toFixed(2)+' m':'';
 if(f.bubble){const [ba,bh,bd]=f.bubble;g.fillStyle=css('--bad');g.globalAlpha=.3;g.beginPath();g.moveTo(cx,cy);for(let a=ba-bh;a<=ba+bh;a+=0.005)g.lineTo(...P(a,Math.min(bd,10)));g.closePath();g.fill();g.globalAlpha=1}
 if(f.gap){g.fillStyle=css('--good');g.globalAlpha=.25;g.beginPath();g.moveTo(cx,cy);for(let a=f.gap[0];a<=f.gap[1];a+=0.01)g.lineTo(...P(a,10));g.closePath();g.fill();g.globalAlpha=1}
 g.fillStyle=css('--ink');r.scan.forEach((d,i)=>{const a=r.amin+r.inc*(i+0.45);const [x,y]=P(a,Math.min(d,10));g.fillRect(x-1.5,y-1.5,3,3)});
 const ray=(a,c)=>{g.strokeStyle=c;g.lineWidth=2;g.beginPath();g.moveTo(cx,cy);g.lineTo(...P(a,9));g.stroke();g.lineWidth=1};
 if(f.route!=null){g.setLineDash([6,4]);ray(f.route,css('--route'));g.setLineDash([])}if(f.aim!==undefined)ray(f.aim,css('--warn'));ray(r.st*0.5236,css('--acc'));
 g.fillStyle=css('--quiet');g.font='11px system-ui,sans-serif';g.textAlign='center';g.strokeStyle=css('--panel');g.lineWidth=3;for(const d of [2,6,10]){g.strokeText(d+' m',cx+d*sc,cy+14);g.fillText(d+' m',cx+d*sc,cy+14)}g.lineWidth=1;g.textAlign='start';
 g.fillStyle=css('--ink');g.beginPath();g.moveTo(cx,cy-8);g.lineTo(cx-5,cy+4);g.lineTo(cx+5,cy+4);g.fill()}
function drawTs(e){const [g,w,h]=fit($('ts'));const R=e.rows;if(R.length<2)return;const T=R.at(-1).t-e.t0||1,X=t=>40+(t-e.t0)/T*(w-50);
 const vmax=Math.max(8,...R.map(r=>Math.max(r.v,25*r.th))),Y=v=>h-20-(v+5)/(vmax+5)*(h-30);
 const kind=r=>r.rev?'--bad':r.ftg&&r.ftg.boxed?'--quiet':null;g.globalAlpha=.18;// one rect per span, not per step: overlapping steps striped
 for(let i=1,s=0;i<=R.length;i++){if(i<R.length&&kind(R[i])===kind(R[s]))continue;const k=kind(R[s]);if(k){g.fillStyle=css(k);g.fillRect(X(R[Math.max(0,s-1)].t),10,X(R[i-1].t)-X(R[Math.max(0,s-1)].t),h-30)}s=i}g.globalAlpha=1;
 g.font='11px system-ui,sans-serif';g.strokeStyle=css('--rule');g.fillStyle=css('--quiet');for(let v=0;v<=vmax;v+=vmax>12?4:2){g.beginPath();g.moveTo(40,Y(v));g.lineTo(w-10,Y(v));g.stroke();g.fillText(v,8,Y(v)+4)}
 g.textAlign='center';for(let t=0,dt=T>240?60:T>120?30:T>60?10:5;t<=T;t+=dt){g.fillText(t+' s',X(e.t0+t),h-4)}g.textAlign='start';
 const line=(f,c)=>{g.strokeStyle=c;g.lineWidth=1.5;g.beginPath();R.forEach((r,i)=>i?g.lineTo(X(r.t),Y(f(r))):g.moveTo(X(r.t),Y(f(r))));g.stroke()};
 line(r=>r.st*5,css('--acc'));line(r=>25*r.th,css('--warn'));line(r=>r.v,css('--ink'));
 g.strokeStyle=css('--bad');g.lineWidth=2;for(const q of contacts(e)){g.beginPath();g.moveTo(X(q.t),10);g.lineTo(X(q.t),24);g.stroke()}g.lineWidth=1;
 const c=R[Math.min(cur,R.length-1)];g.strokeStyle=css('--ink');g.setLineDash([3,3]);g.beginPath();g.moveTo(X(c.t),10);g.lineTo(X(c.t),h-20);g.stroke();g.setLineDash([])}
function drawSum(e){const R=e.rows,dur=R.length?R.at(-1).t-e.t0:0,rev=R.filter(r=>r.rev).length,box=R.filter(r=>r.ftg&&r.ftg.boxed).length,vm=Math.max(0,...R.map(r=>r.v));
 const gains=Object.entries(e.gains).map(([k,v])=>`<tr><td>${k}</td><td>${(+v).toFixed(3)}</td></tr>`).join('');
 $('gn').innerHTML=gains?`<table>${gains}</table>`:'<p class="q empty">No gains logged for this episode.</p>';
 $('sum').innerHTML=`<table><tr><td>Duration</td><td>${dur.toFixed(1)} s</td></tr><tr><td>Steps</td><td>${R.length}</td></tr><tr><td>Top speed</td><td>${vm.toFixed(2)} m/s</td></tr><tr><td>Steps reversing</td><td>${rev}</td></tr><tr><td>Steps boxed in</td><td>${box}</td></tr><tr><td>Furthest progress</td><td>${fmt(best(e))}</td></tr><tr><td>Wall contacts</td><td>${contacts(e).length}</td></tr><tr><td>Outcome</td><td>${outcome(e).split(', ').slice(1).join(', ')}</td></tr></table>`}
function drawIters(){if(!it.length){$('gains').hidden=true;$('iters').innerHTML='<p class="q empty">No SPSA iterations for this run. Only fine-tune runs write an iterations file.</p>';return}
 $('gains').hidden=false;
 const names=Object.keys(it[0]).filter(k=>/^[A-Z_]+$/.test(k));$('iters').innerHTML='<table><tr><th>iter</th><th>L+</th><th>L-</th><th>feas</th><th>G</th><th>lam</th>'+names.map(n=>`<th>${n}</th>`).join('')+'</tr>'+
  it.map(r=>`<tr><td>${r.iter}</td><td>${(+r.L_plus).toFixed(2)}</td><td>${(+r.L_minus).toFixed(2)}</td><td>${r.feas_plus}${r.feas_minus}</td><td>${(+r.G_hat).toFixed(3)}</td><td>${(+r.lam).toFixed(2)}</td>`+names.map(n=>`<td>${(+r[n]).toFixed(3)}</td>`).join('')+'</tr>').join('')+'</table>';
 const [g,w,h]=fit($('gains')),N=it.length,X=i=>30+i/Math.max(1,N-1)*(w-40);const col=['--acc','--warn','--good','--bad','--ink','--quiet'];
 names.forEach((n,k)=>{const v=it.map(r=>+r[n]),lo=Math.min(...v),hi=Math.max(...v),Y=x=>h-15-(hi>lo?(x-lo)/(hi-lo):0.5)*(h-30);g.strokeStyle=css(col[k%col.length]);g.globalAlpha=k<6?1:.5;g.beginPath();v.forEach((x,i)=>i?g.lineTo(X(i),Y(x)):g.moveTo(X(i),Y(x)));g.stroke()});g.globalAlpha=1;
 g.fillStyle=css('--quiet');g.fillText('each gain scaled to its own range; table below has the values',30,10)}
$('run').onchange=e=>{$('live').checked=e.target.selectedIndex===0;pick(e.target.value)};$('ep').onchange=()=>{$('live').checked=false;cur=0;draw()};
$('scrub').oninput=e=>{$('live').checked=false;cur=+e.target.value;draw()};$('live').onchange=draw;
$('play').onclick=()=>{const e=ep();if(!playing&&e&&cur>=e.rows.length-1)cur=0;playing=!playing;$('play').textContent=playing?'Pause':'Play';$('play').setAttribute('aria-pressed',playing);if(playing)$('live').checked=false};
let pt=null,pw=0;// playback in episode time, 1x real time: frames advance by the clock, not a fixed step count
(function tick(now){requestAnimationFrame(tick);const e=ep(),dt=Math.min(0.1,(now-pw)/1000);pw=now;if(!playing||!e||!e.rows.length)return pt=null;const R=e.rows;
 if(pt==null||Math.abs(R[cur].t-pt)>0.2)pt=R[cur].t;pt+=dt;let k=cur;while(k<R.length-1&&R[k+1].t<=pt)k++;
 if(k!==cur){cur=k;draw()}if(cur>=R.length-1){playing=false;$('play').textContent='Play';$('play').setAttribute('aria-pressed',false)}})(0);
setInterval(()=>{if($('live').checked)poll()},1000);setInterval(loadRuns,3000);addEventListener('resize',draw);loadRuns();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--port', type=int, default=8050)
    a = ap.parse_args()
    print(f'dashboard on http://localhost:{a.port}  (logs: {LOGS})', flush=True)
    http.server.ThreadingHTTPServer(('127.0.0.1', a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
