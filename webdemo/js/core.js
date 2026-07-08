/* ============================================================
   core.js - shared utilities, verified data, nav, charts
   ============================================================ */
'use strict';

/* ---------- tiny DOM helpers ---------- */
const $  = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => Array.from(r.querySelectorAll(s));
const el = (tag, attrs={}, kids=[]) => {
  const n = document.createElementNS(
    tag === 'svg' || tag === 'g' || tag === 'path' || tag === 'rect' || tag === 'circle' ||
    tag === 'line' || tag === 'text' || tag === 'polyline' || tag === 'polygon' ||
    tag === 'foreignObject'
      ? 'http://www.w3.org/2000/svg' : 'http://www.w3.org/1999/xhtml', tag);
  for (const k in attrs) {
    if (k === 'text') n.textContent = attrs[k];
    else if (k === 'html') n.innerHTML = attrs[k];
    else n.setAttribute(k, attrs[k]);
  }
  (Array.isArray(kids) ? kids : [kids]).forEach(c => c && n.appendChild(c));
  return n;
};
const clamp = (v,a,b) => Math.max(a, Math.min(b, v));
const lerp  = (a,b,t) => a + (b-a)*t;
const fmt   = (v,d=3) => Number(v).toFixed(d);

/* ---------- palette (mirrors css tokens) ---------- */
const C = {
  bio:'#f4b13c', bioGlow:'rgba(244,177,60,0.55)', ctrl:'#8592a6',
  blue:'#4b9bff', violet:'#9a8cf0', red:'#ec6a6a', green:'#37c98a', teal:'#2fd4c6',
  ink:'#eef2f8', dim:'#aab4c6', mute:'#6c7789', grid:'#2c2c2a', surf:'#10151f',
};

/* ============================================================
   VERIFIED RESULT DATA  (copied from repo result files + paper)
   ============================================================ */
const DATA = {
  // Table 1 - CX heading-bump angular error (rad, lower better), mean/3 seeds
  cxHeading: {
    T: [50, 100, 200],
    frozen: {
      cx_bpu:        [1.054, 1.299, 1.441],
      weight_shuffle:[1.060, 1.309, 1.444],
      degree_shuffle:[1.200, 1.373, 1.478],
      random:        [1.160, 1.353, 1.467],
      no_recurrence: [1.167, 1.357, 1.469],
    },
    trainable: {
      cx_bpu:        [0.435, 0.801, 1.134],
      weight_shuffle:[0.451, 0.850, 1.186],
      degree_shuffle:[0.499, 0.904, 1.212],
      random:        [0.524, 0.977, 1.268],
      no_recurrence: [1.154, 1.352, 1.466],
    },
  },
  // Table 3 - MQAR test recall accuracy (higher better)
  mqar: [
    { model:'Attention + short-conv', acc:1.000, note:'SOTA ceiling',           kind:'ceiling' },
    { model:'Connectome · 1000 ep',   acc:0.995, note:'longer training',        kind:'bio' },
    { model:'Connectome · 200 ep',    acc:0.925, sd:0.003, note:'5 seeds',      kind:'bio' },
    { model:'Weight shuffle',         acc:0.914, sd:0.003, note:'MB topology',  kind:'shuffle' },
    { model:'Random sparse',          acc:0.836, sd:0.008, note:'size-matched', kind:'ctrl' },
    { model:'Degree-preserving',      acc:0.768, sd:0.033, note:'degree-matched',kind:'ctrl' },
    { model:'Chance',                 acc:0.031, note:'vocab 32',               kind:'chance' },
  ],
  // Table 2 - MB odor-valence reversal
  reversal: [
    { model:'Hemibrain seeded', query:0.9966, initial:0.9985, reversal:0.9946, loss:0.0098, epoch:17, kind:'bio' },
    { model:'Weight shuffle',   query:0.9963, initial:0.9986, reversal:0.9940, loss:0.0105, epoch:18, kind:'shuffle' },
    { model:'Random sparse',    query:0.9857, initial:0.9927, reversal:0.9787, loss:0.0387, epoch:53, kind:'ctrl' },
  ],
  // Split-Odor continual learning (frozen)
  continual: [
    { model:'Connectome (frozen)', acc:0.819, sd:0.001, forget:0.223, kind:'bio' },
    { model:'Weight shuffle',      acc:0.787, sd:0.004, forget:0.262, kind:'shuffle' },
    { model:'Random (frozen)',     acc:0.748, sd:0.007, forget:0.308, kind:'ctrl' },
  ],
  // Table 5 - optic flow RMSE by training-data fraction (lower better)
  opticFlow: {
    fraction: [5, 10, 20, 50, 100],
    families: {
      sparse_connectome:[0.2196,0.1851,0.1697,0.1471,0.1317],
      sparse_random:    [0.2057,0.1901,0.1732,0.1629,0.1425],
      pruned_connectome:[0.2036,0.1807,0.1690,0.1488,0.1317],
      pruned_random:    [0.2044,0.1913,0.1724,0.1554,0.1392],
      dense_connectome: [0.1909,0.1697,0.1546,0.1356,0.1251],
      dense_random:     [0.1861,0.1705,0.1549,0.1359,0.1254],
    },
  },
  // Region × task advantage matrix - real numbers for ALL 15 cells (from scripts/figures/plot_region_task_heatmap.py CELLS)
  // v = sign-corrected connectome advantage over its random control (%, + = connectome better); raw = "connectome vs random"
  matrix: {
    rows: [
      { key:'OL', name:'Optic lobe',      sub:'visual motion' },
      { key:'MB', name:'Mushroom body',   sub:'associative memory' },
      { key:'CX', name:'Central complex', sub:'navigation' },
    ],
    cols: [
      { key:'flow',  name:'Optic flow',      sub:'DSEC / hex',      metric:'flow error',           better:'low' },
      { key:'mqar',  name:'Assoc. recall',   sub:'MQAR',            metric:'recall accuracy',      better:'high' },
      { key:'path',  name:'Path integration',sub:'heading WM',      metric:'heading error (rad)',  better:'low' },
      { key:'mnist', name:'Seq-MNIST',       sub:'image (foreign)', metric:'accuracy',             better:'high', foreign:true },
      { key:'arith', name:'Arithmetic',      sub:'(foreign)',       metric:'accuracy (~chance)',   better:'high', foreign:true },
    ],
    cells: {
      OL: {
        flow:  { v:12.0, raw:'1.089 vs 1.238', native:true },
        mqar:  { v:8.5,  raw:'0.953 vs 0.878' },
        path:  { v:-3.4, raw:'0.390 vs 0.377' },
        mnist: { v:2.4,  note:'ties its weight-shuffle control (seq-MNIST acc ≈ 0.949); the +2.4% is only over fully-random (generic sparsity), and the no-recurrence ablation collapses to chance - so recurrence is genuinely used.' },
        arith: { v:-1.6, raw:'0.138 vs 0.141' },
      },
      MB: {
        flow:  { v:3.3,  raw:'1.049 vs 1.085' },
        mqar:  { v:10.6, raw:'0.925 vs 0.836', native:true },
        path:  { v:-2.9, raw:'0.386 vs 0.375' },
        mnist: { v:1.4,  note:'ties its weight-shuffle control (seq-MNIST acc ≈ 0.961); the +1.4% is only over fully-random (generic sparsity); recurrence is load-bearing.' },
        arith: { v:1.3,  raw:'0.142 vs 0.141' },
      },
      CX: {
        flow:  { v:0.5,  raw:'1.031 vs 1.036' },
        mqar:  { v:-3.0, raw:'0.816 vs 0.841' },
        path:  { v:7.8,  raw:'0.390 vs 0.423', native:true },
        mnist: { v:1.7,  note:'ties its weight-shuffle control (seq-MNIST acc ≈ 0.971); the +1.7% is only over fully-random (generic sparsity); recurrence is load-bearing.' },
        arith: { v:-0.5, raw:'0.144 vs 0.144' },
      },
    },
    notes: {
      'OL-flow':'Optic lobe on optic flow - the native win, +12.0%. On real DSEC event-camera flow the connectome beats its random control (endpoint error 1.089 vs 1.238); the edge is largest when training data is scarce.',
      'OL-mqar':'Optic lobe scores +8.5% on MQAR - nearly matching the mushroom body. A matched-size control shows this is capacity, not biology: subsample OL to 14,025 neurons and it collapses to ≈ MB. Associative recall is generic structured recurrence.',
      'OL-path':'Optic lobe on path integration: -3.4%. A visual circuit has no heading-integration prior, so it trails its own random control. The wrong region loses.',
      'MB-flow':'Mushroom body on optic flow: +3.3% - a small edge from generic sparse structure, well below the optic lobe’s native +12.0%.',
      'MB-mqar':'Mushroom body on MQAR - the native win, +10.6% (0.925 vs 0.836). Kenyon-cell sparse coding separates stored bindings, reducing interference; reaches 0.995 recall with longer training.',
      'MB-path':'Mushroom body on path integration: -2.9%. An associative circuit has no ring-attractor for heading. The wrong region loses.',
      'CX-flow':'Central complex on optic flow: +0.5% - essentially a tie. A navigation circuit carries no motion-vision prior.',
      'CX-mqar':'Central complex on MQAR: -3.0% (0.816 vs 0.841). A navigation circuit has no associative-memory prior, so it trails its random control.',
      'CX-path':'Central complex on path integration - the cleanest native win, +7.8% (0.390 vs 0.423 rad). The ring-attractor computation IS the wiring: weight-shuffle tracks the connectome, random and degree-shuffle fall behind.',
      'mnist':'Sequential MNIST is a foreign task. Every region merely ties its topology-matched (weight-shuffle) control; the small positive number is only an edge over fully-random, i.e. generic sparsity. The no-recurrence ablation collapses to ≈ chance, proving the recurrence is used but wiring-specific structure buys nothing.',
      'arith':'Running-sum-mod-m arithmetic sits at chance for every model in 20 epochs - a hard task no wiring rescues. The sign of this cell is within noise.',
    },
  },
};

/* ---------- reusable SVG charts (error bars/bands, hover tooltips, a11y) ---------- */
// Grouped bar chart. series: {name,color,kind,vals:[],err:[],labelDirect}. cfg.aria for a11y.
function barChart(container, cfg) {
  const w = cfg.w||520, h = cfg.h||280, m = {t:20,r:16,b:44,l:46};
  const iw = w-m.l-m.r, ih = h-m.t-m.b;
  const yMin = cfg.yMin ?? 0, yMax = cfg.yMax ?? 1;
  const Y = v => m.t + ih - ((v-yMin)/(yMax-yMin))*ih;
  const svg = el('svg',{viewBox:`0 0 ${w} ${h}`, width:'100%', role:'img', 'aria-label':cfg.aria||'bar chart'});
  const ticks = cfg.ticks || 5;
  for (let i=0;i<=ticks;i++){
    const val = yMin + (yMax-yMin)*i/ticks, y = Y(val);
    svg.appendChild(el('line',{x1:m.l,y1:y,x2:w-m.r,y2:y,stroke:C.grid,'stroke-width':1}));
    svg.appendChild(el('text',{x:m.l-8,y:y+4,'text-anchor':'end',fill:C.mute,'font-size':10,'font-family':'ui-monospace,monospace',text:fmt(val,cfg.dec??2)}));
  }
  const gN = cfg.groups.length, sN = cfg.series.length;
  const gW = iw/gN, bW = Math.min(30, (gW*0.72)/sN);
  cfg.groups.forEach((g,gi)=>{
    const gx = m.l + gW*gi + gW/2;
    svg.appendChild(el('text',{x:gx,y:h-24,'text-anchor':'middle',fill:C.dim,'font-size':11,text:g.label}));
    if (g.sub) svg.appendChild(el('text',{x:gx,y:h-11,'text-anchor':'middle',fill:C.mute,'font-size':9,text:g.sub}));
    cfg.series.forEach((s,si)=>{
      const v = s.vals[gi]; if (v==null) return;
      const bx = gx - (sN*bW)/2 + si*bW, cx = bx+bW/2;
      const y = Y(v), bh = Y(yMin)-y;
      const r = el('rect',{x:bx+1,y:y,width:bW-2,height:Math.max(1,bh),rx:3,fill:s.color,opacity:s.kind==='bio'?1:0.82});
      r.style.transition='height .6s var(--ease), y .6s var(--ease)';
      svg.appendChild(r);
      const e = s.err && s.err[gi];
      if (e){
        const yT=Y(v+e), yB=Y(Math.max(yMin,v-e));
        svg.appendChild(el('line',{x1:cx,y1:yT,x2:cx,y2:yB,stroke:C.ink,'stroke-width':1,opacity:.55}));
        svg.appendChild(el('line',{x1:cx-3,y1:yT,x2:cx+3,y2:yT,stroke:C.ink,'stroke-width':1,opacity:.55}));
        svg.appendChild(el('line',{x1:cx-3,y1:yB,x2:cx+3,y2:yB,stroke:C.ink,'stroke-width':1,opacity:.55}));
      }
      if (s.labelDirect) svg.appendChild(el('text',{x:cx,y:(e?Y(v+e):y)-5,'text-anchor':'middle',fill:s.color,'font-size':9.5,'font-family':'ui-monospace,monospace',text:fmt(v,cfg.dec??2)}));
      const hit = el('rect',{x:bx,y:m.t,width:bW,height:ih,fill:'transparent'});
      hit.style.cursor='pointer';
      hit.addEventListener('mousemove',ev=>showTip(ev.clientX,ev.clientY,
        `<div class="tt-h">${s.name||''}</div><div class="tt-r">${g.label}: <b>${fmt(v,cfg.dec??3)}</b>${e?` &plusmn; ${fmt(e,cfg.dec??3)}`:''}</div>`));
      hit.addEventListener('mouseleave',hideTip);
      svg.appendChild(hit);
    });
  });
  container.innerHTML=''; container.appendChild(svg); return svg;
}

// Multi-line chart. series: {name,color,kind,dash,vals:[],band:[],labelDirect}. cfg.aria for a11y.
function lineChart(container, cfg) {
  const w = cfg.w||560, h = cfg.h||300, m={t:18,r:20,b:42,l:50};
  const iw=w-m.l-m.r, ih=h-m.t-m.b;
  const xs=cfg.x, xMin=Math.min(...xs), xMax=Math.max(...xs);
  const yMin=cfg.yMin, yMax=cfg.yMax;
  const X=v=>m.l+((v-xMin)/(xMax-xMin))*iw;
  const Y=v=>m.t+ih-((v-yMin)/(yMax-yMin))*ih;
  const svg=el('svg',{viewBox:`0 0 ${w} ${h}`,width:'100%',role:'img','aria-label':cfg.aria||'line chart'});
  const ticks=cfg.ticks||5;
  for(let i=0;i<=ticks;i++){
    const val=yMin+(yMax-yMin)*i/ticks,y=Y(val);
    svg.appendChild(el('line',{x1:m.l,y1:y,x2:w-m.r,y2:y,stroke:C.grid,'stroke-width':1}));
    svg.appendChild(el('text',{x:m.l-8,y:y+4,'text-anchor':'end',fill:C.mute,'font-size':10,'font-family':'ui-monospace,monospace',text:fmt(val,cfg.dec??2)}));
  }
  xs.forEach(xv=>{
    svg.appendChild(el('text',{x:X(xv),y:h-22,'text-anchor':'middle',fill:C.dim,'font-size':10,text:cfg.xfmt?cfg.xfmt(xv):xv}));
  });
  if(cfg.xlabel) svg.appendChild(el('text',{x:m.l+iw/2,y:h-6,'text-anchor':'middle',fill:C.mute,'font-size':10,text:cfg.xlabel}));
  if(cfg.markX!=null){
    const mx=X(cfg.markX);
    svg.appendChild(el('line',{x1:mx,y1:m.t,x2:mx,y2:m.t+ih,stroke:C.teal,'stroke-width':1.5,'stroke-dasharray':'4 3',opacity:.7}));
  }
  // ±std bands behind lines
  cfg.series.forEach(s=>{
    if(!s.band) return;
    const up=s.vals.map((v,i)=>`${X(xs[i])},${Y(v+(s.band[i]||0))}`);
    const dn=s.vals.map((v,i)=>`${X(xs[i])},${Y(v-(s.band[i]||0))}`).reverse();
    svg.appendChild(el('polygon',{points:up.concat(dn).join(' '),fill:s.color,opacity:.12,stroke:'none'}));
  });
  cfg.series.forEach(s=>{
    const pts=s.vals.map((v,i)=>`${X(xs[i])},${Y(v)}`).join(' ');
    svg.appendChild(el('polyline',{points:pts,fill:'none',stroke:s.color,'stroke-width':s.kind==='bio'?3:2,
      'stroke-dasharray':s.dash||'','stroke-linejoin':'round','stroke-linecap':'round',opacity:s.kind==='ctrl'?0.9:1}));
    s.vals.forEach((v,i)=>svg.appendChild(el('circle',{cx:X(xs[i]),cy:Y(v),r:s.kind==='bio'?3.4:2.6,fill:s.color})));
    if(s.labelDirect){
      const li=s.vals.length-1;
      svg.appendChild(el('text',{x:X(xs[li])+6,y:Y(s.vals[li])+3,fill:s.color,'font-size':10,'font-weight':600,text:s.name}));
    }
  });
  // hover columns
  xs.forEach((xv,i)=>{
    const colW=Math.max(12,iw/xs.length);
    const hit=el('rect',{x:X(xv)-colW/2,y:m.t,width:colW,height:ih,fill:'transparent'});
    hit.style.cursor='crosshair';
    hit.addEventListener('mousemove',ev=>{
      const rows=cfg.series.map(s=>`<div class="tt-r"><span style="color:${s.color}">&#9632;</span> ${s.name}: <b>${fmt(s.vals[i],cfg.dec??3)}</b>${s.band?` &plusmn; ${fmt(s.band[i]||0,cfg.dec??3)}`:''}</div>`).join('');
      showTip(ev.clientX,ev.clientY,`<div class="tt-h">${(cfg.xfmt?cfg.xfmt(xv):xv)}${cfg.xunit||''}</div>${rows}`);
    });
    hit.addEventListener('mouseleave',hideTip);
    svg.appendChild(hit);
  });
  container.innerHTML=''; container.appendChild(svg); return {svg,X,Y};
}

/* ---------- Chart/Table toggle + accessible data table ---------- */
function mountChart(host, spec){
  host.innerHTML='';
  const tools=el('div',{class:'chart-tools'});
  const bC=el('button',{class:'chip on','aria-pressed':'true',type:'button',text:'Chart'});
  const bT=el('button',{class:'chip','aria-pressed':'false',type:'button',text:'Table'});
  tools.append(bC,bT);
  const cw=el('div',{class:'chart-view'});
  const tw=el('div',{class:'table-view',style:'display:none'});
  spec.render(cw);
  if(spec.table) tw.appendChild(dataTable(spec.table));
  host.append(tools,cw,tw);
  const show=chart=>{
    bC.classList.toggle('on',chart); bT.classList.toggle('on',!chart);
    bC.setAttribute('aria-pressed',String(chart)); bT.setAttribute('aria-pressed',String(!chart));
    cw.style.display=chart?'':'none'; tw.style.display=chart?'none':'';
  };
  bC.onclick=()=>show(true); bT.onclick=()=>show(false);
}
function dataTable({caption,cols,rows}){
  const wrap=el('div',{style:'overflow-x:auto'});
  const t=el('table',{class:'dtable'});
  if(caption) t.appendChild(el('caption',{text:caption,style:'caption-side:top;text-align:left;color:var(--text-mute);font-size:.76rem;padding:0 0 8px'}));
  const thead=el('tr'); cols.forEach(c=>thead.appendChild(el('th',{text:c}))); t.appendChild(thead);
  rows.forEach(r=>{
    const cells=r.cells||r, tr=el('tr'); if(r._bio) tr.setAttribute('class','bio');
    cells.forEach(c=>tr.appendChild(el('td',{text:String(c)}))); t.appendChild(tr);
  });
  wrap.appendChild(t); return wrap;
}

/* ---------- nav, reveal, tooltip ---------- */
function initNav(){
  const nav=$('.nav'), links=$$('.nav-links a[href^="#"]'), toggle=$('.nav-toggle'), menu=$('.nav-links');
  toggle && toggle.addEventListener('click',()=>menu.classList.toggle('open'));
  menu && menu.addEventListener('click',e=>{ if(e.target.tagName==='A') menu.classList.remove('open'); });
  const sections=$$('section[id]');
  const spy=()=>{
    const y=window.scrollY+120; let cur='';
    sections.forEach(s=>{ if(s.offsetTop<=y) cur=s.id; });
    links.forEach(a=>a.classList.toggle('active', a.getAttribute('href')==='#'+cur));
  };
  window.addEventListener('scroll',spy,{passive:true}); spy();
}
function initReveal(){
  const io=new IntersectionObserver((es)=>es.forEach(e=>{ if(e.isIntersecting){ e.target.classList.add('in'); io.unobserve(e.target);} }),{threshold:0.12});
  $$('.reveal').forEach(n=>io.observe(n));
}
// generic once-visible activator for demos (so canvases only animate when seen)
function whenVisible(node, cb){
  let done=false;
  const io=new IntersectionObserver((es)=>es.forEach(e=>{ if(e.isIntersecting && !done){ done=true; cb(); io.disconnect(); }}),{threshold:0.15});
  io.observe(node);
}
let _tt;
function tooltip(){ if(!_tt){ _tt=el('div',{class:'tt'}); document.body.appendChild(_tt);} return _tt; }
function showTip(x,y,html){ const t=tooltip(); t.innerHTML=html; t.classList.add('show');
  const r=t.getBoundingClientRect(); t.style.left=clamp(x+14,8,innerWidth-r.width-8)+'px'; t.style.top=clamp(y+14,8,innerHeight-r.height-8)+'px'; }
function hideTip(){ if(_tt) _tt.classList.remove('show'); }

function initTheme(){
  const btn=document.getElementById('theme-toggle');
  let saved=null; try{ saved=localStorage.getItem('theme'); }catch(e){}
  if(saved==='light') document.documentElement.setAttribute('data-theme','light');
  const setIcon=()=>{ if(btn) btn.textContent = document.documentElement.getAttribute('data-theme')==='light'?'☀':'◐'; };
  setIcon();
  btn && btn.addEventListener('click',()=>{
    const isLight=document.documentElement.getAttribute('data-theme')==='light';
    if(isLight) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme','light');
    try{ localStorage.setItem('theme', isLight?'dark':'light'); }catch(e){}
    setIcon();
  });
}
document.addEventListener('DOMContentLoaded',()=>{ initNav(); initReveal(); initTheme(); });
