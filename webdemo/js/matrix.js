/* ============================================================
   matrix.js - region×task matrix, emergence, BPU bars, scale
   ============================================================ */
(function () {

  /* ---------- region × task matrix ---------- */
  function advColor(v) {
    // diverging: red (neg) - gray (0) - green (pos), capped at ±12
    const t = clamp(v / 12, -1, 1);
    if (t >= 0) {
      // gray -> green
      const a = 0.10 + t * 0.42;
      return `rgba(55,201,138,${a})`;
    } else {
      const a = 0.08 + (-t) * 0.30;
      return `rgba(236,106,106,${a})`;
    }
  }
  function buildMatrix() {
    const host = document.getElementById('matrix-grid'); if (!host) return;
    const M = DATA.matrix;
    const tbl = el('table', { class: 'matrix' });
    // header
    const thead = el('tr', {}, [el('th', {})]);
    M.cols.forEach(c => {
      const th = el('th', {}, [
        el('div', { text: c.name, style: 'font-weight:600' }),
        el('div', { text: c.sub, style: 'font-size:.68rem;color:var(--text-mute);font-weight:400' }),
      ]);
      thead.appendChild(th);
    });
    tbl.appendChild(thead);
    M.rows.forEach(r => {
      const tr = el('tr', {});
      const rh = el('th', { class: 'rowh' }, [
        el('div', { text: r.name }),
        el('small', { text: r.sub }),
      ]);
      tr.appendChild(rh);
      M.cols.forEach(c => {
        const cell = M.cells[r.key][c.key];
        const v = cell.v;
        const td = el('td', {});
        const div = document.createElement('div');
        div.className = 'mcell' + (cell.native ? ' native' : '');
        div.setAttribute('role', 'button'); div.tabIndex = 0;
        div.dataset.region = r.key; div.dataset.task = c.key;
        div.setAttribute('aria-label', `${r.name} on ${c.name}: connectome advantage ${v > 0 ? '+' : ''}${v.toFixed(1)} percent. Activate to run the head-to-head.`);
        div.style.background = advColor(v);
        const txt = Math.abs(v) > 7 ? '#fff' : (v >= 1.5 ? 'var(--green)' : (v <= -1.5 ? 'var(--red)' : 'var(--text-dim)'));
        const corner = cell.native ? '<div class="native-tag">native</div>'
          : (c.foreign ? '<div class="native-tag" style="color:var(--text-mute)">✗</div>' : '');
        const kind = cell.native ? 'connectome wins' : (c.foreign ? 'foreign' : (v <= -1.5 ? 'off-native' : (v >= 1.5 ? 'weak edge' : 'tie')));
        div.innerHTML = `${corner}<div class="mv" style="color:${txt}">${v > 0 ? '+' : ''}${v.toFixed(1)}%</div><div class="mk">${kind}</div>`;
        const go = () => runCell(r, c, cell);
        div.addEventListener('click', go);
        div.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } });
        td.appendChild(div);
        tr.appendChild(td);
      });
      tbl.appendChild(tr);
    });
    host.innerHTML = ''; host.appendChild(tbl);
    // run the default head-to-head (CX -> path, the cleanest native win)
    runCell(M.rows.find(x => x.key === 'CX'), M.cols.find(x => x.key === 'path'), M.cells.CX.path);
  }

  /* ---------- in-place head-to-head (former picker, now driven by cell clicks) ---------- */
  function markSelected(rk, ck) {
    $$('#matrix-grid .mcell').forEach(m => m.classList.toggle('sel', m.dataset.region === rk && m.dataset.task === ck));
  }
  function animateNumber(elm, target) {
    if (!elm) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      elm.textContent = (target >= 0 ? '+' : '') + target.toFixed(1) + '%'; return;
    }
    const dur = 600; let t0 = null;
    (function step(ts) {
      if (t0 == null) t0 = ts;
      const p = Math.min(1, (ts - t0) / dur), val = target * p;
      elm.textContent = (val >= 0 ? '+' : '') + val.toFixed(1) + '%';
      if (p < 1) requestAnimationFrame(step);
      else elm.textContent = (target >= 0 ? '+' : '') + target.toFixed(1) + '%';
    })(0);
  }
  function runCell(r, c, cell) {
    const host = document.getElementById('matrix-result'); if (!host) return;
    markSelected(r.key, c.key);
    const M = DATA.matrix, v = cell.v, nearZero = Math.abs(v) < 1.5;
    const story = M.notes[r.key + '-' + c.key] || M.notes[c.key] || '';
    let tagCls, headline;
    if (cell.native) { tagCls = 'pv-native'; headline = `The connectome wins, +${v.toFixed(1)}%`; }
    else if (c.foreign) { tagCls = 'pv-foreign'; headline = nearZero ? 'Tie - no wiring-specific edge' : `Only ${v > 0 ? '+' : ''}${v.toFixed(1)}% vs random`; }
    else if (nearZero) { tagCls = 'pv-foreign'; headline = 'Essentially a tie'; }
    else if (v < 0) { tagCls = 'pv-off'; headline = `The wrong region loses, ${v.toFixed(1)}%`; }
    else { tagCls = 'pv-off'; headline = `A weak +${v.toFixed(1)}% - generic, not aligned`; }
    const deltaCls = nearZero ? 'zero' : (v > 0 ? 'pos' : 'neg');

    const half = Math.min(Math.abs(v), 13) / 13 * 50;
    const barCol = v >= 0 ? 'var(--green)' : 'var(--red)';
    const side = v >= 0 ? 'left:50%' : 'right:50%';
    const advBar =
      `<div style="position:relative;height:20px;background:var(--surface-3);border-radius:6px;margin:14px 0 4px">
         <div style="position:absolute;left:50%;top:-3px;bottom:-3px;width:1px;background:var(--text-mute);opacity:.6"></div>
         <div data-w="${half.toFixed(1)}" style="position:absolute;top:2px;bottom:2px;${side};width:0;background:${barCol};border-radius:4px;transition:width .6s var(--ease)"></div>
       </div>
       <div style="display:flex;justify-content:space-between;font-size:.68rem;color:var(--text-faint)"><span>random better</span><span>0</span><span>connectome better</span></div>`;

    let pairBlock = '';
    if (cell.raw) {
      const [a, b] = cell.raw.split('vs').map(s => parseFloat(s));
      const maxv = Math.max(a, b), minv = Math.min(a, b);
      const len = val => (c.better === 'high' ? val / maxv : minv / val) * 100;
      const showTick = !c.foreign && !nearZero;
      const conWins = c.better === 'high' ? a > b : a < b;
      const rowHtml = (label, val, colr, isCon, win) =>
        `<div style="display:grid;grid-template-columns:88px 1fr auto;align-items:center;gap:10px;margin:6px 0">
           <span style="font-size:.8rem;color:${isCon ? 'var(--bio)' : 'var(--text-dim)'};text-align:right">${label}${win && showTick ? ' ✓' : ''}</span>
           <span style="height:15px;background:var(--surface-3);border-radius:5px;overflow:hidden"><span data-w="${Math.max(6, len(val)).toFixed(1)}" style="display:block;height:100%;width:0;background:${colr};border-radius:5px;transition:width .6s var(--ease)"></span></span>
           <span class="mono" style="font-size:.8rem;color:${isCon ? 'var(--bio)' : 'var(--text-mute)'}">${val.toFixed(3)}</span>
         </div>`;
      pairBlock =
        `<div style="font-size:.7rem;color:var(--text-mute);font-family:var(--mono);margin:14px 0 4px">${c.metric} · ${c.better === 'low' ? 'lower' : 'higher'} is better</div>` +
        rowHtml('Connectome', a, C.bio, true, conWins) + rowHtml('Random', b, C.ctrl, false, !conWins);
    } else if (cell.note) {
      pairBlock = `<div style="font-size:.8rem;color:var(--text-mute);margin-top:14px;line-height:1.5">${cell.note}</div>`;
    }

    host.innerHTML =
      `<div class="pv-tag ${tagCls}">${r.name} &rarr; ${c.name}${cell.native ? ' · NATIVE' : ''}</div>` +
      `<h4 style="margin:2px 0 8px">${headline}</h4>` +
      `<p style="margin:0;color:var(--text-dim);font-size:.9rem">${story}</p>` +
      `<hr class="divider" style="margin:16px 0">` +
      `<div style="font-size:.72rem;color:var(--text-mute);text-transform:uppercase;letter-spacing:.1em;margin-bottom:4px">connectome advantage over random</div>` +
      `<div class="big-delta ${deltaCls}" id="mx-delta">+0.0%</div>` +
      advBar + pairBlock;

    requestAnimationFrame(() => {
      host.querySelectorAll('[data-w]').forEach(elm => { elm.style.width = elm.dataset.w + '%'; });
      animateNumber(document.getElementById('mx-delta'), v);
    });
  }

  /* ---------- input-port emergence ---------- */
  function drawEmerge(progress = 1) {
    const host = document.getElementById('emerge-chart'); if (!host) return;
    const w = 520, h = 280, m = { t: 18, r: 20, b: 42, l: 48 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const X = t => m.l + t * iw;                     // t in 0..1
    const Y = a => m.t + ih - ((a - 0.45) / 0.22) * ih;  // 0.45..0.67
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%' });
    [0.45, 0.50, 0.55, 0.60, 0.65].forEach(v => {
      svg.appendChild(el('line', { x1: m.l, y1: Y(v), x2: w - m.r, y2: Y(v), stroke: C.grid }));
      svg.appendChild(el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', fill: C.mute, 'font-size': 9, 'font-family': 'ui-monospace,monospace', text: v.toFixed(2) }));
    });
    // chance line
    svg.appendChild(el('line', { x1: m.l, y1: Y(0.5), x2: w - m.r, y2: Y(0.5), stroke: C.mute, 'stroke-dasharray': '2 3', opacity: .7 }));
    svg.appendChild(el('text', { x: m.l + iw / 2, y: h - 8, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'training →' }));
    svg.appendChild(el('text', { x: 8, y: Y(0.58), fill: C.mute, 'font-size': 10, transform: `rotate(-90 12 ${Y(0.58)})`, text: 'input-port ROC-AUC' }));
    // connectome curve: 0.5 -> ~0.6
    const bioPts = [], ctrlPts = [];
    const N = Math.max(2, Math.floor(80 * progress));
    for (let i = 0; i <= N; i++) {
      const t = i / 80;
      const bio = 0.5 + 0.098 * (1 - Math.exp(-t * 3.4)) + Math.sin(t * 22) * 0.004;
      const ctrl = 0.5 + Math.sin(t * 15 + 1) * 0.006 - 0.002;
      bioPts.push(`${X(t)},${Y(bio)}`); ctrlPts.push(`${X(t)},${Y(ctrl)}`);
    }
    svg.appendChild(el('polyline', { points: ctrlPts.join(' '), fill: 'none', stroke: C.ctrl, 'stroke-width': 2, 'stroke-dasharray': '5 4' }));
    svg.appendChild(el('polyline', { points: bioPts.join(' '), fill: 'none', stroke: C.bio, 'stroke-width': 3, 'stroke-linejoin': 'round' }));
    if (progress >= 1) {
      svg.appendChild(el('circle', { cx: X(1), cy: Y(0.598), r: 4, fill: C.bio }));
      svg.appendChild(el('text', { x: X(1) - 6, y: Y(0.598) - 8, 'text-anchor': 'end', fill: C.bio, 'font-size': 11, 'font-weight': 600, 'font-family': 'ui-monospace,monospace', text: 'AUC ≈ 0.60' }));
      svg.appendChild(el('text', { x: X(1) - 6, y: Y(0.5) - 6, 'text-anchor': 'end', fill: C.ctrl, 'font-size': 10, text: 'random stays at chance' }));
      svg.appendChild(el('text', { x: X(0.5), y: m.t + 12, 'text-anchor': 'middle', fill: C.teal, 'font-size': 10, 'font-family': 'ui-monospace,monospace', text: 'p = 3.6 × 10⁻⁵' }));
    }
    host.innerHTML = ''; host.appendChild(svg);
  }
  function initEmerge() {
    let played = false;
    const play = () => { let p = 0; const id = setInterval(() => { p += 0.03; drawEmerge(Math.min(1, p)); if (p >= 1) clearInterval(id); }, 26); };
    drawEmerge(1);
    document.getElementById('emerge-replay').addEventListener('click', play);
    whenVisible(document.getElementById('emerge-chart'), () => { if (!played) { played = true; drawEmerge(0); setTimeout(play, 250); } });
  }

  /* ---------- BPU reproduction bars (real, connectome does NOT win) ---------- */
  function bpuBars() {
    const host = document.getElementById('bpu-bars'); if (!host) return;
    const B = window.RESULTS && window.RESULTS.bpu; if (!B) return;
    const colorFor = k => ({ bio: C.bio, ctrl: C.ctrl, shuffle: C.blue, dense: C.violet, mlp: C.mute }[k] || C.ctrl);
    const chart = B.mnist.slice(0, 4);   // connectome, random sparse, weight shuffle, dense trainable
    mountChart(host, {
      render: c => {
        barChart(c, {
          w: 460, h: 250, yMin: 0, yMax: 1, dec: 2, ticks: 5, aria: 'BPU reproduction accuracy on MNIST and CIFAR-10',
          groups: [{ label: 'MNIST' }, { label: 'CIFAR-10' }],
          series: chart.map((mo, i) => ({ name: mo.model, color: colorFor(mo.kind), kind: mo.kind,
            vals: [B.mnist[i].acc, B.cifar[i].acc], err: [B.mnist[i].std, B.cifar[i].std] })),
        });
        const leg = document.createElement('div'); leg.className = 'legend'; leg.style.cssText = 'margin-top:10px;flex-wrap:wrap';
        leg.innerHTML = chart.map(mo => `<span class="li"><span class="swatch" style="background:${colorFor(mo.kind)}"></span>${mo.model}</span>`).join('');
        c.appendChild(leg);
      },
      table: {
        caption: 'BPU reproduction - test accuracy (mean ± std over seeds)',
        cols: ['Model', 'MNIST', 'CIFAR-10'],
        rows: B.mnist.map((mo, i) => { const row = [mo.model, mo.acc.toFixed(3) + ' ± ' + mo.std.toFixed(3), B.cifar[i].acc.toFixed(3) + ' ± ' + B.cifar[i].std.toFixed(3)]; row._bio = mo.kind === 'bio'; return row; }),
      },
    });
  }

  /* ---------- scaling timeline ---------- */
  const SCALE = [
    { name: 'Fly', n: '~140K neurons', note: 'Mapped now', status: 'available',
      detail: 'The fruit-fly connectome (FlyWire / hemibrain) is fully reconstructed - the smallest complete brain we have. Every result on this page comes from three of its circuits: the central complex, mushroom body and optic lobe. This is the proof of principle: evolved wiring beats matched-random controls on the tasks those circuits were built to solve.' },
    { name: 'Mouse', n: '~70M neurons', note: 'In progress', status: 'progress',
      detail: 'Cubic-millimeter cortical volumes (MICrONS) and whole-brain efforts are being reconstructed now. Mouse head-direction and grid systems run on the same ring-attractor logic as the fly compass, at far larger scale - the first test of whether the wiring-as-computation lesson carries to a mammalian brain, with richer navigation, vision and memory priors to mine.' },
    { name: 'Monkey', n: '~6B neurons', note: 'On the horizon', status: 'future',
      detail: 'A primate connectome would supply priors far closer to human cognition - dexterous sensorimotor control, richer visual hierarchies, working memory. Each region becomes a candidate module for the job it evolved to do, tested by the same brutal matched-random control that keeps the fly results honest.' },
    { name: 'Human', n: '~86B neurons', note: 'The goal', status: 'future',
      detail: 'A human wiring diagram is the long-term ambition. The wager is not a brain-in-a-box but a library of evolved, task-specialized priors you compose and fine-tune. Whether that composes all the way to general intelligence is genuinely open - what is already defensible is that mining evolved circuits beats matched-random baselines on aligned tasks. Alignment, not scale alone, is the leverage.' },
  ];
  function buildScale() {
    const track = document.getElementById('scale-track'); if (!track) return;
    track.innerHTML = '';
    SCALE.forEach((s, i) => {
      const node = document.createElement('div');
      node.className = 'scale-node' + (i === 0 ? ' on' : '');
      node.innerHTML = `<div class="sn-dot"></div><div class="sn-name">${s.name}</div><div class="sn-n">${s.n}</div><div class="sn-note">${s.note}</div>`;
      node.addEventListener('mouseenter', () => select(i));
      node.addEventListener('click', () => select(i));
      track.appendChild(node);
    });
    const slider = document.getElementById('scale-slider');
    slider && slider.addEventListener('input', () => select(+slider.value));
    select(0);
  }
  const SCALE_LOG = [5.15, 7.85, 9.81, 10.93]; // log10 neuron counts: fly, mouse, monkey, human
  function drawMorph(idx) {
    const host = document.getElementById('scale-morph'); if (!host) return;
    host.innerHTML = '';
    SCALE_LOG.forEach((l, i) => {
      const b = el('div', { class: 'sm-bar' });
      b.style.height = (l / 11 * 100) + '%'; b.style.opacity = i <= idx ? 1 : 0.25;
      host.appendChild(b);
    });
  }
  function select(idx) {
    $$('#scale-track .scale-node').forEach((n, i) => n.classList.toggle('on', i <= idx));
    const s = SCALE[idx];
    const slider = document.getElementById('scale-slider'); if (slider && +slider.value !== idx) slider.value = idx;
    const val = document.getElementById('scale-val'); if (val) val.textContent = s.name + ' · ' + s.n;
    drawMorph(idx);
    const d = document.getElementById('scale-detail');
    d.innerHTML = `<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;flex-wrap:wrap"><h4 style="margin:0;font-size:1.1rem">${s.name} connectome</h4><span class="mono" style="color:var(--bio)">${s.n}</span><span class="pill" style="margin:0">${s.note}</span></div><p style="margin:0;color:var(--text-dim)">${s.detail}</p>`;
  }

  document.addEventListener('DOMContentLoaded', () => {
    buildMatrix();
    initEmerge();
    bpuBars();
    buildScale();
  });
})();
