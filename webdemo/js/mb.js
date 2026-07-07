/* ============================================================
   mb.js - Mushroom body: MQAR recall, reversal, retention
   ============================================================ */
(function () {

  /* ---------- horizontal bar list (reused) ---------- */
  function hbars(host, rows, opts = {}) {
    const max = opts.max ?? Math.max(...rows.map(r => r.v));
    const min = opts.min ?? 0;
    host.innerHTML = '';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display:flex;flex-direction:column;gap:9px';
    rows.forEach(r => {
      const row = document.createElement('div');
      row.style.cssText = 'display:grid;grid-template-columns:152px 1fr 52px;align-items:center;gap:10px;cursor:default';
      const name = document.createElement('div');
      name.style.cssText = `font-size:.82rem;color:${r.kind === 'bio' ? 'var(--bio)' : 'var(--text-dim)'};${r.kind === 'bio' ? 'font-weight:600' : ''};text-align:right`;
      name.textContent = r.model;
      const track = document.createElement('div');
      track.style.cssText = 'height:16px;background:var(--surface-3);border-radius:5px;overflow:hidden;position:relative';
      const fill = document.createElement('div');
      const pct = ((r.v - min) / (max - min)) * 100;
      fill.style.cssText = `height:100%;width:0;border-radius:5px;background:${r.color};transition:width .8s var(--ease)`;
      requestAnimationFrame(() => fill.style.width = pct + '%');
      track.appendChild(fill);
      const val = document.createElement('div');
      val.style.cssText = `font-family:var(--mono);font-size:.82rem;color:${r.kind === 'bio' ? 'var(--bio)' : 'var(--text-mute)'};font-variant-numeric:tabular-nums`;
      val.textContent = fmt(r.v, opts.dec ?? 3);
      row.append(name, track, val);
      row.addEventListener('mousemove', ev => showTip(ev.clientX, ev.clientY,
        `<div class="tt-h">${r.model}</div><div class="tt-r"><b>${fmt(r.v, opts.dec ?? 3)}</b>${r.sd ? ` &plusmn; ${r.sd}` : ''}${r.note ? ` &middot; ${r.note}` : ''}</div>`));
      row.addEventListener('mouseleave', hideTip);
      wrap.appendChild(row);
    });
    host.appendChild(wrap);
  }

  /* ---------- MQAR benchmark bars (real means, chart+table) ---------- */
  function mqarBars() {
    const host = document.getElementById('mqar-bars'); if (!host) return;
    const MQ = (window.RESULTS && window.RESULTS.mqar) || DATA.mqar;
    const colorFor = k => ({ bio: C.bio, shuffle: C.blue, ctrl: C.ctrl, ceiling: C.teal, chance: C.mute }[k] || C.ctrl);
    mountChart(host, {
      render: c => hbars(c, MQ.map(m => ({ model: m.model, v: m.acc, kind: m.kind, sd: m.sd, note: m.note, color: colorFor(m.kind) })), { min: 0, max: 1, dec: 3 }),
      table: {
        caption: 'MQAR test recall accuracy (higher is better)',
        cols: ['Model', 'Recall', '± std'],
        rows: MQ.map(m => { const row = [m.model, m.acc.toFixed(3), m.sd ? m.sd.toFixed(3) : ' - ']; row._bio = m.kind === 'bio'; return row; }),
      },
    });
  }

  /* ---------- MQAR interactive recall (plays back REAL trained-model outputs) ---------- */
  const GLYPHS = ['🍋', '🌸', '🍫', '🌿', '🍯', '🧭', '🍓', '🌶️', '🧄', '🍊', '🌰', '🍇', '🌼', '🍑', '🫐', '🍒', '🌹', '🥥', '🍍', '🌵', '🍄', '🌻', '🥝', '🍎', '🌷', '🥭', '🫒', '🌲', '🍈', '🌾', '🥕', '🌴'];
  const REC = window.MQAR_REC;
  const VOCAB = (REC && REC.vocab) || 32;
  const REAL = !!(REC && REC.episodes && REC.episodes.length);
  const EPISODES = REAL ? REC.episodes : synthEpisodes();
  let epIdx = 0, queried = -1;

  function synthEpisodes() {           // fallback only, until the recorded file lands
    const eps = [];
    for (let e = 0; e < 6; e++) {
      const pool = [...Array(VOCAB).keys()];
      for (let i = pool.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0;[pool[i], pool[j]] = [pool[j], pool[i]]; }
      const keys = pool.slice(0, 8), values = keys.map(() => (Math.random() * VOCAB) | 0);
      const queries = keys.map((k, qi) => {
        const truth = values[qi];
        const bio = Array.from({ length: VOCAB }, () => Math.random() * 0.02); bio[truth] = 0.9 + Math.random() * 0.08;
        const wrong = Math.random() > 0.82, distr = (truth + 1 + ((Math.random() * 5) | 0)) % VOCAB;
        const ctrl = Array.from({ length: VOCAB }, () => Math.random() * 0.04); ctrl[truth] = wrong ? 0.25 : 0.55; ctrl[distr] = wrong ? 0.6 : 0.3;
        const bp = bio.indexOf(Math.max(...bio)), cp = ctrl.indexOf(Math.max(...ctrl));
        return { key: k, truth, connectome: { probs: bio, pred: bp }, random: { probs: ctrl, pred: cp } };
      });
      eps.push({ pairs: keys.map((k, qi) => [k, values[qi]]), queries });
    }
    return eps;
  }

  function renderKeys() {
    const host = document.getElementById('mqar-keys'); if (!host) return;
    host.style.cssText = 'display:flex;flex-wrap:wrap;gap:9px';
    host.innerHTML = '';
    EPISODES[epIdx].pairs.forEach((pair, i) => {
      const [kt, vt] = pair;
      const chip = document.createElement('button');
      chip.className = 'btn-sm';
      chip.style.cssText = `display:flex;align-items:center;gap:7px;padding:8px 12px;${i === queried ? 'border-color:var(--bio);background:var(--bio-soft)' : ''}`;
      chip.innerHTML = `<span style="font-size:1.15rem">${GLYPHS[kt]}</span><span style="color:var(--text-mute)">→</span><span style="font-size:1.05rem">${GLYPHS[vt]}</span>`;
      chip.setAttribute('aria-label', `key ${kt} bound to value ${vt}, query it`);
      chip.addEventListener('click', () => { queried = i; renderKeys(); doQuery(i); });
      host.appendChild(chip);
    });
  }

  function doQuery(i) {
    const q0 = EPISODES[epIdx].queries[i];
    renderRecall({ bio: q0.connectome.probs, ctrl: q0.random.probs, truth: q0.truth, bioPick: q0.connectome.pred, ctrlPick: q0.random.pred });
    const okC = q0.connectome.pred === q0.truth, okR = q0.random.pred === q0.truth;
    const st = document.getElementById('mqar-status');
    st.innerHTML = `Query <b>${GLYPHS[q0.key]}</b>: connectome recalled <b style="color:var(--bio)">${GLYPHS[q0.connectome.pred]}</b> ${okC ? '✓' : '<span style="color:var(--red)">✗</span>'} · random recalled <b style="color:var(--ctrl)">${GLYPHS[q0.random.pred]}</b> ${okR ? '✓' : '<span style="color:var(--red)">✗ wrong</span>'}`;
  }

  function renderRecall(q) {
    const host = document.getElementById('mqar-recall'); if (!host) return;
    const V = VOCAB, w = 470, h = 200, m = { t: 14, r: 10, b: 40, l: 30 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', role: 'img', 'aria-label': 'recall probability over the vocabulary' });
    svg.appendChild(el('line', { x1: m.l, y1: m.t + ih, x2: w - m.r, y2: m.t + ih, stroke: C.grid }));
    if (!q) {
      svg.appendChild(el('text', { x: w / 2, y: h / 2, 'text-anchor': 'middle', fill: C.mute, 'font-size': 12, text: 'Query a key to see recall →' }));
      host.innerHTML = ''; host.appendChild(svg); return;
    }
    const gW = iw / V, bW = Math.max(2.4, gW * 0.34);
    const maxp = Math.max(0.02, ...q.bio, ...q.ctrl);
    const H = v => (v / maxp) * ih;
    const tx = m.l + gW * q.truth + gW / 2;
    svg.appendChild(el('rect', { x: tx - gW / 2, y: m.t, width: gW, height: ih, fill: C.teal, opacity: .12 }));
    for (let i = 0; i < V; i++) {
      const gx = m.l + gW * i + gW / 2;
      svg.appendChild(el('rect', { x: gx - bW - 0.4, y: m.t + ih - H(q.bio[i]), width: bW, height: H(q.bio[i]), rx: 1.5, fill: C.bio, opacity: i === q.bioPick ? 1 : .85 }));
      svg.appendChild(el('rect', { x: gx + 0.4, y: m.t + ih - H(q.ctrl[i]), width: bW, height: H(q.ctrl[i]), rx: 1.5, fill: C.ctrl, opacity: i === q.ctrlPick ? 1 : .7 }));
    }
    svg.appendChild(el('text', { x: tx, y: h - 20, 'text-anchor': 'middle', fill: C.teal, 'font-size': 15, text: GLYPHS[q.truth] }));
    svg.appendChild(el('text', { x: tx, y: h - 6, 'text-anchor': 'middle', fill: C.teal, 'font-size': 8, 'font-weight': 700, text: 'answer' }));
    host.innerHTML = ''; host.appendChild(svg);
  }

  function showEpisode(idx) {
    epIdx = ((idx % EPISODES.length) + EPISODES.length) % EPISODES.length;
    queried = -1; renderKeys();
    const st = document.getElementById('mqar-status');
    if (st) st.innerHTML = REAL ? 'Real trained-model outputs. Click a key to query it.' : 'Click a key to query it.';
    renderRecall(null);
  }
  function newBindings() { showEpisode(epIdx + 1); }

  /* ---------- reversal episode: recall over time, with a valence flip ---------- */
  const _rev = (window.RESULTS && window.RESULTS.reversal) || [];
  const _revRecall = k => { const r = _rev.find(x => x.kind === k); return r ? r.recall : null; };
  const REV = {
    T: 120,
    models: [
      { name: 'Connectome', color: C.bio, kind: 'bio', riseTau: 8, ceil1: 0.99, dip: 0.33, recTau: 6, ceil2: _revRecall('bio') || 0.9925 },
      { name: 'Random', color: C.ctrl, riseTau: 15, ceil1: 0.95, dip: 0.20, recTau: 11, ceil2: _revRecall('ctrl') || 0.9735 },
    ],
  };
  let revState = { running: false, t: 120, flipAt: 60, id: null };
  function revAcc(mo, t, flipAt) {
    if (t < flipAt) return mo.ceil1 - (mo.ceil1 - 0.5) * Math.exp(-t / mo.riseTau);
    return mo.ceil2 - (mo.ceil2 - mo.dip) * Math.exp(-(t - flipAt) / mo.recTau);
  }
  function drawReversal() {
    const host = document.getElementById('rev-curve'); if (!host) return;
    const w = 380, h = 230, m = { t: 22, r: 16, b: 40, l: 40 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const T = REV.T, tNow = revState.t, flipAt = revState.flipAt;
    const X = t => m.l + (t / T) * iw;
    const Y = a => m.t + ih - a * ih;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%' });
    [0, 0.25, 0.5, 0.75, 1].forEach(v => {
      svg.appendChild(el('line', { x1: m.l, y1: Y(v), x2: w - m.r, y2: Y(v), stroke: C.grid }));
      svg.appendChild(el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', fill: C.mute, 'font-size': 9, 'font-family': 'ui-monospace,monospace', text: v.toFixed(2) }));
    });
    svg.appendChild(el('text', { x: m.l + iw / 2, y: h - 7, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'time in episode' }));
    svg.appendChild(el('text', { x: m.l - 30, y: m.t + ih / 2, 'text-anchor': 'middle', fill: C.mute, 'font-size': 9, transform: `rotate(-90 ${m.l - 30} ${m.t + ih / 2})`, text: 'recall accuracy' }));
    // flip marker (revealed once the episode reaches it)
    if (tNow >= flipAt) {
      const fx = X(flipAt);
      svg.appendChild(el('line', { x1: fx, y1: m.t, x2: fx, y2: m.t + ih, stroke: C.red, 'stroke-width': 1.5, 'stroke-dasharray': '4 3', opacity: .85 }));
      svg.appendChild(el('text', { x: fx, y: m.t - 7, 'text-anchor': 'middle', fill: C.red, 'font-size': 9, 'font-weight': 600, text: '⚡ valence flips' }));
    }
    REV.models.forEach(mo => {
      const seg = [];
      for (let t = 0; t <= tNow; t += 1) seg.push([t, revAcc(mo, t, flipAt)]);
      if (!seg.length) return;
      const toPts = arr => arr.map(p => `${X(p[0])},${Y(p[1])}`).join(' ');
      const pre = seg.filter(p => p[0] < flipAt), post = seg.filter(p => p[0] >= flipAt);
      const sw = mo.kind === 'bio' ? 3 : 2, op = mo.kind === 'bio' ? 1 : .85;
      if (pre.length) svg.appendChild(el('polyline', { points: toPts(pre), fill: 'none', stroke: mo.color, 'stroke-width': sw, 'stroke-linejoin': 'round', opacity: op }));
      if (post.length) {
        if (pre.length) svg.appendChild(el('polyline', { points: toPts([pre[pre.length - 1], post[0]]), fill: 'none', stroke: mo.color, 'stroke-width': sw, opacity: op * .55 }));
        svg.appendChild(el('polyline', { points: toPts(post), fill: 'none', stroke: mo.color, 'stroke-width': sw, 'stroke-linejoin': 'round', opacity: op }));
      }
      const cur = seg[seg.length - 1];
      svg.appendChild(el('circle', { cx: X(cur[0]), cy: Y(cur[1]), r: mo.kind === 'bio' ? 3.6 : 2.9, fill: mo.color }));
    });
    svg.appendChild(el('text', { x: w - m.r, y: m.t + 11, 'text-anchor': 'end', fill: C.bio, 'font-size': 9, 'font-weight': 600, text: 'connectome' }));
    svg.appendChild(el('text', { x: w - m.r, y: m.t + 23, 'text-anchor': 'end', fill: C.ctrl, 'font-size': 9, text: 'random' }));
    host.innerHTML = ''; host.appendChild(svg);
    // live readouts (snap to reported ceilings at the very end)
    const atEnd = tNow >= T - 0.5;
    const rb = document.getElementById('rev-bio'), rc = document.getElementById('rev-ctrl');
    if (rb) rb.textContent = (atEnd ? REV.models[0].ceil2 : revAcc(REV.models[0], tNow, flipAt)).toFixed(3);
    if (rc) rc.textContent = (atEnd ? REV.models[1].ceil2 : revAcc(REV.models[1], tNow, flipAt)).toFixed(3);
  }
  function revStop() { if (revState.id) { clearInterval(revState.id); revState.id = null; } revState.running = false; }
  function revStart(flipAt) {
    revStop();
    revState = { running: true, t: 0, flipAt, id: null };
    drawReversal();
    revState.id = setInterval(() => {
      revState.t += 1.4;
      if (revState.t >= REV.T) { revState.t = REV.T; drawReversal(); revStop(); return; }
      drawReversal();
    }, 26);
  }
  function initReversal() {
    revState.t = REV.T; revState.flipAt = 60;      // show a completed episode by default
    drawReversal();
    const play = document.getElementById('rev-play');
    play && play.addEventListener('click', () => revStart(60));
    const flip = document.getElementById('rev-flip');
    flip && flip.addEventListener('click', () => {
      if (!revState.running || revState.t >= revState.flipAt) revStart(16);   // idle or already flipped: fresh episode, early flip
      else { revState.flipAt = Math.max(6, Math.round(revState.t)); drawReversal(); }  // running & pre-flip: flip right now
      flip.textContent = '⚡ Valence flipped'; setTimeout(() => { flip.textContent = '⚡ Flip valence now'; }, 900);
    });
  }

  /* ---------- continual-learning retention ---------- */
  function drawRetention(progress = 1) {
    const host = document.getElementById('ret-curve'); if (!host) return;
    const w = 380, h = 230, m = { t: 18, r: 20, b: 40, l: 44 };
    const iw = w - m.l - m.r, ih = h - m.t - m.b;
    const X = t => m.l + ((t - 1) / 4) * iw;
    const Y = a => m.t + ih - ((a - 0.6) / 0.42) * ih;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%' });
    [0.6, 0.7, 0.8, 0.9, 1.0].forEach(v => {
      svg.appendChild(el('line', { x1: m.l, y1: Y(v), x2: w - m.r, y2: Y(v), stroke: C.grid }));
      svg.appendChild(el('text', { x: m.l - 6, y: Y(v) + 4, 'text-anchor': 'end', fill: C.mute, 'font-size': 9, 'font-family': 'ui-monospace,monospace', text: v.toFixed(1) }));
    });
    for (let t = 1; t <= 5; t++) svg.appendChild(el('text', { x: X(t), y: h - 22, 'text-anchor': 'middle', fill: C.mute, 'font-size': 9, text: t }));
    svg.appendChild(el('text', { x: m.l + iw / 2, y: h - 6, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'tasks learned' }));
    // real retention curve from the Split-Odor R-matrices (mean accuracy over tasks seen so far)
    const RC = (window.RESULTS && window.RESULTS.continual && window.RESULTS.continual.retentionCurve) || null;
    const models = [
      { name: 'Connectome', color: C.bio, kind: 'bio', curve: RC ? RC.connectome : [0.998, 0.958, 0.918, 0.878, 0.819] },
      { name: 'Weight-shuffle', color: C.blue, curve: RC ? RC.weight_shuffle : [0.997, 0.945, 0.895, 0.845, 0.787] },
      { name: 'Random', color: C.ctrl, curve: RC ? RC.random : [0.994, 0.93, 0.87, 0.81, 0.748] },
    ];
    const shown = 1 + Math.floor(progress * 4 + 0.0001);
    models.forEach(mo => {
      let pts = [];
      for (let t = 1; t <= shown; t++) pts.push([X(t), Y(mo.curve[t - 1])]);
      svg.appendChild(el('polyline', { points: pts.map(p => p.join(',')).join(' '), fill: 'none', stroke: mo.color, 'stroke-width': mo.kind === 'bio' ? 3 : 2, 'stroke-linejoin': 'round' }));
      pts.forEach((p, i) => {
        const dot = el('circle', { cx: p[0], cy: p[1], r: mo.kind === 'bio' ? 3.2 : 2.4, fill: mo.color });
        dot.addEventListener('mousemove', ev => showTip(ev.clientX, ev.clientY, `<div class="tt-h">${mo.name}</div><div class="tt-r">after ${i + 1} task${i ? 's' : ''}: <b>${mo.curve[i].toFixed(3)}</b></div>`));
        dot.addEventListener('mouseleave', hideTip);
        svg.appendChild(dot);
      });
      if (shown === 5) {
        const last = pts[pts.length - 1];
        svg.appendChild(el('text', { x: last[0] - 4, y: last[1] + (mo.kind === 'bio' ? -8 : 12), 'text-anchor': 'end', fill: mo.color, 'font-size': 10, 'font-weight': 600, 'font-family': 'ui-monospace,monospace', text: mo.curve[4].toFixed(3) }));
      }
    });
    host.innerHTML = ''; host.appendChild(svg);
  }
  function initRetention() {
    drawRetention(1);
    const play = document.getElementById('ret-play');
    play && play.addEventListener('click', () => {
      let p = 0; drawRetention(0);
      const id = setInterval(() => { p += 0.02; drawRetention(Math.min(1, p)); if (p >= 1) clearInterval(id); }, 40);
    });
  }

  /* ---------- tabs ---------- */
  function initTabs() {
    const tabs = document.getElementById('mb-tabs'); if (!tabs) return;
    tabs.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      tabs.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      $$('.tabpane').forEach(p => p.classList.toggle('on', p.dataset.pane === b.dataset.tab));
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    mqarBars();
    showEpisode(0);
    if (REAL && REC.testAcc) {
      const rc = document.getElementById('mqar-recall');
      if (rc && !document.getElementById('mqar-acc-note')) {
        const note = document.createElement('figcaption');
        note.id = 'mqar-acc-note'; note.style.marginTop = '8px';
        note.innerHTML = `Recorded from two models trained here on real MQAR episodes: <span class="hl-bio">connectome ${REC.testAcc.connectome.toFixed(3)}</span> vs <span style="color:var(--ctrl)">random ${REC.testAcc.random.toFixed(3)}</span> held-out recall (single seed; the benchmark bars below are the 5-seed means).`;
        rc.parentNode.insertBefore(note, rc.nextSibling);
      }
    }
    document.getElementById('mqar-shuffle').addEventListener('click', newBindings);
    initReversal();
    initRetention();
  });
})();
