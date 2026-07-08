/* ============================================================
   cx.js - Central complex: path-integration sandbox,
           heading ring, error spark, evidence bars, idea diagram
   ============================================================ */
(function () {

  /* ---------- "connectome → network" idea diagram ---------- */
  function ideaDiagram() {
    const host = document.getElementById('idea-diagram');
    if (!host) return;
    const svg = el('svg', { viewBox: '0 0 620 150', width: '100%' });
    // 1. graph
    const gn = [[40, 40], [30, 95], [80, 70], [95, 30], [70, 115]];
    const ge = [[0, 2], [0, 3], [2, 1], [2, 4], [3, 2], [1, 4]];
    ge.forEach(([a, b]) => svg.appendChild(el('line', { x1: gn[a][0], y1: gn[a][1], x2: gn[b][0], y2: gn[b][1], stroke: '#4b9bff', 'stroke-width': 1.4, opacity: .6 })));
    gn.forEach((p, i) => svg.appendChild(el('circle', { cx: p[0], cy: p[1], r: 6, fill: i === 2 ? C.teal : C.bio })));
    svg.appendChild(el('text', { x: 60, y: 142, 'text-anchor': 'middle', fill: C.mute, 'font-size': 11, text: 'connectome' }));
    // arrow
    svg.appendChild(el('text', { x: 150, y: 78, 'text-anchor': 'middle', fill: C.mute, 'font-size': 18, text: '→' }));
    // 2. adjacency matrix
    const ox = 185, oy = 30, cell = 15;
    const M = [[0,0,1,1,0],[0,0,1,0,1],[0,1,0,0,1],[0,0,1,0,0],[0,0,1,1,0]];
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      svg.appendChild(el('rect', { x: ox + c * cell, y: oy + r * cell, width: cell - 2, height: cell - 2, rx: 2, fill: M[r][c] ? C.bio : '#1b2434', opacity: M[r][c] ? .9 : 1 }));
    }
    svg.appendChild(el('text', { x: ox + 32, y: 142, 'text-anchor': 'middle', fill: C.mute, 'font-size': 11, text: 'W' }));
    svg.appendChild(el('text', { x: ox + 40, y: 145, 'text-anchor': 'start', fill: C.mute, 'font-size': 8, text: 'rec' }));
    svg.appendChild(el('text', { x: 300, y: 78, 'text-anchor': 'middle', fill: C.mute, 'font-size': 18, text: '→' }));
    // 3. equation
    const eq = el('foreignObject', { x: 320, y: 54, width: 296, height: 52 });
    const div = document.createElement('div');
    div.style.cssText = 'font-family:var(--sans);font-size:15px;color:var(--ink);line-height:1.35;white-space:nowrap';
    div.innerHTML =
      '<i>h</i><sub>t</sub> = φ(' +
      '<span style="color:var(--bio)"><i>W</i><sub>rec</sub></span>&thinsp;<i>h</i><sub>t&#8722;1</sub>' +
      ' &plus; <i>W</i><sub>in</sub>&thinsp;<i>x</i><sub>t</sub>' +
      ' &plus; <i>b</i>)';
    eq.appendChild(div);
    svg.appendChild(eq);
    svg.appendChild(el('text', { x: 468, y: 142, 'text-anchor': 'middle', fill: C.mute, 'font-size': 11, text: 'recurrent net' }));
    host.innerHTML = ''; host.appendChild(svg);
  }

  /* ---------- evidence bar chart ---------- */
  const cxSeriesMeta = [
    { key: 'cx_bpu', name: 'Connectome', color: C.bio, kind: 'bio' },
    { key: 'weight_shuffle', name: 'Weight-shuffle', color: C.blue },
    { key: 'random', name: 'Random', color: C.ctrl },
    { key: 'no_recurrence', name: 'No recurrence', color: C.red },
  ];
  const pm = (m, s) => (m == null ? ' - ' : (s ? m.toFixed(3) + ' ± ' + s.toFixed(3) : m.toFixed(3)));
  function drawCxBars(regime) {
    const host = document.getElementById('cx-bars'); if (!host) return;
    const RD = (window.RESULTS && window.RESULTS.cxHeading) || DATA.cxHeading;
    const d = RD[regime], Ts = RD.T || DATA.cxHeading.T;
    const val = k => d[k].mean || d[k], err = k => d[k].std;
    mountChart(host, {
      render: c => barChart(c, {
        w: 540, h: 260, yMin: 0.3, yMax: 1.55, dec: 2, ticks: 5,
        aria: `Central-complex heading error by sequence length (${regime})`,
        groups: Ts.map(t => ({ label: 'T = ' + t })),
        series: cxSeriesMeta.map(s => ({ ...s, vals: val(s.key), err: err(s.key) })),
      }),
      table: {
        caption: `Heading-bump angular error (radians, lower is better) - ${regime} reservoir, mean ± std over seeds`,
        cols: ['Model', ...Ts.map(t => 'T=' + t)],
        rows: cxSeriesMeta.map(s => {
          const row = [s.name, ...Ts.map((t, i) => pm(val(s.key)[i], err(s.key) && err(s.key)[i]))];
          row._bio = s.kind === 'bio'; return row;
        }),
      },
    });
    const leg = document.getElementById('cx-legend');
    if (leg) leg.innerHTML = cxSeriesMeta.map(s =>
      `<span class="li"><span class="line" style="border-color:${s.color}"></span>${s.name}</span>`).join('');
  }
  function initCxBars() {
    drawCxBars('frozen');
    const seg = document.getElementById('cx-regime');
    seg && seg.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      seg.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      drawCxBars(b.dataset.r);
    });
  }

  /* ---------- the interactive sandbox ---------- */
  function initSandbox() {
    const stage = document.getElementById('cx-stage');
    const ring = document.getElementById('cx-ring');
    const spark = document.getElementById('cx-spark');
    if (!stage) return;
    const sctx = stage.getContext('2d');
    const rctx = ring.getContext('2d');
    const pctx = spark.getContext('2d');
    let DPR = Math.min(2, devicePixelRatio || 1), W, H, rW, rH, pW, pH;

    function fit(cv, cx) {
      const r = cv.getBoundingClientRect();
      cv.width = r.width * DPR; cv.height = r.height * DPR;
      cx.setTransform(DPR, 0, 0, DPR, 0, 0);
      return [r.width, r.height];
    }
    function resize() { [W, H] = fit(stage, sctx); [rW, rH] = fit(ring, rctx); [pW, pH] = fit(spark, pctx); }

    // world state
    let fly, bio, ctrl, trail, mode = 'drive', dragging = false, target = null, homing = false, errHist = [];
    const TURN = 0.055, SPEED = 1.35;

    function reset() {
      fly = { x: W / 2, y: H / 2, th: -Math.PI / 2 };
      // each estimator: estimated heading + estimated displacement from origin (ex,ey)
      bio = { th: fly.th, ex: 0, ey: 0, err: 0, sig: 0.012 };
      ctrl = { th: fly.th, ex: 0, ey: 0, err: 0, sig: 0.030 };
      trail = [{ x: fly.x, y: fly.y }];
      errHist = [];
      homing = false; target = null;
      document.getElementById('cx-home').disabled = false;
    }

    function noise(sig) { // gaussian-ish
      return (Math.random() + Math.random() + Math.random() - 1.5) * sig;
    }

    let last = 0;
    function step(dt) {
      if (homing) { doHome(dt); return; }
      let turning = 0, walking = 0;
      if (mode === 'auto') {
        fly._w = (fly._w || 0) + (Math.random() - .5) * 0.4;
        fly._w = clamp(fly._w, -1, 1);
        turning = fly._w; walking = 1;
      } else {
        if (keys.left) turning -= 1;
        if (keys.right) turning += 1;
        if (keys.up || dragging) walking = 1;
        if (dragging && target) {
          // steer toward pointer
          let d = Math.atan2(target.y - fly.y, target.x - fly.x) - fly.th;
          while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
          turning = clamp(d / TURN, -1, 1);
        }
      }
      const omega = turning * TURN;
      fly.th += omega;
      // both estimators integrate the same angular velocity, with per-model noise
      bio.th += omega + noise(bio.sig);
      ctrl.th += omega + noise(ctrl.sig);

      if (walking) {
        const v = SPEED;
        fly.x += Math.cos(fly.th) * v; fly.y += Math.sin(fly.th) * v;
        // reflect at walls
        if (fly.x < 12) { fly.x = 12; fly.th = Math.PI - fly.th; }
        if (fly.x > W - 12) { fly.x = W - 12; fly.th = Math.PI - fly.th; }
        if (fly.y < 12) { fly.y = 12; fly.th = -fly.th; }
        if (fly.y > H - 12) { fly.y = H - 12; fly.th = -fly.th; }
        // each estimator integrates displacement along its own heading estimate
        bio.ex += Math.cos(bio.th) * v; bio.ey += Math.sin(bio.th) * v;
        ctrl.ex += Math.cos(ctrl.th) * v; ctrl.ey += Math.sin(ctrl.th) * v;
        trail.push({ x: fly.x, y: fly.y });
        if (trail.length > 900) trail.shift();
      }
      // heading error (circular)
      bio.err = smoothErr(bio.err, angDist(bio.th, fly.th));
      ctrl.err = smoothErr(ctrl.err, angDist(ctrl.th, fly.th));
      errHist.push([bio.err, ctrl.err]); if (errHist.length > 240) errHist.shift();
    }
    function smoothErr(prev, cur) { return prev + (cur - prev) * 0.05; }
    function angDist(a, b) { let d = Math.abs(a - b) % (2 * Math.PI); return d > Math.PI ? 2 * Math.PI - d : d; }

    // homing animation: fly walks toward its connectome-estimated origin
    let homeInfo = null;
    function startHome() {
      if (homing) return;
      homing = true; document.getElementById('cx-home').disabled = true;
      // estimated origin in world coords = flyPos - estimated displacement
      const bioGuess = { x: fly.x - bio.ex, y: fly.y - bio.ey };
      const ctrlGuess = { x: fly.x - ctrl.ex, y: fly.y - ctrl.ey };
      const trueO = { x: W / 2, y: H / 2 };
      homeInfo = { bioGuess, ctrlGuess, trueO,
        bioMiss: Math.hypot(bioGuess.x - trueO.x, bioGuess.y - trueO.y),
        ctrlMiss: Math.hypot(ctrlGuess.x - trueO.x, ctrlGuess.y - trueO.y), t: 0 };
    }
    function doHome(dt) {
      homeInfo.t += 0.02;
      const t = Math.min(1, homeInfo.t);
      const g = homeInfo.bioGuess;
      fly.x = lerp(fly._sx ?? (fly._sx = fly.x), g.x, t);
      fly.y = lerp(fly._sy ?? (fly._sy = fly.y), g.y, t);
      if (t >= 1) {
        homing = false; fly._sx = fly._sy = null;
        const hint = document.getElementById('cx-hint');
        const scale = 0.06; // px→"meters" flavor
        hint.innerHTML = `The <span class="hl-bio">connectome</span> missed home by <b style="color:var(--bio)">${(homeInfo.bioMiss * scale).toFixed(1)}</b> - the <span style="color:var(--ctrl)">random</span> ring by <b style="color:var(--ctrl)">${(homeInfo.ctrlMiss * scale).toFixed(1)}</b>. Less heading drift ⇒ a better fix on home.`;
        document.getElementById('cx-home').disabled = false;
      }
    }

    /* ----- rendering ----- */
    function drawStage() {
      sctx.clearRect(0, 0, W, H);
      // grid
      sctx.strokeStyle = 'rgba(120,150,190,0.06)'; sctx.lineWidth = 1;
      for (let x = 0; x < W; x += 34) { sctx.beginPath(); sctx.moveTo(x, 0); sctx.lineTo(x, H); sctx.stroke(); }
      for (let y = 0; y < H; y += 34) { sctx.beginPath(); sctx.moveTo(0, y); sctx.lineTo(W, y); sctx.stroke(); }
      // true home
      const O = { x: W / 2, y: H / 2 };
      ringMark(sctx, O.x, O.y, C.teal, 'home');
      // trail
      sctx.strokeStyle = 'rgba(170,180,200,0.32)'; sctx.lineWidth = 1.5;
      sctx.beginPath(); trail.forEach((p, i) => i ? sctx.lineTo(p.x, p.y) : sctx.moveTo(p.x, p.y)); sctx.stroke();
      // estimated origins - only shown once the walk has begun (declutters the rest state)
      const bg = { x: fly.x - bio.ex, y: fly.y - bio.ey };
      const cg = { x: fly.x - ctrl.ex, y: fly.y - ctrl.ey };
      const moved = Math.hypot(bio.ex, bio.ey) > 24 || Math.hypot(ctrl.ex, ctrl.ey) > 24;
      if (moved) {
        dot(sctx, cg.x, cg.y, C.ctrl, 4);
        sctx.setLineDash([3, 3]); sctx.strokeStyle = 'rgba(133,146,166,.5)';
        sctx.beginPath(); sctx.moveTo(fly.x, fly.y); sctx.lineTo(cg.x, cg.y); sctx.stroke();
        sctx.strokeStyle = 'rgba(244,177,60,.6)';
        sctx.beginPath(); sctx.moveTo(fly.x, fly.y); sctx.lineTo(bg.x, bg.y); sctx.stroke(); sctx.setLineDash([]);
        dot(sctx, bg.x, bg.y, C.bio, 5);
      }
      // fly
      drawFly(sctx, fly.x, fly.y, fly.th);
      // labels near guesses
      if (moved) {
        sctx.font = '10px ui-monospace,monospace'; sctx.textAlign = 'center';
        sctx.fillStyle = C.bio; sctx.fillText("connectome's home", bg.x, bg.y - 10);
        sctx.fillStyle = C.ctrl; sctx.fillText("random's home", cg.x, cg.y + 16);
      } else {
        sctx.font = '10px ui-monospace,monospace'; sctx.textAlign = 'center';
        sctx.fillStyle = C.mute; sctx.fillText('drag or press ↑ to walk', fly.x, fly.y + 26);
      }
    }
    function ringMark(c, x, y, col, label) {
      c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 9, 0, 7); c.stroke();
      c.fillStyle = col; c.beginPath(); c.arc(x, y, 2.5, 0, 7); c.fill();
      c.font = '10px ui-monospace,monospace'; c.textAlign = 'center'; c.fillStyle = col; c.fillText(label, x, y - 14);
    }
    function dot(c, x, y, col, r) { c.fillStyle = col; c.beginPath(); c.arc(x, y, r, 0, 7); c.fill(); }
    function drawFly(c, x, y, th) {
      c.save(); c.translate(x, y); c.rotate(th);
      c.fillStyle = '#fff'; c.strokeStyle = C.bio; c.lineWidth = 2;
      c.beginPath(); c.moveTo(11, 0); c.lineTo(-6, 6); c.lineTo(-2, 0); c.lineTo(-6, -6); c.closePath();
      c.fill(); c.stroke();
      c.restore();
    }

    function drawRing() {
      rctx.clearRect(0, 0, rW, rH);
      const cx = rW / 2, cy = rH / 2 + 4, R = Math.min(rW, rH) / 2 - 16, N = 28;
      // ring neurons
      for (let i = 0; i < N; i++) {
        const a = (i / N) * 2 * Math.PI - Math.PI / 2;
        // bump = gaussian around bio estimated heading
        let d = angDist(a, bio.th);
        const act = Math.exp(-(d * d) / 0.14);
        const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
        rctx.fillStyle = `rgba(244,177,60,${0.12 + act * 0.88})`;
        rctx.beginPath(); rctx.arc(x, y, 2.5 + act * 4, 0, 7); rctx.fill();
      }
      // true heading tick
      const ta = fly.th;
      rctx.strokeStyle = C.teal; rctx.lineWidth = 2;
      rctx.beginPath(); rctx.moveTo(cx + Math.cos(ta) * (R - 8), cy + Math.sin(ta) * (R - 8));
      rctx.lineTo(cx + Math.cos(ta) * (R + 6), cy + Math.sin(ta) * (R + 6)); rctx.stroke();
      rctx.fillStyle = C.mute; rctx.font = '9px ui-monospace,monospace'; rctx.textAlign = 'center';
      rctx.fillText('bump = heading estimate', cx, cy + R + 12);
    }

    function drawSpark() {
      pctx.clearRect(0, 0, pW, pH);
      const maxE = 0.9, pad = 4;
      pctx.strokeStyle = C.grid; pctx.lineWidth = 1;
      pctx.beginPath(); pctx.moveTo(0, pH - pad); pctx.lineTo(pW, pH - pad); pctx.stroke();
      const draw = (idx, col, w) => {
        pctx.strokeStyle = col; pctx.lineWidth = w; pctx.beginPath();
        errHist.forEach((e, i) => {
          const x = (i / 240) * pW, y = pH - pad - Math.min(1, e[idx] / maxE) * (pH - 2 * pad);
          i ? pctx.lineTo(x, y) : pctx.moveTo(x, y);
        }); pctx.stroke();
      };
      draw(1, C.ctrl, 1.5); draw(0, C.bio, 2);
    }

    function frame(ts) {
      const dt = Math.min(50, ts - last); last = ts;
      step(dt);
      drawStage(); drawRing(); drawSpark();
      document.getElementById('cx-err-bio').textContent = bio.err.toFixed(2);
      document.getElementById('cx-err-ctrl').textContent = ctrl.err.toFixed(2);
      raf = requestAnimationFrame(frame);
    }

    // input
    const keys = { left: false, right: false, up: false };
    function key(e, v) {
      const k = e.key;
      if (k === 'ArrowLeft') { keys.left = v; }
      else if (k === 'ArrowRight') { keys.right = v; }
      else if (k === 'ArrowUp') { keys.up = v; }
      else return;
      if (focused) e.preventDefault();
    }
    let focused = false;
    stage.tabIndex = 0;
    stage.addEventListener('focus', () => focused = true);
    stage.addEventListener('blur', () => { focused = true; keys.left = keys.right = keys.up = false; });
    stage.addEventListener('mouseenter', () => focused = true);
    window.addEventListener('keydown', e => { if (visible) key(e, true); });
    window.addEventListener('keyup', e => { if (visible) key(e, false); });
    function ptr(e) {
      const r = stage.getBoundingClientRect();
      const cx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const cy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      target = { x: cx, y: cy };
    }
    stage.addEventListener('mousedown', e => { dragging = true; ptr(e); stage.focus(); });
    window.addEventListener('mousemove', e => { if (dragging) ptr(e); });
    window.addEventListener('mouseup', () => dragging = false);
    stage.addEventListener('touchstart', e => { dragging = true; ptr(e); e.preventDefault(); }, { passive: false });
    stage.addEventListener('touchmove', e => { if (dragging) { ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchend', () => dragging = false);

    document.getElementById('cx-mode').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      e.currentTarget.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      mode = b.dataset.mode;
    });
    document.getElementById('cx-home').addEventListener('click', startHome);
    document.getElementById('cx-reset').addEventListener('click', () => {
      reset();
      document.getElementById('cx-hint').innerHTML = 'Hold <b>← →</b> to turn, <b>↑</b> to walk (or drag on the arena). Both a <span class="hl-bio">connectome</span> ring and a <span style="color:var(--ctrl)">random</span> ring track your heading - watch their estimates of "home" drift apart.';
    });

    let raf, visible = false;
    window.addEventListener('resize', () => { resize(); });
    resize(); reset();
    const io = new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting;
      if (visible) { if (!raf) { last = performance.now(); raf = requestAnimationFrame(frame); } }
      else { cancelAnimationFrame(raf); raf = null; }
    }), { threshold: 0.05 });
    io.observe(stage);
  }

  document.addEventListener('DOMContentLoaded', () => {
    ideaDiagram();
    initCxBars();
    initSandbox();
  });
})();
