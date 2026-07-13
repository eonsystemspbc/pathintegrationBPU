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
    ge.forEach(([a, b]) => svg.appendChild(el('line', { x1: gn[a][0], y1: gn[a][1], x2: gn[b][0], y2: gn[b][1], stroke: '#7B8FE8', 'stroke-width': 1.4, opacity: .6 })));
    gn.forEach((p, i) => svg.appendChild(el('circle', { cx: p[0], cy: p[1], r: 6, fill: i === 2 ? C.teal : C.bio })));
    svg.appendChild(el('text', { x: 60, y: 142, 'text-anchor': 'middle', fill: C.mute, 'font-size': 11, text: 'connectome' }));
    // arrow
    svg.appendChild(el('text', { x: 150, y: 78, 'text-anchor': 'middle', fill: C.mute, 'font-size': 18, text: '→' }));
    // 2. adjacency matrix
    const ox = 185, oy = 30, cell = 15;
    const M = [[0,0,1,1,0],[0,0,1,0,1],[0,1,0,0,1],[0,0,1,0,0],[0,0,1,1,0]];
    for (let r = 0; r < 5; r++) for (let c = 0; c < 5; c++) {
      svg.appendChild(el('rect', { x: ox + c * cell, y: oy + r * cell, width: cell - 2, height: cell - 2, rx: 2, fill: M[r][c] ? C.bio : '#2A2538', opacity: M[r][c] ? .9 : 1 }));
    }
    svg.appendChild(el('text', { x: ox + 32, y: 142, 'text-anchor': 'middle', fill: C.mute, 'font-size': 11, text: 'W' }));
    svg.appendChild(el('text', { x: ox + 40, y: 145, 'text-anchor': 'start', fill: C.mute, 'font-size': 8, text: 'rec' }));
    svg.appendChild(el('text', { x: 300, y: 78, 'text-anchor': 'middle', fill: C.mute, 'font-size': 18, text: '→' }));
    // 3. equation
    const eq = el('foreignObject', { x: 300, y: 50, width: 318, height: 60 });
    const div = document.createElement('div');
    div.style.cssText = 'font-family:var(--sans);font-size:19px;color:#EDE9F2;line-height:1.35;white-space:nowrap';
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

  /* ---------- the interactive sandbox (live path integration) ----------
     The fly moves in the trained model's own units (DT=1 integration). Every
     ~50ms we push the commanded [v, omega] to the frozen connectome AND the
     size-matched random reservoir on the GPU; each returns a heading bump +
     home vector, which we decode exactly as the benchmark does (sigmoid the
     bump, then circular-mean). Rendering runs at 60fps, interpolating between
     model steps. If the GPU socket is unavailable we fall back to a synthetic
     dead-reckoning illustration so the page still works offline. */
  const CXBINS = 32;
  const BIN_ANG = Array.from({ length: CXBINS }, (_, i) => -Math.PI + 2 * Math.PI * i / CXBINS);
  const sigmoid = x => 1 / (1 + Math.exp(-x));
  function decodeHeading(bump) {           // matches src/train._decode_bump_angle on sigmoid(bump)
    let s = 0, c = 0;
    for (let i = 0; i < CXBINS; i++) { const b = sigmoid(bump[i]); s += b * Math.sin(BIN_ANG[i]); c += b * Math.cos(BIN_ANG[i]); }
    return Math.atan2(s, c);
  }
  function wrapPi(a) { a = (a + Math.PI) % (2 * Math.PI); return (a < 0 ? a + 2 * Math.PI : a) - Math.PI; }
  function angDist(a, b) { return Math.abs(wrapPi(a - b)); }

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

    const PXPU = 5;                         // pixels per model distance unit (arena ~ training range)
    const TICK_MS = 50;                     // model-step cadence (20 Hz)
    const OMEGA = 0.34, V_RUN = 0.95, V_TURN = 0.18;   // model-unit command magnitudes
    let live = false, resetting = false;

    // world state (model frame; origin = home = arena centre)
    // fly.th starts at 0 (model episode frame). bio/ctrl carry the two models' estimates.
    let fly, pfly, bio, ctrl, trail, mode = 'drive', dragging = false, target = null, homing = false, errHist = [];

    // A ring attractor integrates heading over a bounded excursion, not forever: the
    // frozen readout is trained on episodes of ~50-200 steps and its edge over random
    // fades past that. So play is structured into foraging BOUTS - each re-anchors the
    // ring at home - while the running-mean error accumulates across bouts.
    const BOUT = 50;         // ~2.5 s foraging excursion: the horizon where the connectome's edge is clearest
    let stepInBout = 0, boutGen = 0;   // boutGen tags steps so stale in-flight responses are dropped
    function freshEst(sig) { return { th: 0, thTgt: 0, homeAng: 0, homeDist: 0, ex: 0, ey: 0, err: 0, sumErr: 0, cnt: 0, sig }; }
    function newBout(hard) {
      fly = { x: 0, y: 0, th: 0 };
      pfly = { x: 0, y: 0, th: 0 };
      if (hard) { bio = freshEst(0.012); ctrl = freshEst(0.030); errHist = []; }
      else { for (const e of [bio, ctrl]) { e.th = e.thTgt = e.homeAng = e.homeDist = e.ex = e.ey = 0; } }
      trail = [{ x: 0, y: 0 }];
      pendingTh.length = 0; stepInBout = 0; boutGen++; homeResult = null;
      if (window.Live && Live.isOpen() && Live.has('cx')) { resetting = true; Live.call({ op: 'cx_reset' }).catch(() => {}).finally(() => { resetting = false; }); }
    }
    function reset() {
      newBout(true);
      homing = false; target = null;
      document.getElementById('cx-home').disabled = false;
    }
    function noise(sig) { return (Math.random() + Math.random() + Math.random() - 1.5) * sig; }

    // model-unit -> pixel (home at arena centre)
    const toPx = (x, y) => ({ x: W / 2 + x * PXPU, y: H / 2 + y * PXPU });
    // Estimated home = current position minus the displacement the model *thinks* it travelled,
    // integrated from its own per-step heading estimate. That makes the home marker accurate in
    // distance too (as good as the heading tracking), instead of the raw home-distance readout,
    // which saturates. Perfect tracking -> ex/ey == fly -> estimated home lands on the true nest.
    function estOrigin(e) { return { x: fly.x - e.ex, y: fly.y - e.ey }; }

    // ---- one model step (20 Hz) ----
    // auto mode replays the training control distribution (alternating run/turn bouts)
    // so the frozen reservoir stays in-distribution and its real advantage shows.
    const auto = { mode: 'run', left: 0, v: 0.9, omega: 0 };
    const rnd = (a, b) => a + Math.random() * (b - a);
    function nextAuto() {
      if (auto.left <= 0) {
        if (auto.mode === 'run') { auto.mode = 'turn'; auto.left = (2 + Math.random() * 5) | 0; auto.omega = (Math.random() < .5 ? -1 : 1) * rnd(0.18, 0.62); auto.v = rnd(0.05, 0.35); }
        else { auto.mode = 'run'; auto.left = (6 + Math.random() * 12) | 0; auto.omega = (Math.random() - .5) * 0.05; auto.v = rnd(0.55, 1.15); }
      }
      auto.left--;
      return { v: Math.max(0, auto.v + (Math.random() - .5) * 0.06), omega: auto.omega + (Math.random() - .5) * 0.04 };
    }
    function command() {
      if (mode === 'auto') return nextAuto();
      let turning = 0, walking = 0;
      if (keys.left) turning -= 1;
      if (keys.right) turning += 1;
      if (keys.up || dragging) walking = 1;
      if (dragging && target) {
        let d = wrapPi(Math.atan2(target.y - fly.y, target.x - fly.x) - fly.th);
        turning = clamp(d / OMEGA, -1, 1);
      }
      const v = walking ? (Math.abs(turning) > 0.4 ? V_TURN + (V_RUN - V_TURN) * (1 - Math.abs(turning)) : V_RUN) : (Math.abs(turning) > 0 ? V_TURN : 0);
      return { v, omega: turning * OMEGA };
    }
    // keep the fly inside the arena by blocking forward motion at the wall (v->0),
    // which the model sees too, so its integration stays consistent with the truth.
    function bounded(v, th) {
      const nx = fly.x + Math.cos(th) * v, ny = fly.y + Math.sin(th) * v;
      const R = (Math.min(W, H) / 2 - 16) / PXPU;
      return Math.hypot(nx, ny) > R ? 0 : v;
    }

    function tick() {
      if (!visible || resetting) return;   // hold the world still until cx_reset lands (keeps model in sync)
      if (homing) { homeStep(); return; }
      if (homeResult) { if (performance.now() < homeResult.until) return; endResult(); return; }  // hold on the reveal
      if (mode === 'auto') {
        // end the excursion (re-anchor) at the bout limit or near the wall, so every
        // command stays in the trained run/turn distribution where the connectome leads
        const r = Math.hypot(fly.x, fly.y), R = (Math.min(W, H) / 2 - 16) / PXPU;
        if (++stepInBout >= BOUT || r > 0.85 * R) { newBout(false); return; }
      }
      const { v: v0, omega } = command();
      const th = wrapPi(fly.th + omega);
      const v = bounded(v0, th);
      // advance ground truth (model units, DT=1)
      pfly = { x: fly.x, y: fly.y, th: fly.th };
      fly.th = th; fly.x += Math.cos(th) * v; fly.y += Math.sin(th) * v;
      trail.push({ x: fly.x, y: fly.y }); if (trail.length > 1200) trail.shift();

      if (live && !resetting) {
        // responses arrive in order; remember the true heading AND velocity for THIS step so the
        // error pairs the model's estimate with the truth, and the home vector integrates correctly.
        const g = boutGen;
        pendingTh.push({ th: fly.th, v });
        Live.call({ op: 'cx_step', fwd: v, turn: omega }).then(r => applyStep(r, g)).catch(() => { if (g === boutGen) pendingTh.shift(); });
      } else if (!live) {
        // synthetic dead-reckoning fallback
        for (const e of [bio, ctrl]) {
          e.thTgt = e.th = e.th + omega + noise(e.sig);
          e.ex += Math.cos(e.th) * v; e.ey += Math.sin(e.th) * v;
        }
        recordErr(fly.th);
      }
    }
    const pendingTh = [];
    function applyStep(r, g) {
      if (g !== boutGen || !pendingTh.length) return;   // stale response from a finished bout
      const { th: trueTh, v } = pendingTh.shift();
      apply1(bio, r.conn, v); apply1(ctrl, r.rand, v);
      recordErr(trueTh);
    }
    function apply1(e, o, v) {
      e.thTgt = decodeHeading(o.bump);          // heading estimate (absolute, model frame)
      // integrate that heading into an estimated displacement -> the home vector's distance
      e.ex += Math.cos(e.thTgt) * v; e.ey += Math.sin(e.thTgt) * v;
    }
    function recordErr(trueTh) {
      bio.err = angDist(bio.thTgt, trueTh);
      ctrl.err = angDist(ctrl.thTgt, trueTh);
      bio.sumErr += bio.err; bio.cnt++;
      ctrl.sumErr += ctrl.err; ctrl.cnt++;
      errHist.push([bio.err, ctrl.err]); if (errHist.length > 240) errHist.shift();
    }

    // ---- "try to walk home": steer to the connectome's estimated nest, then reveal the miss ----
    let homeInfo = null, homeResult = null;
    function startHome() {
      if (homing || homeResult) return;
      if (Math.hypot(fly.x, fly.y) < 3) return;         // already essentially home
      homing = true; document.getElementById('cx-home').disabled = true;
      const bg = estOrigin(bio), cg = estOrigin(ctrl);
      homeInfo = { bg, cg, sx: fly.x, sy: fly.y, sth: fly.th, t: 0,
        bioMiss: Math.hypot(bg.x, bg.y), ctrlMiss: Math.hypot(cg.x, cg.y) };   // model-unit distance to true nest
    }
    function homeStep() {
      homeInfo.t = Math.min(1, homeInfo.t + 0.022);
      const t = homeInfo.t, e = t * t * (3 - 2 * t);    // smoothstep
      pfly = { x: fly.x, y: fly.y, th: fly.th };
      fly.x = lerp(homeInfo.sx, homeInfo.bg.x, e);
      fly.y = lerp(homeInfo.sy, homeInfo.bg.y, e);
      const dir = Math.atan2(homeInfo.bg.y - homeInfo.sy, homeInfo.bg.x - homeInfo.sx);   // face the walk
      fly.th = homeInfo.sth + wrapPi(dir - homeInfo.sth) * Math.min(1, t * 4);
      if (t >= 1) {
        homing = false;
        homeResult = { bg: homeInfo.bg, cg: homeInfo.cg, bioMiss: homeInfo.bioMiss, ctrlMiss: homeInfo.ctrlMiss, until: performance.now() + 3600 };
        const tighter = homeInfo.bioMiss <= homeInfo.ctrlMiss;
        document.getElementById('cx-hint').innerHTML =
          `The <span class="hl-bio">connectome's</span> guess of home was <b style="color:var(--bio)">${homeInfo.bioMiss.toFixed(1)}</b> off the true nest; the <span style="color:var(--ctrl)">random</span> reservoir's was <b style="color:var(--ctrl)">${homeInfo.ctrlMiss.toFixed(1)}</b> off.`
          + (tighter ? ' Less heading drift &rArr; a tighter fix on home.' : '');
        document.getElementById('cx-home').disabled = false;
      }
    }
    function endResult() { if (homeResult) { homeResult = null; newBout(false); } }

    /* ----- rendering (60fps, interpolated between 20 Hz model steps) ----- */
    function drawStage(alpha) {
      sctx.clearRect(0, 0, W, H);
      sctx.strokeStyle = 'rgba(120,150,190,0.06)'; sctx.lineWidth = 1;
      for (let x = 0; x < W; x += 34) { sctx.beginPath(); sctx.moveTo(x, 0); sctx.lineTo(x, H); sctx.stroke(); }
      for (let y = 0; y < H; y += 34) { sctx.beginPath(); sctx.moveTo(0, y); sctx.lineTo(W, y); sctx.stroke(); }
      const O = toPx(0, 0);
      ringMark(sctx, O.x, O.y, C.teal, 'home');
      // trail
      sctx.strokeStyle = 'rgba(170,180,200,0.32)'; sctx.lineWidth = 1.5;
      sctx.beginPath(); trail.forEach((p, i) => { const q = toPx(p.x, p.y); i ? sctx.lineTo(q.x, q.y) : sctx.moveTo(q.x, q.y); }); sctx.stroke();
      // interpolated fly pose
      const ix = lerp(pfly.x, fly.x, alpha), iy = lerp(pfly.y, fly.y, alpha);
      const ith = pfly.th + wrapPi(fly.th - pfly.th) * alpha;
      const fp = toPx(ix, iy);
      const moved = Math.hypot(fly.x, fly.y) > 4;
      sctx.font = '10px ui-monospace,monospace'; sctx.textAlign = 'center';
      if (homeResult) {
        // reveal: both guesses vs the true nest, each with its miss distance
        const bg = toPx(homeResult.bg.x, homeResult.bg.y), cg = toPx(homeResult.cg.x, homeResult.cg.y);
        const pr = 12 + 4 * Math.sin(performance.now() / 260);
        sctx.strokeStyle = C.teal; sctx.lineWidth = 1.5; sctx.globalAlpha = .7;
        sctx.beginPath(); sctx.arc(O.x, O.y, pr, 0, 7); sctx.stroke(); sctx.globalAlpha = 1;
        sctx.setLineDash([4, 4]); sctx.lineWidth = 1.5;
        sctx.strokeStyle = 'rgba(133,146,166,.55)';
        sctx.beginPath(); sctx.moveTo(cg.x, cg.y); sctx.lineTo(O.x, O.y); sctx.stroke();
        dot(sctx, cg.x, cg.y, C.ctrl, 4);
        sctx.strokeStyle = 'rgba(212,160,176,.85)';
        sctx.beginPath(); sctx.moveTo(bg.x, bg.y); sctx.lineTo(O.x, O.y); sctx.stroke(); sctx.setLineDash([]);
        dot(sctx, bg.x, bg.y, C.bio, 5);
        sctx.fillStyle = C.ctrl; sctx.fillText(`random · ${homeResult.ctrlMiss.toFixed(1)} off`, cg.x, cg.y + 16);
        sctx.fillStyle = C.bio; sctx.fillText(`connectome · ${homeResult.bioMiss.toFixed(1)} off`, bg.x, bg.y - 12);
      } else if (moved) {
        // live estimated nests, at accurate distance now (integrated from each model's heading)
        const bg = toPx(estOrigin(bio).x, estOrigin(bio).y), cg = toPx(estOrigin(ctrl).x, estOrigin(ctrl).y);
        dot(sctx, cg.x, cg.y, C.ctrl, 4);
        sctx.setLineDash([3, 3]); sctx.strokeStyle = 'rgba(133,146,166,.5)';
        sctx.beginPath(); sctx.moveTo(fp.x, fp.y); sctx.lineTo(cg.x, cg.y); sctx.stroke();
        sctx.strokeStyle = 'rgba(212,160,176,.6)';
        sctx.beginPath(); sctx.moveTo(fp.x, fp.y); sctx.lineTo(bg.x, bg.y); sctx.stroke(); sctx.setLineDash([]);
        dot(sctx, bg.x, bg.y, C.bio, 5);
        sctx.fillStyle = C.bio; sctx.fillText("connectome's home", bg.x, bg.y - 10);
        sctx.fillStyle = C.ctrl; sctx.fillText("random's home", cg.x, cg.y + 16);
      }
      drawFly(sctx, fp.x, fp.y, ith);
      if (!moved && !homeResult) {
        sctx.fillStyle = C.mute; sctx.fillText('drag or press ↑ to walk', fp.x, fp.y + 26);
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
      const cx = rW / 2, cy = rH / 2 + 4, R = Math.min(rW, rH) / 2 - 16, N = 32;
      // smooth the displayed bump toward the model's estimate
      bio.th += wrapPi(bio.thTgt - bio.th) * 0.35;
      ctrl.th += wrapPi(ctrl.thTgt - ctrl.th) * 0.35;
      // random reservoir bump (faint), then connectome bump (bright)
      for (let i = 0; i < N; i++) {
        const a = (i / N) * 2 * Math.PI - Math.PI / 2;
        const d = angDist(a, ctrl.th), act = Math.exp(-(d * d) / 0.14);
        const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
        rctx.fillStyle = `rgba(133,146,166,${0.10 + act * 0.5})`;
        rctx.beginPath(); rctx.arc(x, y, 2 + act * 3, 0, 7); rctx.fill();
      }
      for (let i = 0; i < N; i++) {
        const a = (i / N) * 2 * Math.PI - Math.PI / 2;
        const d = angDist(a, bio.th), act = Math.exp(-(d * d) / 0.14);
        const x = cx + Math.cos(a) * R, y = cy + Math.sin(a) * R;
        rctx.fillStyle = `rgba(212,160,176,${0.12 + act * 0.88})`;
        rctx.beginPath(); rctx.arc(x, y, 2.5 + act * 4, 0, 7); rctx.fill();
      }
      // true heading tick
      rctx.strokeStyle = C.teal; rctx.lineWidth = 2;
      rctx.beginPath(); rctx.moveTo(cx + Math.cos(fly.th) * (R - 8), cy + Math.sin(fly.th) * (R - 8));
      rctx.lineTo(cx + Math.cos(fly.th) * (R + 6), cy + Math.sin(fly.th) * (R + 6)); rctx.stroke();
      rctx.fillStyle = C.mute; rctx.font = '9px ui-monospace,monospace'; rctx.textAlign = 'center';
      rctx.fillText('bump = heading estimate', cx, cy + R + 12);
    }
    function drawSpark() {
      pctx.clearRect(0, 0, pW, pH);
      const maxE = Math.PI, pad = 4;
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

    let last = 0, acc = 0;
    function frame(ts) {
      const dt = Math.min(80, ts - last); last = ts; acc += dt;
      while (acc >= TICK_MS) { tick(); acc -= TICK_MS; }
      const alpha = homing ? 1 : clamp(acc / TICK_MS, 0, 1);
      drawStage(alpha); drawRing(); drawSpark();
      // show the running-mean heading error: the frozen connectome's ~8-11% edge is an
      // average, so it only reads cleanly once enough steps have accumulated.
      document.getElementById('cx-err-bio').textContent = (bio.cnt ? bio.sumErr / bio.cnt : 0).toFixed(2);
      document.getElementById('cx-err-ctrl').textContent = (ctrl.cnt ? ctrl.sumErr / ctrl.cnt : 0).toFixed(2);
      raf = requestAnimationFrame(frame);
    }

    // input
    const keys = { left: false, right: false, up: false };
    function key(e, v) {
      const k = e.key;
      if (k === 'ArrowLeft') keys.left = v;
      else if (k === 'ArrowRight') keys.right = v;
      else if (k === 'ArrowUp') keys.up = v;
      else return;
      if (v) endResult();                    // interrupt a home-result reveal
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
      const px = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const py = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      target = { x: (px - W / 2) / PXPU, y: (py - H / 2) / PXPU };    // pointer in model units
    }
    stage.addEventListener('mousedown', e => { endResult(); dragging = true; ptr(e); stage.focus(); });
    window.addEventListener('mousemove', e => { if (dragging) ptr(e); });
    window.addEventListener('mouseup', () => dragging = false);
    stage.addEventListener('touchstart', e => { endResult(); dragging = true; ptr(e); e.preventDefault(); }, { passive: false });
    stage.addEventListener('touchmove', e => { if (dragging) { ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchend', () => dragging = false);

    document.getElementById('cx-mode').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      e.currentTarget.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      mode = b.dataset.mode; endResult();
    });
    document.getElementById('cx-home').addEventListener('click', startHome);
    const baseHint = 'Hold <b>← →</b> to turn, <b>↑</b> to walk (or drag on the arena). A frozen <span class="hl-bio">connectome</span> ring and a size-matched <span style="color:var(--ctrl)">random</span> ring each track your heading live on the GPU - watch their estimates of "home" drift apart.';
    document.getElementById('cx-reset').addEventListener('click', () => { reset(); document.getElementById('cx-hint').innerHTML = baseHint; });

    // connect/refresh live status
    function refreshLive() {
      const on = !!(window.Live && Live.isOpen() && Live.has('cx'));
      if (on && !live) { live = true; reset(); }
      else live = on;
    }
    if (window.Live) Live.onState(refreshLive);
    refreshLive();

    let raf, visible = false;
    window.addEventListener('resize', () => resize());
    resize(); reset();
    const io = new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting;
      if (visible) { if (!raf) { last = performance.now(); acc = 0; raf = requestAnimationFrame(frame); } }
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
