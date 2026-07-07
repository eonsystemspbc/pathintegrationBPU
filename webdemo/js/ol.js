/* ============================================================
   ol.js - Optic lobe: hex-lattice optic-flow sandbox +
           data-efficiency curve
   ============================================================ */
(function () {
  const OF = (window.RESULTS && window.RESULTS.opticFlow) || DATA.opticFlow;
  const FRACS = OF.fraction;
  const FAM = {
    sparse: { c: 'sparse_connectome', r: 'sparse_random' },
    pruned: { c: 'pruned_connectome', r: 'pruned_random' },
    dense:  { c: 'dense_connectome',  r: 'dense_random' },
  };
  let family = 'sparse', fracIdx = FRACS.length - 1;
  const famData = which => OF.families[FAM[family][which]];
  const rmse = which => { const f = famData(which); return (f.mean || f)[fracIdx]; };
  const pm3 = (m, s) => (m == null ? ' - ' : (s ? m.toFixed(4) + ' ± ' + s.toFixed(4) : m.toFixed(4)));

  /* ---------- data-efficiency line chart (real means + ±std bands) ---------- */
  function drawChart() {
    const host = document.getElementById('ol-chart'); if (!host) return;
    const cf = famData('c'), rf = famData('r');
    mountChart(host, {
      render: c => lineChart(c, {
        w: 540, h: 280, yMin: 0.11, yMax: 0.23, dec: 3, ticks: 4,
        x: FRACS, xlabel: 'training-data fraction (%)', xunit: '%', markX: FRACS[fracIdx],
        aria: `Optic-flow RMSE by training-data fraction (${family} family)`,
        series: [
          { name: 'Connectome', color: C.bio, kind: 'bio', vals: cf.mean || cf, band: cf.std },
          { name: 'Random', color: C.ctrl, kind: 'ctrl', dash: '5 4', vals: rf.mean || rf, band: rf.std },
        ],
      }),
      table: {
        caption: `Optic-flow test RMSE (lower is better) - ${family} family, mean ± std over 3 seeds`,
        cols: ['Fraction %', 'Connectome', 'Random'],
        rows: FRACS.map((fr, i) => [fr, pm3((cf.mean || cf)[i], cf.std && cf.std[i]), pm3((rf.mean || rf)[i], rf.std && rf.std[i])]),
      },
    });
  }
  function updateTiles() {
    document.getElementById('ol-rmse-bio').textContent = fmt(rmse('c'), 3);
    document.getElementById('ol-rmse-ctrl').textContent = fmt(rmse('r'), 3);
  }

  /* ---------- hex sensor sandbox ---------- */
  function initSandbox() {
    const stage = document.getElementById('ol-stage'); if (!stage) return;
    const ctx = stage.getContext('2d');
    let DPR = Math.min(2, devicePixelRatio || 1), W, H, hexes = [], dots = [];
    let vel = { x: 0.0, y: -0.9 }, targetVel = { x: 0, y: -0.9 }, scene = 'drag', dragging = false, lastP = null, raf, visible = false;

    function resize() {
      const r = stage.getBoundingClientRect();
      stage.width = r.width * DPR; stage.height = r.height * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      W = r.width; H = r.height;
      buildHex(); buildDots();
    }
    function buildHex() {
      hexes = [];
      const R = 22, dx = R * 1.73, dy = R * 1.5;
      for (let row = 0, y = 30; y < H - 20; y += dy, row++) {
        for (let x = 30 + (row % 2 ? dx / 2 : 0); x < W - 20; x += dx) hexes.push({ x, y });
      }
    }
    function buildDots() {
      dots = [];
      for (let i = 0; i < 90; i++) dots.push({ x: Math.random() * W, y: Math.random() * H, r: 1 + Math.random() * 2 });
    }
    function hexPath(x, y, R) {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) { const a = Math.PI / 180 * (60 * i - 30); const px = x + R * Math.cos(a), py = y + R * Math.sin(a); i ? ctx.lineTo(px, py) : ctx.moveTo(px, py); }
      ctx.closePath();
    }

    function fracNoise() { return (FRACS.length - 1 - fracIdx) / (FRACS.length - 1); } // 0 at 100%, 1 at 5%

    function frame() {
      // ease velocity
      if (scene === 'auto') {
        targetVel.x = Math.cos(performance.now() / 2600) * 1.1;
        targetVel.y = Math.sin(performance.now() / 1900) * 1.1 - 0.2;
      }
      vel.x += (targetVel.x - vel.x) * 0.08; vel.y += (targetVel.y - vel.y) * 0.08;
      // move dots by world velocity (scene slides)
      for (const d of dots) {
        d.x += vel.x; d.y += vel.y;
        if (d.x < 0) d.x += W; if (d.x > W) d.x -= W;
        if (d.y < 0) d.y += H; if (d.y > H) d.y -= H;
      }
      ctx.clearRect(0, 0, W, H);
      // dots
      ctx.fillStyle = 'rgba(170,180,200,0.5)';
      for (const d of dots) { ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7); ctx.fill(); }
      // hex sensor + local flow arrows
      const sp = Math.hypot(vel.x, vel.y);
      for (const h of hexes) {
        hexPath(h.x, h.y, 20);
        ctx.strokeStyle = 'rgba(75,155,255,0.14)'; ctx.lineWidth = 1; ctx.stroke();
        // local flow = -vel (world slides one way ⇒ retinal flow the other)
        const fx = -vel.x, fy = -vel.y, fmag = Math.hypot(fx, fy);
        if (fmag > 0.05) {
          const s = 9 / (fmag + 0.001);
          ctx.strokeStyle = `rgba(47,212,198,${0.25 + Math.min(0.6, fmag * 0.5)})`; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.moveTo(h.x, h.y); ctx.lineTo(h.x + fx * s, h.y + fy * s); ctx.stroke();
          // arrowhead
          const a = Math.atan2(fy, fx);
          ctx.beginPath(); ctx.moveTo(h.x + fx * s, h.y + fy * s);
          ctx.lineTo(h.x + fx * s - 4 * Math.cos(a - .4), h.y + fy * s - 4 * Math.sin(a - .4));
          ctx.moveTo(h.x + fx * s, h.y + fy * s);
          ctx.lineTo(h.x + fx * s - 4 * Math.cos(a + .4), h.y + fy * s - 4 * Math.sin(a + .4)); ctx.stroke();
        }
      }
      // ego-motion estimates at center
      const cx = W / 2, cy = H / 2, trueA = Math.atan2(-vel.y, -vel.x), L = 46 * Math.min(1, sp);
      const nz = fracNoise();
      const bioA = trueA + (Math.random() - .5) * 0.10 * nz;
      const ctrlA = trueA + (Math.random() - .5) * 0.34 * nz;
      const bioL = L * (1 + (Math.random() - .5) * 0.10 * nz);
      const ctrlL = L * (1 + (Math.random() - .5) * 0.30 * nz);
      arrow(cx, cy, trueA, L + 6, C.teal, 3);          // truth
      arrow(cx, cy, ctrlA, ctrlL, C.ctrl, 2.2);        // random
      arrow(cx, cy, bioA, bioL, C.bio, 2.6);           // connectome
      ctx.fillStyle = C.mute; ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'left';
      ctx.fillText('ego-motion', cx + 8, cy - 8);
      // motion text
      const yaw = (vel.x * 40).toFixed(0), fwd = (-vel.y).toFixed(2);
      const mt = document.getElementById('ol-motion');
      if (mt) mt.innerHTML = sp < 0.06 ? ' - idle - ' : `slip <b style="color:var(--teal)">${(sp).toFixed(2)}</b> · yaw <b>${yaw}</b>°/s`;
      if (visible) raf = requestAnimationFrame(frame);
    }
    function arrow(x, y, a, len, col, w) {
      const ex = x + Math.cos(a) * len, ey = y + Math.sin(a) * len;
      ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(ex, ey); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 7 * Math.cos(a - .4), ey - 7 * Math.sin(a - .4)); ctx.moveTo(ex, ey);
      ctx.lineTo(ex - 7 * Math.cos(a + .4), ey - 7 * Math.sin(a + .4)); ctx.stroke();
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
    }

    function ptr(e) {
      const r = stage.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      if (lastP) { targetVel.x = clamp((x - lastP.x) * 0.5, -3, 3); targetVel.y = clamp((y - lastP.y) * 0.5, -3, 3); }
      lastP = { x, y };
    }
    stage.addEventListener('mousedown', e => { if (scene === 'drag') { dragging = true; lastP = null; ptr(e); } });
    window.addEventListener('mousemove', e => { if (dragging) ptr(e); });
    window.addEventListener('mouseup', () => { dragging = false; lastP = null; });
    stage.addEventListener('touchstart', e => { if (scene === 'drag') { dragging = true; lastP = null; ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchmove', e => { if (dragging) { ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchend', () => { dragging = false; lastP = null; });
    // keyboard access
    stage.tabIndex = 0;
    stage.setAttribute('role', 'application');
    stage.setAttribute('aria-label', 'Optic-flow sensor. Drag, or use arrow keys, to move the scene and read out self-motion.');
    stage.addEventListener('keydown', e => {
      const v = { ArrowLeft: [-1.4, 0], ArrowRight: [1.4, 0], ArrowUp: [0, -1.4], ArrowDown: [0, 1.4] }[e.key];
      if (!v) return;
      targetVel.x = v[0]; targetVel.y = v[1]; e.preventDefault();
    });

    document.getElementById('ol-scene').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      e.currentTarget.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      scene = b.dataset.s;
    });

    window.addEventListener('resize', resize);
    resize();
    const io = new IntersectionObserver(es => es.forEach(e => {
      visible = e.isIntersecting;
      if (visible) { if (!raf) raf = requestAnimationFrame(frame); }
      else { cancelAnimationFrame(raf); raf = null; }
    }), { threshold: 0.05 });
    io.observe(stage);
  }

  document.addEventListener('DOMContentLoaded', () => {
    drawChart(); updateTiles();
    const slider = document.getElementById('ol-frac');
    slider && slider.addEventListener('input', () => {
      fracIdx = +slider.value;
      document.getElementById('ol-frac-lbl').textContent = FRACS[fracIdx] + '%';
      updateTiles(); drawChart();
    });
    const fam = document.getElementById('ol-family');
    fam && fam.addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      fam.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      family = b.dataset.f; updateTiles(); drawChart();
    });
    initSandbox();
  });
})();
