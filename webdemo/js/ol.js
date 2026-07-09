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
    // data-efficiency RMSE at the current training-data fraction (responds to the slider)
    const o = document.getElementById('ol-overall');
    if (o) o.innerHTML = `connectome <b style="color:var(--bio)">${fmt(rmse('c'), 3)}</b> · random <b style="color:var(--ctrl)">${fmt(rmse('r'), 3)}</b>`;
  }

  /* ---------- hex sensor sandbox (live optic-flow inference) ----------
     Commands a fly-like ego-motion (yaw / forward / lateral); the server renders a
     real hex-lattice glimpse of that motion over a panorama and runs BOTH the
     optic-lobe connectome and a size-matched random reservoir, which read out
     [yaw, forward, lateral]. We draw the actual ommatidia the models see plus each
     model's ego-motion estimate against the truth. Falls back to a synthetic
     illustration when the GPU socket is unavailable. */
  const MAXY = 0.25, MAXF = 0.55, MAXL = 0.35;      // spec max rates
  function initSandbox() {
    const stage = document.getElementById('ol-stage'); if (!stage) return;
    const ctx = stage.getContext('2d');
    let DPR = Math.min(2, devicePixelRatio || 1), W, H, dots = [];
    let scene = 'drag', dragging = false, lastP = null, raf, visible = false;
    let live = false, lattice = null, latest = null, stepping = false, lastStep = 0;
    const cmd = { yaw: 0, fwd: 0, lat: 0 }, tgt = { yaw: 0, fwd: 0, lat: 0 };
    const errAcc = { c: 0, r: 0, n: 0 };     // running-mean tracking RMSE (the connectome's edge is an average)
    // first-person view of the real panorama the model samples
    let panoCv = null, pw = 0, ph = 0, fovAz = 2.6, fovEl = 1.66;
    const view = { yaw: 0, fwd: 0, lat: 0 };   // accumulated pose for smooth visual flow (display only)
    // smoothed estimate arrows (screen-space), so display stays fluid between glimpses
    const shown = { truth: [0, 0, 0], conn: [0, 0, 0], rand: [0, 0, 0] };

    function resize() {
      const r = stage.getBoundingClientRect();
      stage.width = r.width * DPR; stage.height = r.height * DPR;
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      W = r.width; H = r.height;
      buildDots();
    }
    function buildDots() { dots = []; for (let i = 0; i < 90; i++) dots.push({ x: Math.random() * W, y: Math.random() * H, r: 1 + Math.random() * 2 }); }
    function fracNoise() { return (FRACS.length - 1 - fracIdx) / (FRACS.length - 1); }

    function refreshLive() {
      const on = !!(window.Live && Live.isOpen() && Live.has('ol'));
      if (on && !live) { live = true; olReset(); } else live = on;
    }
    function olReset() {
      errAcc.c = errAcc.r = errAcc.n = 0;
      Live.call({ op: 'ol_reset', seed: (Math.random() * 1e6) | 0 }).then(r => {
        lattice = r.lattice;
        if (r.pano) { fovAz = r.fov_az * Math.PI / 180; fovEl = r.fov_el * Math.PI / 180; buildPano(r.pano, r.pw, r.ph); }
      }).catch(() => {});
    }
    function buildPano(b64, w, h) {
      const bin = atob(b64), off = document.createElement('canvas');
      off.width = w; off.height = h;
      const octx = off.getContext('2d'), img = octx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) {
        const v = bin.charCodeAt(i);                 // cool-tinted grayscale sky/ground panorama
        img.data[i * 4] = Math.round(v * 0.74); img.data[i * 4 + 1] = Math.round(v * 0.85);
        img.data[i * 4 + 2] = Math.min(255, Math.round(v * 1.02 + 16)); img.data[i * 4 + 3] = 255;
      }
      octx.putImageData(img, 0, 0);
      panoCv = off; pw = w; ph = h;
    }
    function maybeStep(ts) {
      if (!live || !lattice || stepping || ts - lastStep < 70) return;   // ~14 Hz glimpses
      lastStep = ts; stepping = true;
      Live.call({ op: 'ol_step', yaw: cmd.yaw, fwd: cmd.fwd, lat: cmd.lat })
        .then(r => {
          latest = r;
          // only accumulate error when the scene is actually moving (idle glimpses are trivial)
          if (Math.hypot(r.truth[0], r.truth[1], r.truth[2]) > 0.08) { errAcc.c += r.conn_err; errAcc.r += r.rand_err; errAcc.n++; }
        }).catch(() => {}).finally(() => { stepping = false; });
    }

    // ---- first-person view: the real panorama the model samples, panning with yaw,
    //      expanding with forward, shifting with lateral (visual accumulation only) ----
    function drawScene() {
      const zoom = 1 + clamp(view.fwd, -0.7, 1.0) * 0.22;
      const srcW = (fovAz / (2 * Math.PI)) * pw / zoom;
      let srcH = Math.min((fovEl / Math.PI) * ph / zoom, ph);
      const c = (((view.yaw + view.lat * 0.5) / (2 * Math.PI)) % 1 + 1) % 1;   // 0..1 centre in panorama
      let srcX = c * pw - srcW / 2;
      const srcY = clamp(ph / 2 - srcH / 2 - view.fwd * 3, 0, ph - srcH);
      srcX = ((srcX % pw) + pw) % pw;
      ctx.imageSmoothingEnabled = true;
      if (srcX + srcW <= pw) ctx.drawImage(panoCv, srcX, srcY, srcW, srcH, 0, 0, W, H);
      else { const w1 = pw - srcX, f = w1 / srcW; ctx.drawImage(panoCv, srcX, srcY, w1, srcH, 0, 0, W * f, H); ctx.drawImage(panoCv, 0, srcY, srcW - w1, srcH, W * f, 0, W * (1 - f), H); }
      // vignette so the sensor overlay reads
      const g = ctx.createRadialGradient(W / 2, H / 2, Math.min(W, H) * 0.25, W / 2, H / 2, Math.max(W, H) * 0.62);
      g.addColorStop(0, 'rgba(6,10,18,0)'); g.addColorStop(1, 'rgba(6,10,18,0.55)');
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    }
    // compound-eye facets, head-fixed, drawn over the scene (this is what the model reads)
    function facetPos(az, el) {
      return { x: W / 2 + (az / (fovAz / 2)) * (W * 0.46), y: H / 2 - (el / (fovEl / 2)) * (H * 0.42) };
    }
    function drawFacets() {
      if (!lattice) return;
      for (let i = 0; i < lattice.length; i++) {
        const p = facetPos(lattice[i][0], lattice[i][1]);
        const v = latest ? latest.hex[i] : 0.5;
        ctx.beginPath();
        for (let k = 0; k < 6; k++) { const a = Math.PI / 180 * (60 * k - 30), x = p.x + 8.5 * Math.cos(a), y = p.y + 8.5 * Math.sin(a); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
        ctx.closePath();
        ctx.fillStyle = `rgba(210,232,255,${0.05 + 0.16 * v})`; ctx.fill();
        ctx.strokeStyle = 'rgba(130,185,255,0.22)'; ctx.lineWidth = 1; ctx.stroke();
      }
    }
    function drawSyntheticScene() {
      const sp = Math.hypot(cmd.lat, cmd.fwd) + Math.abs(cmd.yaw);
      for (const d of dots) {
        d.x += (cmd.lat * 40 - cmd.yaw * 60); d.y += (-cmd.fwd * 40);
        if (d.x < 0) d.x += W; if (d.x > W) d.x -= W; if (d.y < 0) d.y += H; if (d.y > H) d.y -= H;
      }
      ctx.fillStyle = 'rgba(170,180,200,0.5)';
      for (const d of dots) { ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, 7); ctx.fill(); }
      return sp;
    }

    // ---- ego-motion arrows: m = [yaw, forward, lateral] ----
    function drawEgo() {
      const cx = W / 2, cy = H * 0.86, S = 90;
      const arrow = (m, col, w) => {
        const ex = cx + m[2] * S, ey = cy - m[1] * S;   // lateral -> x, forward -> up
        ctx.strokeStyle = col; ctx.lineWidth = w; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(ex, ey); ctx.stroke();
        const a = Math.atan2(ey - cy, ex - cx);
        ctx.beginPath(); ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 7 * Math.cos(a - .4), ey - 7 * Math.sin(a - .4)); ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 7 * Math.cos(a + .4), ey - 7 * Math.sin(a + .4)); ctx.stroke();
      };
      ctx.fillStyle = C.teal; ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 7); ctx.fill();
      arrow(shown.truth, C.teal, 3); arrow(shown.rand, C.ctrl, 2.2); arrow(shown.conn, C.bio, 2.6);
      ctx.font = '10px ui-monospace,monospace'; ctx.textAlign = 'center';
      ctx.fillStyle = C.mute; ctx.fillText('translation', cx, cy + 16);
    }

    function frame(ts) {
      if (scene === 'auto') { tgt.yaw = Math.sin(ts / 1700) * MAXY * 0.9; tgt.fwd = Math.cos(ts / 2300) * MAXF * 0.7 + 0.06; tgt.lat = Math.sin(ts / 3100) * MAXL * 0.8; }
      for (const k of ['yaw', 'fwd', 'lat']) cmd[k] += (tgt[k] - cmd[k]) * 0.1;
      // accumulate the display pose so the world flows smoothly (decoupled from the model)
      view.yaw += cmd.yaw * 0.11; view.fwd += (cmd.fwd - view.fwd) * 0.12; view.lat += (cmd.lat - view.lat) * 0.12;
      maybeStep(ts);
      ctx.clearRect(0, 0, W, H);

      let motionTxt;
      if (live && lattice) {
        if (panoCv) { drawScene(); drawFacets(); }
        else { ctx.fillStyle = '#0F0D15'; ctx.fillRect(0, 0, W, H); drawFacets(); }
        // ease the shown arrows toward the model's latest estimates
        if (latest) for (const key of ['truth', 'conn', 'rand']) for (let i = 0; i < 3; i++) shown[key][i] += (latest[key][i] - shown[key][i]) * 0.25;
        drawEgo();
        // live running-mean tracking error -> the two prominent tiles (converges to the connectome's edge)
        const eb = document.getElementById('ol-err-bio'), ecc = document.getElementById('ol-err-ctrl');
        if (errAcc.n >= 8) {
          if (eb) eb.textContent = (errAcc.c / errAcc.n).toFixed(3);
          if (ecc) ecc.textContent = (errAcc.r / errAcc.n).toFixed(3);
        }
        // live-motion line: the current commanded ego-motion
        const t = latest ? latest.truth : [cmd.yaw, cmd.fwd, cmd.lat];
        const sp = Math.hypot(t[0], t[1], t[2]);
        motionTxt = sp < 0.05 ? '<span class="live-dot"></span>idle &middot; drag or auto-drift' :
          `<span class="live-dot"></span>tracking &middot; yaw <b>${t[0].toFixed(2)}</b> &middot; fwd <b>${t[1].toFixed(2)}</b>`;
      } else {
        const sp = drawSyntheticScene();
        const nz = fracNoise(), t = [cmd.yaw, cmd.fwd, cmd.lat];
        shown.truth = t;
        shown.conn = [t[0], t[1] * (1 + (Math.random() - .5) * 0.1 * nz), t[2] + (Math.random() - .5) * 0.06 * nz];
        shown.rand = [t[0], t[1] * (1 + (Math.random() - .5) * 0.3 * nz), t[2] + (Math.random() - .5) * 0.20 * nz];
        drawEgo();
        // offline fallback: tiles show the data-efficiency RMSE
        const eb = document.getElementById('ol-err-bio'), ecc = document.getElementById('ol-err-ctrl');
        if (eb) eb.textContent = fmt(rmse('c'), 3);
        if (ecc) ecc.textContent = fmt(rmse('r'), 3);
        motionTxt = sp < 0.05 ? ' - idle - ' : `slip <b style="color:var(--teal)">${sp.toFixed(2)}</b> &middot; yaw <b>${(cmd.yaw).toFixed(2)}</b>`;
      }
      const mt = document.getElementById('ol-motion'); if (mt) mt.innerHTML = motionTxt;
      if (visible) raf = requestAnimationFrame(frame);
    }

    // ---- input: horizontal drag -> yaw, vertical drag -> forward ----
    function ptr(e) {
      const r = stage.getBoundingClientRect();
      const x = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      const y = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      tgt.yaw = clamp((x - W / 2) / (W / 2) * MAXY, -MAXY, MAXY);
      tgt.fwd = clamp(-(y - H / 2) / (H / 2) * MAXF, -MAXF, MAXF);
      tgt.lat = 0;
      lastP = { x, y };
    }
    stage.addEventListener('mousedown', e => { if (scene === 'drag') { dragging = true; ptr(e); } });
    window.addEventListener('mousemove', e => { if (dragging) ptr(e); });
    window.addEventListener('mouseup', () => { dragging = false; tgt.yaw = 0; tgt.fwd = 0; tgt.lat = 0; });
    stage.addEventListener('touchstart', e => { if (scene === 'drag') { dragging = true; ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchmove', e => { if (dragging) { ptr(e); e.preventDefault(); } }, { passive: false });
    stage.addEventListener('touchend', () => { dragging = false; tgt.yaw = 0; tgt.fwd = 0; tgt.lat = 0; });
    stage.tabIndex = 0;
    stage.setAttribute('role', 'application');
    stage.setAttribute('aria-label', 'Optic-flow sensor. Drag, or use arrow keys, to move and read out self-motion.');
    stage.addEventListener('keydown', e => {
      const v = { ArrowLeft: ['yaw', -MAXY], ArrowRight: ['yaw', MAXY], ArrowUp: ['fwd', MAXF], ArrowDown: ['fwd', -MAXF] }[e.key];
      if (!v) return; tgt[v[0]] = v[1]; e.preventDefault();
      clearTimeout(stage._kt); stage._kt = setTimeout(() => { tgt.yaw = 0; tgt.fwd = 0; }, 260);
    });

    document.getElementById('ol-scene').addEventListener('click', e => {
      const b = e.target.closest('button'); if (!b) return;
      e.currentTarget.querySelectorAll('button').forEach(x => x.classList.remove('on')); b.classList.add('on');
      scene = b.dataset.s;
    });

    if (window.Live) Live.onState(refreshLive);
    refreshLive();
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
