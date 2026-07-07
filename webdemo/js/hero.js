/* ============================================================
   hero.js - living connectome background
   ============================================================ */
(function () {
  const cv = document.getElementById('hero-canvas');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W, H, DPR, nodes = [], edges = [], pulses = [], raf, mouse = { x: -1e4, y: -1e4 };

  const PAL = ['#f4b13c', '#4b9bff', '#2fd4c6', '#9a8cf0'];

  function resize() {
    DPR = Math.min(2, window.devicePixelRatio || 1);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * DPR; cv.height = H * DPR;
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    build();
  }

  function build() {
    const density = Math.max(38, Math.min(96, Math.floor((W * H) / 16000)));
    nodes = [];
    for (let i = 0; i < density; i++) {
      nodes.push({
        x: Math.random() * W, y: Math.random() * H,
        vx: (Math.random() - .5) * 0.12, vy: (Math.random() - .5) * 0.12,
        r: 1.1 + Math.random() * 2.2,
        c: PAL[(Math.random() * PAL.length) | 0],
        ph: Math.random() * Math.PI * 2,
      });
    }
    computeEdges();
  }

  function computeEdges() {
    edges = [];
    const R = Math.min(W, H) * 0.18, R2 = R * R;
    for (let i = 0; i < nodes.length; i++) {
      let cnt = 0;
      for (let j = i + 1; j < nodes.length && cnt < 4; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const d2 = dx * dx + dy * dy;
        if (d2 < R2) { edges.push([i, j, Math.sqrt(d2)]); cnt++; }
      }
    }
  }

  function spawnPulse() {
    if (edges.length === 0 || pulses.length > 22) return;
    const e = edges[(Math.random() * edges.length) | 0];
    pulses.push({ e, t: 0, sp: 0.006 + Math.random() * 0.014, c: nodes[e[0]].c });
  }

  let frame = 0;
  function draw() {
    ctx.clearRect(0, 0, W, H);
    frame++;
    // move + edge recompute occasionally
    for (const n of nodes) {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
      // gentle mouse repel
      const dx = n.x - mouse.x, dy = n.y - mouse.y, d2 = dx * dx + dy * dy;
      if (d2 < 14000) { const f = (14000 - d2) / 14000 * 0.4; n.x += dx / Math.sqrt(d2 + 1) * f; n.y += dy / Math.sqrt(d2 + 1) * f; }
    }
    if (frame % 30 === 0) computeEdges();

    // edges
    const R = Math.min(W, H) * 0.18;
    ctx.lineWidth = 1;
    for (const [i, j, d] of edges) {
      const a = (1 - d / R) * 0.16;
      ctx.strokeStyle = `rgba(120,150,190,${a})`;
      ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke();
    }
    // pulses
    for (let k = pulses.length - 1; k >= 0; k--) {
      const p = pulses[k]; p.t += p.sp;
      if (p.t >= 1) { pulses.splice(k, 1); continue; }
      const [i, j] = p.e;
      const x = nodes[i].x + (nodes[j].x - nodes[i].x) * p.t;
      const y = nodes[i].y + (nodes[j].y - nodes[i].y) * p.t;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 6);
      g.addColorStop(0, p.c); g.addColorStop(1, 'transparent');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 6, 0, 7); ctx.fill();
    }
    // nodes
    for (const n of nodes) {
      const pulse = 0.6 + 0.4 * Math.sin(frame * 0.02 + n.ph);
      ctx.fillStyle = n.c;
      ctx.globalAlpha = 0.55 * pulse;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 7); ctx.fill();
      ctx.globalAlpha = 0.12 * pulse;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r * 3, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1;
    if (frame % 10 === 0) spawnPulse();
    raf = requestAnimationFrame(draw);
  }

  function staticDraw() { // reduced-motion: one frame
    ctx.clearRect(0, 0, W, H);
    for (const [i, j, d] of edges) {
      const R = Math.min(W, H) * 0.18;
      ctx.strokeStyle = `rgba(120,150,190,${(1 - d / R) * 0.14})`;
      ctx.beginPath(); ctx.moveTo(nodes[i].x, nodes[i].y); ctx.lineTo(nodes[j].x, nodes[j].y); ctx.stroke();
    }
    for (const n of nodes) { ctx.fillStyle = n.c; ctx.globalAlpha = .5; ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, 7); ctx.fill(); }
    ctx.globalAlpha = 1;
  }

  window.addEventListener('resize', resize);
  window.addEventListener('mousemove', e => { const r = cv.getBoundingClientRect(); mouse.x = e.clientX - r.left; mouse.y = e.clientY - r.top; });
  cv.addEventListener('mouseleave', () => { mouse.x = mouse.y = -1e4; });
  resize();
  if (reduce) staticDraw();
  else {
    // pause when hero off-screen
    const io = new IntersectionObserver(es => es.forEach(e => {
      if (e.isIntersecting) { if (!raf) raf = requestAnimationFrame(draw); }
      else { cancelAnimationFrame(raf); raf = null; }
    }), { threshold: 0 });
    io.observe(cv);
  }
})();
