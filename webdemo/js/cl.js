/* ============================================================
   cl.js - Continual learning (energy minimization) schematics.
   Stylized conceptual line-ink figures illustrating the finding
   that an energy-minimized transformer develops connectome-like
   structure and resists catastrophic forgetting across a curriculum.
   Colours use the shared C palette (dark instrument panels).
   ============================================================ */
(function () {
  const OPEN = 'rgba(143,160,200,0.34)';   // faint ink for open marks / grid on the dark panel
  function svg(host, w, h) { host.innerHTML = ''; const s = el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%' }); host.appendChild(s); return s; }

  /* 1 - descend an energy landscape -> modular ("connectome-like") organization */
  function fig1() {
    const host = document.getElementById('cl-fig-1'); if (!host) return;
    const s = svg(host, 300, 150);
    // energy landscape (a well) on the left
    s.appendChild(el('path', { d: 'M14,44 C48,44 56,104 90,104 C124,104 132,44 166,44', fill: 'none', stroke: C.mute, 'stroke-width': 1.6, opacity: .7 }));
    // ball descending into the minimum, with a dashed trail
    s.appendChild(el('path', { d: 'M58,30 C68,54 82,92 90,100', fill: 'none', stroke: C.bio, 'stroke-width': 1.3, 'stroke-dasharray': '3 3', opacity: .85 }));
    s.appendChild(el('circle', { cx: 90, cy: 101, r: 5, fill: C.bio }));
    s.appendChild(el('text', { x: 90, y: 133, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'minimize energy' }));
    s.appendChild(el('text', { x: 179, y: 76, 'text-anchor': 'middle', fill: C.mute, 'font-size': 16, text: '→' }));
    // resulting modular graph (two tight clusters, one bridge) on the right
    const A = [[214, 46], [236, 38], [230, 64]], B = [[268, 92], [286, 74], [266, 104]];
    [[A[0], A[1]], [A[1], A[2]], [A[0], A[2]], [B[0], B[1]], [B[1], B[2]], [B[0], B[2]], [A[2], B[0]]]
      .forEach(([p, q]) => s.appendChild(el('line', { x1: p[0], y1: p[1], x2: q[0], y2: q[1], stroke: C.bio, 'stroke-width': 1.3, opacity: .75 })));
    [...A, ...B].forEach(p => s.appendChild(el('circle', { cx: p[0], cy: p[1], r: 4, fill: C.bio })));
    s.appendChild(el('text', { x: 250, y: 133, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'modular structure' }));
  }

  /* 2 - connectivity matrix: block-modular (energy-min) vs unstructured (normal) */
  function fig2() {
    const host = document.getElementById('cl-fig-2'); if (!host) return;
    const s = svg(host, 300, 150);
    const N = 8, cell = 12, top = 20;
    function grid(ox, filled, col, label) {
      for (let r = 0; r < N; r++) for (let c = 0; c < N; c++) {
        const on = filled(r, c);
        s.appendChild(el('rect', { x: ox + c * cell, y: top + r * cell, width: cell - 2, height: cell - 2, rx: 1.5,
          fill: on ? col : 'none', stroke: on ? 'none' : OPEN, 'stroke-width': 1, opacity: on ? .92 : 1 }));
      }
      s.appendChild(el('text', { x: ox + N * cell / 2 - 1, y: top + N * cell + 16, 'text-anchor': 'middle', fill: col, 'font-size': 10, text: label }));
    }
    const blocks = [[0, 3], [3, 6], [6, 8]];
    const modular = (r, c) => blocks.some(([a, b]) => r >= a && r < b && c >= a && c < b) && (r + c) % 3 !== 0;
    const scatter = (r, c) => ((r * 7 + c * 13 + (r * c) % 5) % 4) === 0;   // deterministic "unstructured" fill
    grid(24, modular, C.bio, 'energy-minimized');
    grid(196, scatter, C.ctrl, 'normal transformer');
  }

  /* 3 - errors per 100 problems: ~1 (energy-min) vs ~10 (normal) */
  function fig3() {
    const host = document.getElementById('cl-fig-3'); if (!host) return;
    const s = svg(host, 300, 150);
    const sp = 9.4, top = 40;
    function panel(ox, nerr, col, label) {
      s.appendChild(el('text', { x: ox + 42, y: 22, 'text-anchor': 'middle', fill: col, 'font-size': 15, 'font-weight': 700, text: label }));
      const errSet = new Set(); const step = Math.floor(100 / Math.max(nerr, 1));
      for (let e = 0; e < nerr; e++) errSet.add((e * step + 4) % 100);
      let idx = 0;
      for (let i = 0; i < 10; i++) for (let j = 0; j < 10; j++) {
        const x = ox + j * sp, y = top + i * sp, err = errSet.has(idx); idx++;
        if (err) s.appendChild(el('circle', { cx: x, cy: y, r: 2.6, fill: col }));
        else s.appendChild(el('circle', { cx: x, cy: y, r: 2, fill: 'none', stroke: OPEN, 'stroke-width': 1 }));
      }
    }
    panel(24, 1, C.bio, '≈1 error');
    panel(190, 10, C.ctrl, '≈10 errors');
    s.appendChild(el('text', { x: 150, y: 147, 'text-anchor': 'middle', fill: C.mute, 'font-size': 10, text: 'per 100 problems · end of curriculum' }));
  }

  document.addEventListener('DOMContentLoaded', () => { fig1(); fig2(); fig3(); });
})();
