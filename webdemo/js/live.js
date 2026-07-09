/* ============================================================
   live.js - shared WebSocket client for real-time GPU inference
   Talks to the /infer endpoint (Caddy -> infer_server.py on the box).
   Every op is request/response, correlated by an incrementing id.
   Exposes window.Live:
     Live.call({op, ...})  -> Promise(response)   (rejects if offline/timeout)
     Live.has('mqar'|'cx'|'ol')                   (model available server-side)
     Live.isOpen()                                 (socket connected)
     Live.onState(fn)                              (fn(state, models) on change)
   Demos use Live when present and fall back to their synthetic path otherwise,
   so the page still works from file:// or when the server is down.
   ============================================================ */
(function () {
  const WS_URL = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/infer';
  const REQ_TIMEOUT = 8000;
  let ws = null, connected = false, nextId = 1;
  const pending = new Map();
  const models = new Set();
  const listeners = new Set();
  let backoff = 500;

  function emit(state) { listeners.forEach(f => { try { f(state, models); } catch (e) {} }); }

  function connect() {
    let sock;
    try { sock = new WebSocket(WS_URL); } catch (e) { schedule(); return; }
    ws = sock;
    sock.onopen = () => {
      connected = true; backoff = 500;
      send({ op: 'ping' }).then(r => {
        (r.models || []).forEach(m => models.add(m));
        emit('open');
      }).catch(() => emit('open'));
    };
    sock.onmessage = ev => {
      let m; try { m = JSON.parse(ev.data); } catch (e) { return; }
      const p = m.id && pending.get(m.id);
      if (p) { pending.delete(m.id); p.resolve(m); }
    };
    sock.onclose = () => { connected = false; failAll(); emit('closed'); schedule(); };
    sock.onerror = () => { try { sock.close(); } catch (e) {} };
  }

  function failAll() { pending.forEach(p => p.reject(new Error('socket closed'))); pending.clear(); }
  function schedule() { setTimeout(connect, backoff); backoff = Math.min(8000, backoff * 1.7); }

  function send(obj) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== 1) { reject(new Error('not connected')); return; }
      const id = nextId++;
      pending.set(id, { resolve, reject });
      try { ws.send(JSON.stringify(Object.assign({}, obj, { id }))); }
      catch (e) { pending.delete(id); reject(e); return; }
      setTimeout(() => { if (pending.delete(id)) reject(new Error('timeout')); }, REQ_TIMEOUT);
    });
  }

  connect();

  window.Live = {
    call: send,
    has: m => models.has(m),
    isOpen: () => connected,
    onState: fn => { listeners.add(fn); return () => listeners.delete(fn); },
  };
})();
