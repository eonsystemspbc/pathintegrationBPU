#!/usr/bin/env python3
"""Real-time GPU inference server for the connectomics-for-AI demo sandboxes.

Loads the trained connectome + matched-random models and streams live inference to the
browser over a WebSocket. Ops:
  {op:'mqar', bindings:[[k,v],...], query:k}         -> real recall distribution (both models)
  {op:'cx_reset'} / {op:'cx_step', fwd, turn}        -> heading bump + home vector (both models)
  {op:'ol_reset'} / {op:'ol_step', hex:[...]}        -> ego-motion estimate (both models)
The mushroom-body op is stateless; CX/OL keep per-connection hidden state.
"""
import asyncio, base64, json, os, sys, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import websockets

ROOT = Path("/home/ec2-user/pathintegrationBPU")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
PORT = int(os.environ.get("INFER_PORT", "8765"))

def log(*a): print("[infer]", *a, flush=True)

MODELS = {}   # what loaded successfully

# ================= Mushroom body (MQAR) =================
try:
    for p in (ROOT / "scripts/mqar", ROOT / "scripts", ROOT):
        sys.path.insert(0, str(p))
    import run_mqar_associative_recall as mq
    mb = mq.mb
    MQ_VOCAB, MQ_ROLE = 32, mq.ROLE_DIMS
    _mq_base = mb.load_base_matrix(str(ROOT / "connectomes/flywire_mushroom_body/adjacency_unsigned.npz"), 0)
    _mq_args = SimpleNamespace(vocab_size=MQ_VOCAB, init_seed=0, recurrent_runtime="sparse",
                               state_clip=0.0, freeze_recurrent=False, num_pairs=8, num_queries=8, reversal_pairs=0)
    def _mq_load(name, ckpt):
        m = mq.build_model(_mq_base, name, _mq_args, seed=0, device=DEVICE)
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=False); m.eval(); return m
    MQ_CONN = _mq_load("hemibrain_seeded", ROOT / "outputs/mqar_demo_capture_conn/model_hemibrain_seeded_seed0.pt")
    MQ_RAND = _mq_load("random_sparse", ROOT / "outputs/mqar_demo_capture_rand/model_random_sparse_seed0.pt")
    MODELS["mqar"] = True
    log(f"MQAR loaded: vocab={MQ_VOCAB} N={int(MQ_CONN.N)}")
except Exception as e:
    log("MQAR load FAILED:", e); traceback.print_exc()

def mqar_infer(bindings, query_key):
    keys = [int(k) for k, v in bindings]; values = [int(v) for k, v in bindings]
    D = len(bindings); T = 2 * D + 1
    x = np.zeros((1, T, MQ_VOCAB + MQ_ROLE), np.float32); step = 0
    for k, v in zip(keys, values):
        x[0, step, k] = 1; x[0, step, MQ_VOCAB + 0] = 1; step += 1
        x[0, step, v] = 1; x[0, step, MQ_VOCAB + 1] = 1; step += 1
    x[0, 2 * D, int(query_key)] = 1; x[0, 2 * D, MQ_VOCAB + 2] = 1
    xt = torch.from_numpy(x).to(DEVICE)
    res = {}
    for name, m in (("connectome", MQ_CONN), ("random", MQ_RAND)):
        probs = torch.softmax(m(xt)[0, 2 * D], -1).cpu().numpy()
        res[name] = {"probs": [round(float(p), 4) for p in probs], "pred": int(probs.argmax())}
    truth = values[keys.index(int(query_key))] if int(query_key) in keys else -1
    return {"truth": truth, **res}

# ================= Central complex (path integration) =================
# Frozen-reservoir checkpoints store W_rec dense; trainable ones store it as sparse
# COO (indices + values). Support both and always step through a coalesced sparse tensor.
def _sparse_stepper(ckpt, N):
    sd = torch.load(ckpt, map_location=DEVICE)
    if "W_rec_indices" in sd:
        W_rec = torch.sparse_coo_tensor(sd["W_rec_indices"], sd["W_rec_values"], (N, N))
    else:
        W_rec = sd["W_rec"].to_sparse()
    W_rec = W_rec.coalesce().to(DEVICE)
    return {"W_rec": W_rec, "W_in": sd["W_in"].to(DEVICE), "b_in": sd["b_in"].to(DEVICE),
            "W_out": sd["W_out"].to(DEVICE), "b_out": sd["b_out"].to(DEVICE),
            "sensory": sd["sensory_indices"].to(DEVICE), "output": sd["output_indices"].to(DEVICE)}
CX_DIR = "outputs/cx_demo_models_frozen"   # frozen reservoir: advantage is the wiring itself
try:
    _cx_meta = json.load(open(ROOT / CX_DIR / "meta.json"))
    CX_K, CX_N, CX_BINS = _cx_meta["K"], _cx_meta["N"], _cx_meta["heading_bins"]
    CX_HDS, CX_CLIP = _cx_meta["home_distance_scale"], _cx_meta["state_clip"]
    CX_CONN = _sparse_stepper(ROOT / CX_DIR / "model_cx_bpu_seed0.pt", CX_N)
    CX_RAND = _sparse_stepper(ROOT / CX_DIR / "model_random_seed0.pt", CX_N)
    MODELS["cx"] = True
    log(f"CX loaded: N={CX_N} K={CX_K} bins={CX_BINS}")
except Exception as e:
    log("CX load FAILED:", e)

def cx_step(mdl, fwd, turn, h):
    if h is None: h = torch.zeros(1, CX_N, device=DEVICE)
    x = torch.tensor([[float(fwd), float(turn)]], device=DEVICE)
    injection = x @ mdl["W_in"].t() + mdl["b_in"]
    for micro in range(CX_K):
        next_h = torch.sparse.mm(mdl["W_rec"], h.t()).t()
        if micro == 0:
            next_h = next_h.index_add(1, mdl["sensory"], injection)
        h = torch.relu(next_h)
        if CX_CLIP > 0: h = torch.clamp(h, max=CX_CLIP)
    out = (h.index_select(1, mdl["output"]) @ mdl["W_out"].t() + mdl["b_out"])[0]
    return out, h

def cx_out(out):
    o = out.cpu().numpy()
    return {"bump": [round(float(v), 4) for v in o[:CX_BINS]],
            "home": [float(o[CX_BINS]), float(o[CX_BINS + 1]), float(o[CX_BINS + 2]) * CX_HDS]}

# ================= Optic lobe (optic flow) =================
# Live ego-motion estimation from real optic flow. Each step renders a fresh
# training-distribution "glimpse" (a 16-frame hex-lattice sequence of the current
# constant ego-motion over a fixed panorama) and runs both models on it. The frozen
# recurrent net reads out [yaw_rate, forward, lateral] just as in the benchmark.
try:
    import run_optic_flow_benchmark as of  # noqa
    _ol_meta = json.load(open(ROOT / "outputs/ol_demo_models/meta.json"))
    OL_SPEC = of.OpticFlowSpec(**{k: v for k, v in _ol_meta["spec"].items() if k in of.OpticFlowSpec.__dataclass_fields__})
    OL_N = _ol_meta["N"]; OL_IN = _ol_meta["input_dim"]; OL_OUT = _ol_meta["output_dim"]; OL_CLIP = _ol_meta["state_clip"]
    OL_T = OL_SPEC.timesteps
    OL_BASE = of.lattice_angles(OL_SPEC)          # [IN, 2] azimuth/elevation per ommatidium
    def _ol_load(ckpt):
        sd = torch.load(ckpt, map_location=DEVICE)
        return {"W_rec": torch.sparse_coo_tensor(sd["edge_indices"], sd["W_rec_values"], (OL_N, OL_N)).coalesce().to(DEVICE),
                "W_in": sd["W_in"].to(DEVICE), "b_rec": sd["b_rec"].to(DEVICE),
                "ro_w": sd["readout.weight"].to(DEVICE), "ro_b": sd["readout.bias"].to(DEVICE)}
    OL_CONN = _ol_load(ROOT / "outputs/ol_demo_models/model_optic_lobe_seeded_seed0.pt")
    OL_RAND = _ol_load(ROOT / "outputs/ol_demo_models/model_random_sparse_seed0.pt")
    MODELS["ol"] = True
    log(f"OL loaded: N={OL_N} in={OL_IN} out={OL_OUT} T={OL_T}")
except Exception as e:
    log("OL load FAILED:", e); traceback.print_exc()

def ol_new_pano(seed):
    return of.make_panorama(np.random.default_rng(int(seed) & 0x7fffffff), OL_SPEC)

def ol_glimpse(pano, yaw_rate, forward, lateral, rng):
    seq = np.zeros((OL_T, OL_IN), np.float32)
    for t in range(OL_T):
        frac = t / max(OL_T - 1, 1)
        jitter = 1.0 + OL_SPEC.temporal_contrast_jitter * np.sin(2.0 * np.pi * frac)
        frame = of.render_hex_frame(pano, OL_BASE, yaw=yaw_rate * t,
                                    forward=forward * t, lateral=lateral * t, spec=OL_SPEC, rng=rng)
        frame = 0.5 + jitter * (frame - 0.5)
        if OL_SPEC.sensor_noise_std > 0:
            frame = frame + rng.normal(0.0, OL_SPEC.sensor_noise_std, size=frame.shape).astype(np.float32)
        seq[t] = np.clip(frame, 0.0, 1.0)
    return seq

def ol_run(mdl, seq):
    h = torch.zeros(1, OL_N, device=DEVICE)
    xt = torch.from_numpy(seq).to(DEVICE)
    outs = []
    for t in range(seq.shape[0]):
        rec = torch.sparse.mm(mdl["W_rec"], h.t()).t() + mdl["b_rec"] + xt[t:t + 1] @ mdl["W_in"].t()
        h = torch.relu(rec)
        if OL_CLIP > 0:
            h = torch.clamp(h, max=OL_CLIP)
        outs.append((h @ mdl["ro_w"].t() + mdl["ro_b"])[0])
    return torch.stack(outs).cpu().numpy()   # [T, 3]

# ================= WebSocket handler =================
async def handler(ws):
    st = {"cx_conn": None, "cx_rand": None, "ol_pano": None, "ol_rng": None}
    log("client connected")
    try:
        async for msg in ws:
            req = json.loads(msg); op = req.get("op"); rid = req.get("id")
            if op == "ping":
                await ws.send(json.dumps({"op": "pong", "id": rid, "models": list(MODELS.keys())}))
            elif op == "mqar" and "mqar" in MODELS:
                await ws.send(json.dumps({"op": "mqar", "id": rid, **mqar_infer(req["bindings"], req["query"])}))
            elif op == "cx_reset":
                st["cx_conn"] = st["cx_rand"] = None
                await ws.send(json.dumps({"op": "cx_reset", "id": rid}))
            elif op == "cx_step" and "cx" in MODELS:
                oc, st["cx_conn"] = cx_step(CX_CONN, req["fwd"], req["turn"], st["cx_conn"])
                orr, st["cx_rand"] = cx_step(CX_RAND, req["fwd"], req["turn"], st["cx_rand"])
                await ws.send(json.dumps({"op": "cx_step", "id": rid, "conn": cx_out(oc), "rand": cx_out(orr)}))
            elif op == "ol_reset":
                pano = ol_new_pano(req.get("seed", 0))
                st["ol_pano"] = pano
                st["ol_rng"] = np.random.default_rng((int(req.get("seed", 0)) + 1) & 0x7fffffff)
                pano_u8 = (np.clip(pano, 0.0, 1.0) * 255).astype(np.uint8)   # [H, W] row-major
                await ws.send(json.dumps({"op": "ol_reset", "id": rid,
                                          "lattice": [[round(float(a), 4), round(float(e), 4)] for a, e in OL_BASE],
                                          "pano": base64.b64encode(pano_u8.tobytes()).decode(),
                                          "pw": int(pano.shape[1]), "ph": int(pano.shape[0]),
                                          "fov_az": OL_SPEC.fov_azimuth_deg, "fov_el": OL_SPEC.fov_elevation_deg,
                                          "motion_scale": OL_SPEC.motion_scale}))
            elif op == "ol_step" and "ol" in MODELS:
                if st["ol_pano"] is None:
                    st["ol_pano"] = ol_new_pano(0); st["ol_rng"] = np.random.default_rng(1)
                yaw, fwd, lat = float(req["yaw"]), float(req["fwd"]), float(req["lat"])
                seq = ol_glimpse(st["ol_pano"], yaw, fwd, lat, st["ol_rng"])
                oc, orr = ol_run(OL_CONN, seq), ol_run(OL_RAND, seq)   # [T, 3] each
                tgt = np.array([yaw, fwd, lat], np.float32)
                c_err = float(np.sqrt(np.mean((oc - tgt) ** 2)))       # per-glimpse tracking RMSE
                r_err = float(np.sqrt(np.mean((orr - tgt) ** 2)))
                await ws.send(json.dumps({"op": "ol_step", "id": rid,
                                          "conn": [round(float(v), 4) for v in oc[-1]],   # final estimate (arrows)
                                          "rand": [round(float(v), 4) for v in orr[-1]],
                                          "conn_err": round(c_err, 4), "rand_err": round(r_err, 4),
                                          "truth": [yaw, fwd, lat],
                                          "hex": [round(float(v), 3) for v in seq[-1]]}))
            else:
                await ws.send(json.dumps({"op": "error", "id": rid, "msg": f"unknown/unavailable op {op}"}))
    except websockets.ConnectionClosed:
        pass
    except Exception as e:
        log("handler error:", e); traceback.print_exc()
    finally:
        log("client disconnected")

async def main():
    log(f"starting on 127.0.0.1:{PORT}  models={list(MODELS.keys())}  device={DEVICE}")
    async with websockets.serve(handler, "127.0.0.1", PORT, max_size=2**20, ping_interval=20):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
