"""
FastAPI WebSocket server for the Nassenger 8 digital twin.

Runs the P101 fabric roll simulation at 50 Hz in a background thread,
broadcasts full blackboard + agent internals to all WebSocket clients
every tick, and accepts control/fault/speed commands from the UI.
"""

import asyncio
import json
import math
import os
import time
import threading
from pathlib import Path
from typing import Set, Optional

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse, JSONResponse

# Simulation imports
from parts.s1.p101_fabric_roll import build_p101_pipeline, get_initial_state

# ─────────────────────────────────────────────
# Agent-layer metadata (for the UI)
# Maps agent_id → (layer_number, group_label)
# ─────────────────────────────────────────────
AGENT_META = {
    "P101_rho_eff":    {"layer": 1, "group": "A"},
    "P101_r_next":     {"layer": 2, "group": "B"},
    "P101_m_roll":     {"layer": 3, "group": "C"},
    "P101_sigma_res":  {"layer": 3, "group": "C"},
    "P101_j_roll":     {"layer": 4, "group": "D"},
    "P101_l_rem":      {"layer": 4, "group": "D"},
    "P101_depletion":  {"layer": 4, "group": "D"},
    "P101_n_layers":   {"layer": 4, "group": "D"},
    "P101_v_fabric":   {"layer": 4, "group": "D"},
}

# ─────────────────────────────────────────────
# Simulation state (shared between threads)
# ─────────────────────────────────────────────
class SimState:
    def __init__(self):
        self.pipeline = None
        self.state: dict = {}
        self.lock = threading.Lock()

        self.paused = False
        self.speed_multiplier = 1
        self.fault_active: Optional[str] = None
        self.tick = 0
        self.start_wall = time.monotonic()

        # Latest broadcast snapshot (written by sim thread, read by WS)
        self.latest_snapshot: Optional[dict] = None
        self._manual_omega: Optional[float] = None

    def build(self):
        self.pipeline = build_p101_pipeline()
        self.state = get_initial_state()
        # Explicit reset of all state-feedback vars to initial values
        self.state["omega_roll"]    = np.float64(0.0)
        self.state["r_current"]     = np.float64(0.300)
        self.state["v_fabric_prev"] = np.float64(0.0)
        self.tick = 0
        self.start_wall = time.monotonic()
        self.fault_active = None
        self.paused = False
        self.latest_snapshot = None
        self._manual_omega = None

    def reset(self):
        with self.lock:
            self.build()

    def _collect_agents(self) -> dict:
        """Gather get_ui_state() from every agent in the pipeline."""
        agents_out = {}
        for layer in self.pipeline.layers:
            for agent in layer.agents:
                meta = AGENT_META.get(agent.agent_id, {"layer": 0, "group": "?"})
                d = agent.get_ui_state()
                d["layer"] = meta["layer"]
                d["group"] = meta["group"]
                agents_out[agent.agent_id] = d
        return agents_out

    def step(self) -> dict:
        """Run one simulation tick and return the broadcast snapshot."""
        with self.lock:
            s = self.state

            # State feedback
            if self.tick > 0:
                s["r_current"]    = float(s.get("r_next",     s.get("r_current", 0.3)))
                s["v_fabric_prev"] = float(s.get("v_fabric",   0.0))

            # Motor spinup — _manual_omega overrides auto-ramp when set via /api/command
            base_omega = (self._manual_omega
                          if self._manual_omega is not None
                          else min(10.0 + self.tick * 0.05, 50.0))
            s["omega_roll"] = np.float64(base_omega)

            # Fault overrides
            if self.fault_active == "OVERSPEED":
                s["omega_roll"] = np.float64(500.0)
            elif self.fault_active == "UNDERSPEED":
                s["omega_roll"] = np.float64(0.001)
            elif self.fault_active == "BELOW_CORE":
                s["r_current"]  = np.float64(0.049)

            # Reset fired_this_tick on all agents before step
            for layer in self.pipeline.layers:
                for agent in layer.agents:
                    agent.fired_this_tick = False

            self.state = self.pipeline.step(s)
            self.tick += 1

            # Collect agent internals
            agents = self._collect_agents()

            # Build anomaly_map
            anomaly_map = {
                aid: round(a["validator"]["anomaly_flag"], 6)
                for aid, a in agents.items()
            }

            # Blackboard subset for UI
            bb_keys = [
                "r_next", "m_roll", "J_roll", "L_rem",
                "depletion_pct", "N_layers", "rho_eff",
                "sigma_res", "v_fabric", "dr_dt", "depletion_warning"
            ]
            blackboard = {
                k: round(float(self.state.get(k, 0.0)), 6)
                for k in bb_keys
            }
            blackboard["omega_roll"] = round(float(self.state.get("omega_roll", 0.0)), 6)
            blackboard["r_current"]  = round(float(self.state.get("r_current", 0.3)), 6)

            snapshot = {
                "type":             "tick",
                "tick":             self.tick,
                "t":                round(self.tick * 0.02, 6),
                "dt":               0.02,
                "speed_multiplier": self.speed_multiplier,
                "fault_active":     self.fault_active,
                "paused":           self.paused,
                "blackboard":       blackboard,
                "agents":           agents,
                "anomaly_map":      anomaly_map,
            }
            self.latest_snapshot = snapshot
            return snapshot


sim = SimState()

# ─────────────────────────────────────────────
# WebSocket connection manager
# ─────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.active.add(ws)

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self.active.discard(ws)

    async def broadcast(self, data: str):
        dead = set()
        async with self._lock:
            targets = set(self.active)
        for ws in targets:
            try:
                await ws.send_text(data)
            except Exception:
                dead.add(ws)
        if dead:
            async with self._lock:
                self.active -= dead


manager = ConnectionManager()

# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI(title="Nassenger 8 Digital Twin")


@app.on_event("startup")
async def startup():
    sim.build()
    asyncio.create_task(_sim_loop())


async def _sim_loop():
    """Background coroutine: runs the simulation at 50 Hz * speed_multiplier.
    When paused the loop sleeps without calling sim.step() — tick does not advance.
    """
    DT = 0.02  # base timestep seconds
    while True:
        if sim.paused:
            # Simulation is frozen — do not tick, do not broadcast
            await asyncio.sleep(DT)
            continue

        steps = sim.speed_multiplier
        snapshot = None
        for _ in range(steps):
            snapshot = sim.step()

        if snapshot and manager.active:
            await manager.broadcast(json.dumps(snapshot, default=_json_default))

        await asyncio.sleep(DT)


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent.parent / "ui" / "dashboard.html"
    return html_path.read_text(encoding="utf-8")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    # Send the current snapshot immediately so UI isn't blank
    if sim.latest_snapshot:
        await ws.send_text(json.dumps(sim.latest_snapshot, default=_json_default))
    try:
        while True:
            raw = await ws.receive_text()
            await _handle_command(json.loads(raw))
    except WebSocketDisconnect:
        await manager.disconnect(ws)
    except Exception:
        await manager.disconnect(ws)


async def _handle_command(msg: dict):
    t = msg.get("type")
    if t == "control":
        cmd = msg.get("command")
        if cmd == "pause":
            sim.paused = True
        elif cmd == "run":
            sim.paused = False
        elif cmd == "reset":
            sim.reset()
            # Broadcast reset confirmation so all clients clear their UI state
            reset_msg = json.dumps({"type": "reset"})
            await manager.broadcast(reset_msg)
    elif t == "fault":
        fault = msg.get("fault", "CLEAR")
        sim.fault_active = None if fault == "CLEAR" else fault
    elif t == "speed":
        m = int(msg.get("multiplier", 1))
        sim.speed_multiplier = max(1, min(10, m))


# ─────────────────────────────────────────────
# REST API
# ─────────────────────────────────────────────

@app.get("/api/state")
async def api_state():
    """Current simulation state snapshot for polling clients (JARVIS UI)."""
    snap = sim.latest_snapshot
    if snap is None:
        return JSONResponse({"error": "simulation not ready"}, status_code=503)

    bb          = snap.get("blackboard", {})
    anomaly_map = snap.get("anomaly_map", {})

    def _health(score: float) -> str:
        if score > 0.7:  return "FAULT"
        if score > 0.3:  return "WARNING"
        return "NOMINAL"

    scores        = list(anomaly_map.values())
    max_score     = max(scores) if scores else 0.0
    fabric_health = _health(max_score)

    anomalies = [
        {"agent": aid, "score": round(score, 4)}
        for aid, score in sorted(anomaly_map.items(), key=lambda x: -x[1])
        if score > 0.1
    ]

    return {
        "tick":         snap.get("tick", 0),
        "t":            snap.get("t", 0.0),
        "paused":       snap.get("paused", False),
        "fault_active": snap.get("fault_active", None),
        "omega_roll":   round(float(bb.get("omega_roll", 0.0)), 4),
        "health": {
            "fabric_roll": fabric_health,
            "overall":     fabric_health,
        },
        "key_values": {
            "v_fabric":      round(float(bb.get("v_fabric",      0.0)), 4),
            "r_next":        round(float(bb.get("r_next",        0.0)), 4),
            "depletion_pct": round(float(bb.get("depletion_pct", 0.0)), 4),
            "anomaly_score": round(float(max_score), 4),
        },
        "anomalies": anomalies,
    }


@app.post("/api/command")
async def api_command(request: Request):
    """Accept control/fault/speed commands via REST (JARVIS UI)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid JSON body"}, status_code=400)

    action = body.get("action", "")
    value  = body.get("value")

    if action == "run":
        sim.paused = False
        return {"ok": True, "message": "simulation running"}

    elif action == "pause":
        sim.paused = True
        return {"ok": True, "message": "simulation paused"}

    elif action == "reset":
        sim.reset()
        await manager.broadcast(json.dumps({"type": "reset"}))
        return {"ok": True, "message": "simulation reset"}

    elif action == "set_omega":
        if value is None:
            return JSONResponse(
                {"ok": False, "error": "value required for set_omega"}, status_code=400)
        v = float(value)
        sim._manual_omega = v
        return {"ok": True, "message": f"omega set to {v:.2f} rad/s"}

    elif action == "fault":
        name  = str(value).upper() if value is not None else "CLEAR"
        valid = {"OVERSPEED", "UNDERSPEED", "BELOW_CORE", "CLEAR"}
        if name not in valid:
            return JSONResponse(
                {"ok": False,
                 "error": f"unknown fault '{name}'. valid: {', '.join(sorted(valid))}"},
                status_code=400)
        sim.fault_active = None if name == "CLEAR" else name
        msg = "fault cleared" if name == "CLEAR" else f"fault {name} injected"
        return {"ok": True, "message": msg}

    elif action == "set_speed":
        if value is None:
            return JSONResponse(
                {"ok": False, "error": "value required for set_speed"}, status_code=400)
        sim.speed_multiplier = max(1, min(10, int(float(value))))
        return {"ok": True, "message": f"speed multiplier → {sim.speed_multiplier}×"}

    else:
        return JSONResponse(
            {"ok": False, "error": f"unknown action '{action}'"}, status_code=400)


@app.get("/jarvis", response_class=HTMLResponse)
async def serve_jarvis():
    html_path = Path(__file__).parent.parent / "ui" / "jarvis.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/3d", response_class=HTMLResponse)
async def serve_3d():
    html_path = Path(__file__).parent.parent / "ui" / "nassenger_3d.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/3dv2", response_class=HTMLResponse)
async def serve_3dv2():
    html_path = Path(__file__).parent.parent / "ui" / "nassenger_3d_v2.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/home", response_class=HTMLResponse)
async def serve_website():
    html_path = Path(__file__).parent.parent / "ui" / "faulter_website.html"
    return html_path.read_text(encoding="utf-8")
