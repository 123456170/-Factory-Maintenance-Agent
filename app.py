import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import time
import hashlib
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, field_validator, ValidationError
from PIL import Image, ImageDraw, ImageFont

try:
    import av
except Exception:
    av = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False


st.set_page_config(
    page_title="Factory Maintenance Agent",
    page_icon="🛠️",
    layout="wide",
)


APP_NAME = "Factory Maintenance Agent"
MAX_CSV_BYTES = 10_000_000
MAX_IMAGE_BYTES = 8_000_000


class Fault(BaseModel):
    code: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class MaintenancePlan(BaseModel):
    machine_id: str
    health_score: int = Field(ge=0, le=100)
    top_faults: List[Fault] = Field(default_factory=list)
    recommended_action: str
    urgency: str
    est_downtime_hours: float = Field(ge=0.0)
    parts_needed: List[str] = Field(default_factory=list)

    @field_validator("health_score", mode="before")
    def _validate_health(cls, v: Any) -> int:
        return max(0, min(100, int(float(v))))

    @field_validator("urgency", mode="before")
    def _validate_urgency(cls, v: Any) -> str:
        value = str(v).upper().strip()
        if value not in {"P1", "P2", "P3", "P4"}:
            raise ValueError("urgency must be one of P1, P2, P3, P4")
        return value

    @field_validator("est_downtime_hours", mode="before")
    def _validate_downtime(cls, v: Any) -> float:
        return max(0.0, float(v))


class SensorStats(BaseModel):
    ok: bool = True
    machine_id: str
    start: Optional[str] = None
    end: Optional[str] = None
    samples: int = 0
    vibration_rms: float = 0.0
    temperature_mean: float = 0.0
    temperature_max: float = 0.0
    sustained_vibration: float = 0.0
    sustained_temperature: float = 0.0
    max_zscore: float = 0.0
    isolated_spike: bool = False
    anomaly_flags: List[str] = Field(default_factory=list)
    health_signal: float = 1.0
    error: Optional[str] = None


class DefectBox(BaseModel):
    x: int
    y: int
    w: int
    h: int
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class DefectResult(BaseModel):
    ok: bool = True
    image_path: str = "memory"
    width: int = 0
    height: int = 0
    glare_score: float = 0.0
    boxes: List[DefectBox] = Field(default_factory=list)
    error: Optional[str] = None


class ManualSnippet(BaseModel):
    ok: bool = True
    fault_code: str
    title: str = ""
    steps: List[str] = Field(default_factory=list)
    safety_notes: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class SpareStock(BaseModel):
    ok: bool = True
    part_no: str
    stock: int = 0
    lead_time_days: int = 0
    location: str = ""
    error: Optional[str] = None


class TicketDraft(BaseModel):
    machine_id: str
    severity: str
    summary: str
    parts_needed: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    idempotency_key: str = ""


SCENARIOS = {
    "CNC spindle — rising vibration + worn belt": {
        "machine_id": "CNC-SPINDLE-07",
        "pattern": "rising_vibration",
        "base_vib": 2.1,
        "noise_vib": 0.25,
        "base_temp": 48.0,
        "noise_temp": 1.5,
        "fault_code": "VIB-04",
        "photo_label": "worn_belt",
        "parts": ["BLT-CNC-220"],
    },
    "Healthy pump — noisy sensor spike": {
        "machine_id": "PUMP-HP-22",
        "pattern": "isolated_spike",
        "base_vib": 1.8,
        "noise_vib": 0.22,
        "base_temp": 42.0,
        "noise_temp": 1.2,
        "fault_code": "SENSOR-NOISE",
        "photo_label": "healthy",
        "parts": [],
    },
    "Conveyor motor — overheating + loose guard": {
        "machine_id": "CONV-MOTOR-14",
        "pattern": "overheat",
        "base_vib": 2.6,
        "noise_vib": 0.30,
        "base_temp": 55.0,
        "noise_temp": 1.8,
        "fault_code": "GUARD-01",
        "photo_label": "loose_guard",
        "parts": ["GRD-CONV-114"],
    },
}


MANUALS = {
    "VIB-04": ManualSnippet(
        fault_code="VIB-04",
        title="Excessive spindle vibration / belt wear",
        steps=[
            "Isolate the spindle and apply lockout/tagout.",
            "Inspect drive belt for cracking, glazing, and tension loss.",
            "Replace belt with approved part number.",
            "Re-check alignment and run a vibration trend for 30 minutes.",
        ],
        safety_notes=["Do not inspect rotating equipment while energized."],
    ),
    "TMP-02": ManualSnippet(
        fault_code="TMP-02",
        title="Sustained motor/pump overheating",
        steps=[
            "Verify cooling fan operation and clean heat sink/fins.",
            "Check load current against nameplate rating.",
            "Calibrate temperature sensor if reading is suspect.",
            "Restart under monitoring and confirm thermal trend.",
        ],
        safety_notes=["Allow equipment to cool before contact inspection."],
    ),
    "GUARD-01": ManualSnippet(
        fault_code="GUARD-01",
        title="Loose or missing machine guard",
        steps=[
            "Stop conveyor and apply lockout/tagout.",
            "Inspect guard fasteners and protective shielding.",
            "Replace or refit guard using approved part.",
            "Perform safety sign-off before restart.",
        ],
        safety_notes=["Safety hazard. Do not restart until guard is secured."],
    ),
    "SENSOR-NOISE": ManualSnippet(
        fault_code="SENSOR-NOISE",
        title="Isolated sensor spike / transient noise",
        steps=[
            "Confirm spike is isolated and not repeated.",
            "Check sensor cabling and connector seating.",
            "Monitor for 24 hours before scheduling maintenance.",
        ],
        safety_notes=["No immediate machine action required for a single transient spike."],
    ),
}


PARTS = {
    "BLT-CNC-220": SpareStock(
        part_no="BLT-CNC-220",
        stock=4,
        lead_time_days=2,
        location="Bin A-14",
    ),
    "TMP-SENSOR-09": SpareStock(
        part_no="TMP-SENSOR-09",
        stock=0,
        lead_time_days=7,
        location="Bin C-03",
    ),
    "GRD-CONV-114": SpareStock(
        part_no="GRD-CONV-114",
        stock=2,
        lead_time_days=3,
        location="Rack G-09",
    ),
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_identifier(value: Any, default: str) -> str:
    raw = str(value or default).strip()
    cleaned = "".join(ch for ch in raw if ch.isalnum() or ch in "-_.")[:64]
    return cleaned or default


def short_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@st.cache_resource(show_spinner=False)
def get_runtime() -> Dict[str, str]:
    return {
        "gemini": "gemini-2.0-flash style offline deterministic mock",
        "detector": "YOLOv8-style offline color/edge mock",
        "graph": "LangGraph-inspired 7-node state machine",
        "sdk": "google-genai not required for offline demo",
    }


@st.cache_data(show_spinner=False)
def make_demo_dataframe(scenario_key: str) -> pd.DataFrame:
    cfg = SCENARIOS[scenario_key]
    seed = int(hashlib.sha256(scenario_key.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)

    n = 576
    end = pd.Timestamp("2026-10-29T12:00:00")
    timestamps = pd.date_range(end=end, periods=n, freq="5min")

    vibration = rng.normal(cfg["base_vib"], cfg["noise_vib"], n)
    temperature = rng.normal(cfg["base_temp"], cfg["noise_temp"], n)
    current = rng.normal(12.0, 0.6, n)

    if cfg["pattern"] == "rising_vibration":
        ramp = np.linspace(0.0, 1.0, n)
        vibration += 6.0 * np.clip((ramp - 0.62) / 0.38, 0.0, 1.0)
    elif cfg["pattern"] == "isolated_spike":
        idx = int(n * 0.57)
        vibration[idx] += 13.0
        temperature[idx] += 6.0
        current[idx] += 2.0
    elif cfg["pattern"] == "overheat":
        ramp = np.linspace(0.0, 1.0, n)
        temperature += 28.0 * np.clip((ramp - 0.45) / 0.55, 0.0, 1.0)
        vibration += 1.8 * np.clip((ramp - 0.55) / 0.45, 0.0, 1.0)

    vibration = np.maximum(0.05, vibration + rng.normal(0.0, 0.05, n))
    temperature = np.maximum(15.0, temperature)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "machine_id": cfg["machine_id"],
            "vibration_rms": vibration,
            "temperature_c": temperature,
            "current_a": current,
        }
    )


@st.cache_data(show_spinner=False)
def make_demo_photo_bytes(scenario_key: str) -> bytes:
    label = scenario_key.lower()
    img = Image.new("RGB", (640, 480), (28, 33, 40))
    d = ImageDraw.Draw(img)

    for x in range(0, 640, 32):
        d.line([(x, 0), (x, 480)], fill=(38, 44, 52), width=1)
    for y in range(0, 480, 32):
        d.line([(0, y), (640, y)], fill=(38, 44, 52), width=1)

    if "cnc" in label:
        d.ellipse([220, 110, 420, 210], fill=(92, 102, 112), outline=(185, 195, 205), width=4)
        d.polygon([(170, 260), (470, 260), (430, 340), (210, 340)], fill=(72, 78, 86), outline=(150, 160, 170))
        d.line(
            [(245, 300), (285, 286), (325, 308), (365, 288), (405, 302)],
            fill=(255, 0, 200),
            width=9,
        )
        d.text((18, 18), "CNC SPINDLE BELT VIEW", fill=(220, 220, 220))
    elif "pump" in label:
        d.rectangle([190, 140, 450, 320], fill=(88, 98, 108), outline=(170, 180, 190), width=3)
        d.ellipse([280, 185, 360, 265], fill=(45, 120, 185), outline=(200, 220, 240), width=3)
        d.rectangle([90, 220, 190, 260], fill=(80, 90, 100), outline=(150, 160, 170))
        d.rectangle([450, 220, 550, 260], fill=(80, 90, 100), outline=(150, 160, 170))
        d.ellipse([500, 80, 522, 102], fill=(0, 255, 80))
        d.text((18, 18), "PUMP HOUSING VIEW", fill=(220, 220, 220))
    else:
        d.rectangle([150, 180, 490, 305], fill=(70, 80, 90), outline=(160, 170, 180), width=3)
        d.rectangle([110, 135, 530, 170], fill=(105, 115, 125), outline=(180, 190, 200), width=2)
        d.rectangle([295, 135, 395, 170], fill=(255, 140, 0), outline=(255, 255, 255), width=2)
        d.text((18, 18), "CONVEYOR MOTOR GUARD", fill=(220, 220, 220))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def normalize_dataframe(df: pd.DataFrame, machine_id: str) -> Tuple[pd.DataFrame, List[str]]:
    warnings: List[str] = []
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    aliases = {
        "timestamp": ["timestamp", "time", "datetime", "ts"],
        "machine_id": ["machine_id", "asset_id", "equipment_id"],
        "vibration_rms": ["vibration_rms", "vibration", "rms", "acceleration_rms"],
        "temperature_c": ["temperature_c", "temperature", "temp", "motor_temp"],
        "current_a": ["current_a", "current", "amps", "amperage"],
    }

    for target, names in aliases.items():
        if target not in out.columns:
            for name in names:
                if name in out.columns:
                    out = out.rename(columns={name: target})
                    break

    if "timestamp" not in out.columns:
        out["timestamp"] = pd.date_range(end=datetime.now(), periods=len(out), freq="5min")
    else:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
        if out["timestamp"].isna().all():
            out["timestamp"] = pd.date_range(end=datetime.now(), periods=len(out), freq="5min")

    for col, default in [("vibration_rms", 0.0), ("temperature_c", 25.0), ("current_a", 0.0)]:
        if col not in out.columns:
            out[col] = default
            warnings.append(f"Missing column '{col}'; filled with default value.")
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["vibration_rms", "temperature_c"])
    if out.empty:
        raise ValueError("No valid numeric rows after cleaning.")

    out["machine_id"] = machine_id
    out = out.sort_values("timestamp").reset_index(drop=True)
    return out, warnings


def load_input_dataframe(
    scenario_key: str,
    uploaded_csv: Any,
    machine_id_input: str,
) -> Tuple[pd.DataFrame, List[str], str]:
    machine_id = sanitize_identifier(machine_id_input, SCENARIOS[scenario_key]["machine_id"])

    if uploaded_csv is not None:
        try:
            size = getattr(uploaded_csv, "size", None)
            if size is None:
                size = len(uploaded_csv.getvalue())
            if size > MAX_CSV_BYTES:
                raise ValueError(f"CSV exceeds {MAX_CSV_BYTES // 1_000_000} MB limit.")

            raw = pd.read_csv(uploaded_csv)
            df, warnings = normalize_dataframe(raw, machine_id)
            warnings.append("Using uploaded sensor CSV.")
            return df, warnings, machine_id
        except Exception as exc:
            return (
                make_demo_dataframe(scenario_key),
                [f"Uploaded CSV could not be used ({exc}). Falling back to built-in demo CSV."],
                machine_id,
            )

    return make_demo_dataframe(scenario_key), ["Loaded built-in demo sensor CSV."], machine_id


def load_photos(
    scenario_key: str,
    uploaded_photos: Any,
    include_demo_if_none: bool,
    warnings: List[str],
) -> List[Dict[str, Any]]:
    photos: List[Dict[str, Any]] = []

    if uploaded_photos:
        for up in uploaded_photos:
            try:
                data = up.getvalue()
                if len(data) > MAX_IMAGE_BYTES:
                    warnings.append(f"Skipped large photo '{up.name}'.")
                    continue
                im = Image.open(io.BytesIO(data)).convert("RGB")
                photos.append({"name": up.name, "bytes": data, "image": im})
            except Exception as exc:
                warnings.append(f"Skipped corrupt photo '{getattr(up, 'name', 'upload')}': {exc}")

    if not photos and include_demo_if_none:
        data = make_demo_photo_bytes(scenario_key)
        im = Image.open(io.BytesIO(data)).convert("RGB")
        photos.append({"name": f"demo_{scenario_key.split()[0].lower()}.jpg", "bytes": data, "image": im})

    return photos


def read_sensor_window(
    df: pd.DataFrame,
    machine_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    strictness: float = 0.65,
) -> SensorStats:
    try:
        if df is None or df.empty:
            raise ValueError("Empty sensor dataframe.")

        data = df.copy()

        if "machine_id" in data.columns:
            matched = data[data["machine_id"].astype(str) == machine_id]
            if not matched.empty:
                data = matched

        if "timestamp" in data.columns:
            if start is not None:
                start_ts = pd.to_datetime(start, errors="coerce")
                if pd.notna(start_ts):
                    filtered = data[data["timestamp"] >= start_ts]
                    if not filtered.empty:
                        data = filtered

            if end is not None:
                end_ts = pd.to_datetime(end, errors="coerce")
                if pd.notna(end_ts):
                    filtered = data[data["timestamp"] <= end_ts]
                    if not filtered.empty:
                        data = filtered

        if data.empty:
            data = df

        vibration = data["vibration_rms"].astype(float)
        temperature = data["temperature_c"].astype(float)
        n = len(data)

        threshold_v = 6.0 * (1.2 - 0.5 * strictness)
        threshold_t = 75.0 * (1.2 - 0.4 * strictness)
        window = max(5, n // 20)

        above_v = vibration > threshold_v
        sustained_v = float(above_v.rolling(window, min_periods=1).mean().max()) if n else 0.0

        above_t = temperature > threshold_t
        sustained_t = float(above_t.rolling(window, min_periods=1).mean().max()) if n else 0.0

        mean_v = float(vibration.mean()) if n else 0.0
        std_v = float(vibration.std()) if n else 0.0
        z = (vibration - mean_v) / (std_v + 1e-6)
        max_z = float(z.abs().max()) if n else 0.0

        isolated = bool(max_z > 4.0 and sustained_v < 0.05 and float(above_v.mean()) < 0.03)

        flags: List[str] = []
        if sustained_v > (0.08 + 0.15 * strictness):
            flags.append("vibration_sustained_high")
        if isolated:
            flags.append("isolated_spike")
        if sustained_t > (0.08 + 0.12 * strictness):
            flags.append("temperature_sustained_high")
        if n and float(temperature.max()) > threshold_t * 1.15 and sustained_t < 0.05:
            flags.append("temperature_spike")
        if std_v > 1.2 * mean_v + 0.5 and not flags:
            flags.append("sensor_noise")

        health_signal = 1.0 - min(
            0.9,
            0.55 * sustained_v + 0.45 * sustained_t + 0.08 * max(0.0, max_z - 3.0),
        )
        if isolated:
            health_signal = max(0.85, health_signal)

        return SensorStats(
            ok=True,
            machine_id=machine_id,
            start=str(data["timestamp"].min()) if "timestamp" in data.columns else None,
            end=str(data["timestamp"].max()) if "timestamp" in data.columns else None,
            samples=n,
            vibration_rms=float(np.sqrt(np.mean(np.square(vibration)))) if n else 0.0,
            temperature_mean=float(temperature.mean()) if n else 0.0,
            temperature_max=float(temperature.max()) if n else 0.0,
            sustained_vibration=sustained_v,
            sustained_temperature=sustained_t,
            max_zscore=max_z,
            isolated_spike=isolated,
            anomaly_flags=flags,
            health_signal=float(np.clip(health_signal, 0.0, 1.0)),
        )
    except Exception as exc:
        return SensorStats(
            ok=False,
            machine_id=machine_id,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_defect_detector(image_bytes: bytes, image_path: str = "memory") -> DefectResult:
    try:
        im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(im)
        h, w = arr.shape[:2]

        glare = float(np.mean(np.all(arr > 235, axis=2)))
        boxes: List[DefectBox] = []

        def add_mask(mask: np.ndarray, label: str, base_conf: float) -> None:
            if int(mask.sum()) < 60:
                return
            ys, xs = np.where(mask)
            x1 = int(xs.min())
            y1 = int(ys.min())
            x2 = int(xs.max())
            y2 = int(ys.max())
            if x2 - x1 < 8 or y2 - y1 < 8:
                return
            conf = min(0.97, base_conf + 0.05 * min(1.0, float(mask.sum()) / 5000.0))
            boxes.append(
                DefectBox(
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    label=label,
                    confidence=float(conf),
                )
            )

        magenta = (arr[:, :, 0] > 170) & (arr[:, :, 1] < 90) & (arr[:, :, 2] > 120)
        add_mask(magenta, "worn_belt", 0.84)

        orange = (arr[:, :, 0] > 200) & (arr[:, :, 1] > 90) & (arr[:, :, 1] < 180) & (arr[:, :, 2] < 80)
        add_mask(orange, "loose_guard", 0.88)

        return DefectResult(
            ok=True,
            image_path=image_path,
            width=w,
            height=h,
            glare_score=glare,
            boxes=boxes,
        )
    except Exception as exc:
        return DefectResult(
            ok=False,
            image_path=image_path,
            error=f"{type(exc).__name__}: {exc}",
        )


def lookup_manual(fault_code: str) -> ManualSnippet:
    code = str(fault_code).upper().strip()
    if code in MANUALS:
        return MANUALS[code]
    return ManualSnippet(ok=False, fault_code=code, error="unknown_fault_code")


def check_spare_parts(part_no: str) -> SpareStock:
    part = str(part_no).upper().strip()
    if part in PARTS:
        return PARTS[part]
    return SpareStock(ok=False, part_no=part, stock=0, lead_time_days=999, error="unknown_part_number")


def make_idempotency_key(
    machine_id: str,
    severity: str,
    summary: str,
    parts_needed: List[str],
    evidence_ids: List[str],
) -> str:
    payload = {
        "machine_id": machine_id,
        "severity": severity,
        "summary": summary,
        "parts_needed": sorted(parts_needed or []),
        "evidence_ids": sorted(evidence_ids or []),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:24]


def create_ticket(
    machine_id: str,
    severity: str,
    summary: str,
    parts_needed: List[str],
    evidence_ids: List[str],
    dry_run: bool = False,
    store: Optional[List[Dict[str, Any]]] = None,
    idempotency_key: Optional[str] = None,
) -> Tuple[Dict[str, Any], bool, str]:
    store = store if store is not None else []
    key = idempotency_key or make_idempotency_key(machine_id, severity, summary, parts_needed, evidence_ids)

    for existing in store:
        if existing.get("idempotency_key") == key:
            return dict(existing), True, key

    prefix = "DRY-" if dry_run else "CMMS-"
    ticket_id = prefix + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + key[:6]

    ticket = {
        "ticket_id": ticket_id,
        "machine_id": machine_id,
        "severity": severity,
        "summary": summary,
        "parts_needed": parts_needed,
        "evidence_ids": evidence_ids,
        "idempotency_key": key,
        "status": "dry_run" if dry_run else "open",
        "created_at": utc_now_iso(),
    }

    if not dry_run:
        store.append(ticket)

    return ticket, False, key


def summarize_result(result: Any) -> str:
    try:
        if isinstance(result, BaseModel):
            data = result.model_dump()
        else:
            data = result
        return json.dumps(data, default=str)[:500]
    except Exception:
        return str(result)[:500]


def log_tool_call(
    tool_calls: List[Dict[str, Any]],
    name: str,
    args: Dict[str, Any],
    result: Any,
    readonly: bool = True,
    side_effect: bool = False,
    timeout_ms: int = 5000,
    error_type: Optional[str] = None,
) -> bool:
    try:
        args_json = json.dumps(args, sort_keys=True, default=str)
    except Exception:
        args_json = str(args)

    signature = hashlib.sha256(args_json.encode("utf-8")).hexdigest()[:16]
    repeats = sum(1 for item in tool_calls if item.get("signature") == signature)

    tool_calls.append(
        {
            "timestamp": utc_now_iso(),
            "tool": name,
            "signature": signature,
            "arguments": args,
            "readonly": readonly,
            "side_effect": side_effect,
            "timeout_ms": timeout_ms,
            "error_type": error_type,
            "repeat_count": repeats + 1,
            "loop_detected": repeats >= 2,
            "result_summary": summarize_result(result),
        }
    )

    return repeats < 2


def log_event(
    events: List[Dict[str, Any]],
    run_id: str,
    node: str,
    message: str,
    status: str = "info",
    details: Optional[Dict[str, Any]] = None,
) -> None:
    events.append(
        {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "node": node,
            "message": message,
            "status": status,
            "details": details or {},
        }
    )


def infer_fault_codes(sensor: SensorStats, defects: List[DefectResult]) -> List[str]:
    codes: List[str] = []

    if sensor.ok:
        if "vibration_sustained_high" in sensor.anomaly_flags:
            codes.append("VIB-04")
        if "temperature_sustained_high" in sensor.anomaly_flags:
            codes.append("TMP-02")

    for dr in defects:
        if not dr.ok:
            continue
        for box in dr.boxes:
            if box.label in {"worn_belt", "surface_defect"}:
                codes.append("VIB-04")
            if box.label == "loose_guard":
                codes.append("GUARD-01")

    if not codes and sensor.ok and sensor.isolated_spike:
        codes.append("SENSOR-NOISE")

    return list(dict.fromkeys(codes))


def mock_gemini_plan(
    machine_id: str,
    sensor: SensorStats,
    defects: List[DefectResult],
    manuals: List[ManualSnippet],
    strictness: float,
) -> Tuple[MaintenancePlan, List[str], List[str]]:
    warnings: List[str] = []
    evidence: Dict[str, str] = {}

    if sensor.ok:
        evidence["EV-SENS-001"] = (
            f"vibration RMS {sensor.vibration_rms:.2f}, "
            f"sustained vibration {sensor.sustained_vibration:.2f}, "
            f"temperature mean {sensor.temperature_mean:.1f} C, "
            f"flags {', '.join(sensor.anomaly_flags) or 'none'}"
        )
    else:
        warnings.append(f"Sensor tool failed: {sensor.error}")

    vis_entries: List[Tuple[str, str, float]] = []
    for i, dr in enumerate(defects, start=1):
        if not dr.ok:
            warnings.append(f"Vision tool failed for {dr.image_path}: {dr.error}")
            continue

        glare_penalty = 1.0
        if dr.glare_score > 0.18:
            glare_penalty = 0.75
            warnings.append(
                f"High glare/oil-smear score {dr.glare_score:.2f} on {dr.image_path}; vision confidence reduced."
            )

        for j, box in enumerate(dr.boxes, start=1):
            eid = f"EV-VIS-{i:02d}-{j:02d}"
            adjusted_conf = round(min(0.98, box.confidence * glare_penalty), 2)
            evidence[eid] = (
                f"{box.label} confidence {adjusted_conf:.2f} "
                f"box x={box.x}, y={box.y}, w={box.w}, h={box.h}"
            )
            vis_entries.append((eid, box.label, adjusted_conf))

    man_entries: List[Tuple[str, str]] = []
    for man in manuals:
        if not man.ok:
            warnings.append(f"Manual lookup failed for {man.fault_code}: {man.error}")
            continue
        eid = f"EV-MAN-{man.fault_code}"
        first_step = man.steps[0] if man.steps else "Procedure available."
        evidence[eid] = f"{man.title}: {first_step}"
        man_entries.append((eid, man.fault_code))

    def ev_text(ids: List[str]) -> str:
        return "; ".join(f"{eid}: {evidence[eid]}" for eid in ids if eid in evidence)

    faults: List[Fault] = []
    safety = False

    sustained_v = sensor.sustained_vibration if sensor.ok else 0.0
    sustained_t = sensor.sustained_temperature if sensor.ok else 0.0

    vib_visual = [(eid, conf) for eid, label, conf in vis_entries if label in {"worn_belt", "surface_defect"}]
    guard_visual = [(eid, conf) for eid, label, conf in vis_entries if label == "loose_guard"]

    if (sensor.ok and "vibration_sustained_high" in sensor.anomaly_flags) or vib_visual:
        ids: List[str] = []
        if sensor.ok:
            ids.append("EV-SENS-001")
        ids.extend(eid for eid, _ in vib_visual)
        ids.extend(eid for eid, code in man_entries if code == "VIB-04")

        if not ids:
            warnings.append("Insufficient evidence for VIB-04.")
        else:
            base = 0.60 if vib_visual else 0.45
            max_visual_conf = max((conf for _, conf in vib_visual), default=0.0)
            conf = base + 0.25 * min(1.0, sustained_v) + 0.15 * max_visual_conf
            conf = round(min(0.97, max(0.0, conf)), 2)
            faults.append(Fault(code="VIB-04", evidence=ev_text(ids), confidence=conf))

    if sensor.ok and "temperature_sustained_high" in sensor.anomaly_flags:
        ids = ["EV-SENS-001"]
        ids.extend(eid for eid, code in man_entries if code == "TMP-02")
        conf = round(min(0.97, 0.58 + 0.30 * sustained_t), 2)
        faults.append(Fault(code="TMP-02", evidence=ev_text(ids), confidence=conf))

    if guard_visual:
        safety = True
        ids = [eid for eid, _ in guard_visual]
        if sensor.ok:
            ids.append("EV-SENS-001")
        ids.extend(eid for eid, code in man_entries if code == "GUARD-01")
        max_guard_conf = max(conf for _, conf in guard_visual)
        conf = 0.72 + 0.20 * max_guard_conf
        if sensor.ok and "temperature_sustained_high" in sensor.anomaly_flags:
            conf += 0.05
        faults.append(Fault(code="GUARD-01", evidence=ev_text(ids), confidence=round(min(0.98, conf), 2)))

    if not faults:
        if sensor.ok and sensor.isolated_spike:
            warnings.append("Only an isolated sensor spike was detected. No maintenance ticket created.")
        elif not sensor.ok:
            warnings.append("Sensor evidence unavailable. Falling back to conservative no-ticket recommendation.")

    parts_map = {
        "VIB-04": "BLT-CNC-220",
        "TMP-02": "TMP-SENSOR-09",
        "GUARD-01": "GRD-CONV-114",
    }

    parts_needed: List[str] = []
    for fault in faults:
        part = parts_map.get(fault.code)
        if part:
            parts_needed.append(part)
    parts_needed = list(dict.fromkeys(parts_needed))

    max_conf = max((fault.confidence for fault in faults), default=0.0)

    if safety or (sensor.ok and sensor.temperature_max > 92.0):
        urgency = "P1"
    elif any(fault.code in {"VIB-04", "TMP-02"} and fault.confidence >= 0.65 for fault in faults):
        urgency = "P2"
    elif faults:
        urgency = "P3"
    else:
        urgency = "P4"

    score = 100.0
    for fault in faults:
        penalty = 32.0 * fault.confidence
        if fault.code == "GUARD-01":
            penalty += 22.0
        if fault.code == "TMP-02":
            penalty += 12.0
        if fault.code == "VIB-04":
            penalty += 10.0
        score -= penalty

    if sensor.ok and sensor.isolated_spike and not faults:
        score -= 3.0

    health_score = int(max(0, min(100, round(score))))

    if urgency == "P1":
        recommended_action = "Immediate safety isolation and guard repair. Lockout/tagout required before restart."
    elif any(fault.code == "VIB-04" for fault in faults):
        recommended_action = "Schedule belt replacement and spindle alignment within 24 hours; re-check vibration after commissioning."
    elif any(fault.code == "TMP-02" for fault in faults):
        recommended_action = "Inspect cooling path, clean heat exchanger/fan, and verify temperature sensor calibration."
    elif faults:
        recommended_action = "Plan corrective maintenance at the next available shift."
    else:
        recommended_action = "No maintenance ticket required. Monitor sensor channel; recalibrate if isolated spikes repeat."

    base_downtime = 0.0
    if urgency == "P1":
        base_downtime = 6.0
    elif urgency == "P2":
        base_downtime = 3.0
    elif urgency == "P3":
        base_downtime = 1.0

    est_downtime_hours = round(base_downtime * (0.7 + 0.3 * max_conf), 1)

    plan = MaintenancePlan(
        machine_id=machine_id,
        health_score=health_score,
        top_faults=faults,
        recommended_action=recommended_action,
        urgency=urgency,
        est_downtime_hours=est_downtime_hours,
        parts_needed=parts_needed,
    )

    return plan, warnings, list(evidence.keys())


def degraded_plan(machine_id: str, sensor: SensorStats, defects: List[DefectResult]) -> MaintenancePlan:
    has_defect = any(dr.ok and dr.boxes for dr in defects)
    has_sensor_issue = bool(sensor.ok and sensor.anomaly_flags)
    urgency = "P3" if has_defect or has_sensor_issue else "P4"
    health = 68 if urgency == "P3" else 92
    downtime = 1.0 if urgency == "P3" else 0.0
    action = (
        "Degraded mode: deterministic fallback produced a conservative review recommendation. "
        "Human engineer should verify raw evidence before action."
    )
    return MaintenancePlan(
        machine_id=machine_id,
        health_score=health,
        top_faults=[],
        recommended_action=action,
        urgency=urgency,
        est_downtime_hours=downtime,
        parts_needed=[],
    )


def should_create_ticket(plan: MaintenancePlan, strictness: float) -> bool:
    if not plan.top_faults:
        return False
    if plan.urgency == "P4":
        return False
    max_conf = max(fault.confidence for fault in plan.top_faults)
    return max_conf >= max(0.55, strictness)


def estimate_metrics(df: Optional[pd.DataFrame], photos: List[Dict[str, Any]], plan: MaintenancePlan) -> Dict[str, Any]:
    rows = 0 if df is None else len(df)
    plan_chars = len(json.dumps(plan.model_dump(), default=str))
    input_tokens = int(rows * 2 + plan_chars / 4 + len(photos) * 700 + 1200)
    output_tokens = int(plan_chars / 3 + 450)
    total_tokens = input_tokens + output_tokens
    cost_usd = round((input_tokens * 0.075 + output_tokens * 0.30) / 1_000_000, 6)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def draw_panel(d: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], title: str) -> None:
    d.rectangle(box, outline=(70, 80, 95), width=1, fill=(18, 22, 28))
    d.text((box[0] + 6, box[1] + 4), title, fill=(170, 190, 210))


def draw_series(
    d: ImageDraw.ImageDraw,
    box: Tuple[int, int, int, int],
    values: List[float],
    current_idx: int,
    threshold: float,
) -> None:
    if not values:
        return

    x0, y0, x1, y1 = box
    current_idx = max(0, min(current_idx, len(values) - 1))
    visible = values[: current_idx + 1]

    vmin = min(min(visible), threshold * 0.75)
    vmax = max(max(visible), threshold * 1.25)
    if vmax <= vmin:
        vmax = vmin + 1.0

    y_threshold = y1 - ((threshold - vmin) / (vmax - vmin)) * (y1 - y0)
    y_threshold = max(y0, min(y1, y_threshold))

    d.rectangle([x0, y0, x1, y_threshold], fill=(62, 26, 32))
    d.line([x0, y1, x1, y1], fill=(85, 95, 108), width=1)
    d.line([x0, y0, x0, y1], fill=(85, 95, 108), width=1)
    d.line([x0, y_threshold, x1, y_threshold], fill=(255, 180, 80), width=1)

    denom = max(1, len(values) - 1)
    points: List[Tuple[float, float]] = []
    for i, value in enumerate(visible):
        x = x0 + (i / denom) * (x1 - x0)
        y = y1 - ((value - vmin) / (vmax - vmin)) * (y1 - y0)
        points.append((x, y))

    if len(points) > 1:
        d.line(points, fill=(80, 200, 255), width=2)
    if points:
        cx, cy = points[-1]
        d.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=(255, 90, 90))


def node_for_progress(progress: float) -> str:
    if progress < 0.12:
        return "Node 1/7 ingest"
    if progress < 0.28:
        return "Node 2/7 perceive"
    if progress < 0.48:
        return "Node 3/7 analyze"
    if progress < 0.64:
        return "Node 4/7 plan"
    if progress < 0.82:
        return "Node 5/7 act"
    if progress < 0.94:
        return "Node 6/7 human gate"
    return "Node 7/7 finalize"


def build_video_logs(result: Dict[str, Any]) -> List[Tuple[float, str]]:
    logs: List[Tuple[float, str]] = []
    logs.append((0.01, f"run_id={result.get('run_id', 'unknown')}"))
    logs.append((0.04, f"scenario={result.get('metrics', {}).get('scenario', 'unknown')}"))

    events = result.get("events", [])
    for i, ev in enumerate(events[:12]):
        progress = 0.08 + i * 0.065
        logs.append((min(progress, 0.90), str(ev.get("message", ""))[:82]))

    if result.get("pending_actions"):
        logs.append((0.80, "PENDING: human approval required before ticket write"))
    else:
        logs.append((0.80, "No side-effect ticket action required"))

    plan = result.get("plan", {})
    logs.append((0.95, f"plan urgency={plan.get('urgency', '?')} health={plan.get('health_score', '?')}"))
    return logs


def render_video_frame(
    idx: int,
    total: int,
    result: Dict[str, Any],
    series: List[float],
    threshold: float,
    photo: Optional[Image.Image],
    defect: Optional[DefectResult],
    logs: List[Tuple[float, str]],
) -> Image.Image:
    W, H = 640, 360
    img = Image.new("RGB", (W, H), (12, 15, 20))
    d = ImageDraw.Draw(img)

    progress = idx / max(1, total - 1)

    d.rectangle([0, 0, W, 30], fill=(24, 29, 36))
    d.text((8, 8), "Factory Maintenance Agent — Live Demo", fill=(240, 240, 240))
    d.text((W - 82, 8), f"{progress * 100:0.0f}%", fill=(160, 200, 255))

    d.rectangle([8, 34, W - 8, 52], fill=(30, 36, 44))
    d.text((12, 38), node_for_progress(progress), fill=(190, 210, 230))

    chart_box = (8, 58, 310, 190)
    draw_panel(d, chart_box, "Vibration RMS")
    draw_series(
        d,
        (chart_box[0] + 6, chart_box[1] + 18, chart_box[2] - 6, chart_box[3] - 6),
        series,
        int(progress * max(0, len(series) - 1)),
        threshold,
    )

    photo_box = (318, 58, W - 8, 190)
    draw_panel(d, photo_box, "Machine photo / defect overlay")
    if photo is not None:
        inner_w = photo_box[2] - photo_box[0] - 12
        inner_h = photo_box[3] - photo_box[1] - 24
        im = photo.copy()
        im.thumbnail((inner_w, inner_h))
        img.paste(im, (photo_box[0] + 6, photo_box[1] + 18))

        if progress > 0.25 and defect is not None and defect.ok and defect.boxes:
            dd = ImageDraw.Draw(img)
            sx = im.width / max(1, photo.width)
            sy = im.height / max(1, photo.height)
            ox = photo_box[0] + 6
            oy = photo_box[1] + 18
            for box in defect.boxes:
                x1 = ox + int(box.x * sx)
                y1 = oy + int(box.y * sy)
                x2 = ox + int((box.x + box.w) * sx)
                y2 = oy + int((box.y + box.h) * sy)
                dd.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=2)
                dd.text((x1 + 2, y1 + 2), f"{box.label} {box.confidence:.2f}", fill=(255, 220, 120))

    log_box = (8, 198, W - 8, H - 28)
    draw_panel(d, log_box, "Agent trace")
    yy = log_box[1] + 18
    for p, msg in logs:
        if progress >= p:
            d.text((log_box[0] + 8, yy), msg[:78], fill=(190, 220, 190))
            yy += 13
            if yy > log_box[3] - 10:
                break

    plan = result.get("plan", {})
    footer = (
        f"Health {plan.get('health_score', '?')} | "
        f"Urgency {plan.get('urgency', '?')} | "
        f"Pending {len(result.get('pending_actions', []))} | "
        f"Created {len(result.get('created_tickets', []))}"
    )
    d.rectangle([0, H - 24, W, H], fill=(24, 29, 36))
    d.text((8, H - 18), footer[:96], fill=(220, 220, 220))

    return img


def encode_frames(frames: List[Image.Image], fps: int) -> Tuple[bytes, str]:
    if not frames:
        return b"", "video/mp4"

    w, h = frames[0].size
    if w % 2 or h % 2:
        w2 = w - (w % 2)
        h2 = h - (h % 2)
        frames = [f.crop((0, 0, w2, h2)) for f in frames]
        w, h = w2, h2

    if av is not None:
        for codec in ("libx264", "mpeg4"):
            try:
                buf = io.BytesIO()
                container = av.open(buf, mode="w", format="mp4")
                stream = container.add_stream(codec, rate=fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"

                for frame_image in frames:
                    arr = np.ascontiguousarray(np.asarray(frame_image.convert("RGB"), dtype=np.uint8))
                    frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

                for packet in stream.encode():
                    container.mux(packet)

                container.close()
                data = buf.getvalue()
                if data:
                    return data, "video/mp4"
            except Exception:
                continue

    try:
        buf = io.BytesIO()
        frames[0].save(
            buf,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=int(1000 / max(1, fps)),
            loop=0,
            optimize=False,
        )
        return buf.getvalue(), "image/gif"
    except Exception:
        return b"", "video/mp4"


def generate_demo_video(
    result: Dict[str, Any],
    df: Optional[pd.DataFrame],
    photos: List[Dict[str, Any]],
    scenario_key: str,
) -> Tuple[bytes, str]:
    fps = 10
    frames_count = 60

    if df is not None and "vibration_rms" in df.columns:
        series = [float(x) for x in df["vibration_rms"].astype(float).tolist()]
    else:
        series = [1.0 + 0.01 * i for i in range(frames_count)]

    strictness = float(result.get("metrics", {}).get("strictness", 0.65))
    threshold = 6.0 * (1.2 - 0.5 * strictness)

    photo = photos[0]["image"] if photos else None
    defect = photos[0].get("defect") if photos else None
    logs = build_video_logs(result)

    frames = [
        render_video_frame(i, frames_count, result, series, threshold, photo, defect, logs)
        for i in range(frames_count)
    ]

    return encode_frames(frames, fps)


def make_trend_chart_image(df: Optional[pd.DataFrame], strictness: float) -> Image.Image:
    W, H = 900, 320
    img = Image.new("RGB", (W, H), (17, 20, 26))
    d = ImageDraw.Draw(img)
    d.text((10, 8), "Sensor trend with anomaly band", fill=(230, 230, 230))

    if df is None or df.empty or "vibration_rms" not in df.columns:
        d.text((10, 40), "No sensor series available.", fill=(220, 120, 120))
        return img

    values = [float(x) for x in df["vibration_rms"].astype(float).tolist()]
    threshold = 6.0 * (1.2 - 0.5 * strictness)
    box = (40, 40, W - 20, H - 40)
    draw_series(d, box, values, len(values) - 1, threshold)
    d.text((box[0], H - 28), f"threshold={threshold:.2f}", fill=(255, 180, 80))
    return img


def make_overlay_image(photo: Dict[str, Any]) -> Image.Image:
    im = photo["image"].copy()
    defect = photo.get("defect")
    if defect is None:
        return im

    d = ImageDraw.Draw(im)
    if defect.ok:
        for box in defect.boxes:
            d.rectangle(
                [box.x, box.y, box.x + box.w, box.y + box.h],
                outline=(255, 80, 80),
                width=3,
            )
            d.text((box.x + 4, box.y + 4), f"{box.label} {box.confidence:.2f}", fill=(255, 220, 120))
        if defect.glare_score > 0.15:
            d.text((10, 10), f"glare {defect.glare_score:.2f}", fill=(255, 220, 120))
    else:
        d.text((10, 10), f"vision error: {defect.error}", fill=(255, 120, 120))

    return im


def events_dataframe(events: List[Dict[str, Any]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame()
    df = pd.DataFrame(events)
    if "details" in df.columns:
        df["details"] = df["details"].apply(lambda x: json.dumps(x, default=str)[:220])
    return df


def tool_calls_dataframe(tool_calls: List[Dict[str, Any]]) -> pd.DataFrame:
    if not tool_calls:
        return pd.DataFrame()
    df = pd.DataFrame(tool_calls)
    for col in ["arguments", "result_summary"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: json.dumps(x, default=str)[:250] if not isinstance(x, str) else x[:250]
            )
    return df


def build_pdf_report(result: Dict[str, Any]) -> Optional[bytes]:
    if not REPORTLAB_OK:
        return None

    try:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("Factory Maintenance Agent Report", styles["Title"]))
        story.append(Spacer(1, 12))

        plan = result.get("plan", {})
        metrics = result.get("metrics", {})

        story.append(Paragraph(f"<b>Run ID:</b> {escape(str(result.get('run_id', '')))}", styles["Normal"]))
        story.append(Paragraph(f"<b>Status:</b> {escape(str(result.get('status', '')))}", styles["Normal"]))
        story.append(Paragraph(f"<b>Exit code:</b> {escape(str(result.get('exit_code', '')))}", styles["Normal"]))
        story.append(Paragraph(f"<b>Latency ms:</b> {escape(str(metrics.get('latency_ms', '')))}", styles["Normal"]))
        story.append(Paragraph(f"<b>Cost USD:</b> {escape(str(metrics.get('cost_usd', '')))}", styles["Normal"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Maintenance Plan", styles["Heading2"]))
        plan_rows = [
            ["Field", "Value"],
            ["machine_id", str(plan.get("machine_id", ""))],
            ["health_score", str(plan.get("health_score", ""))],
            ["urgency", str(plan.get("urgency", ""))],
            ["est_downtime_hours", str(plan.get("est_downtime_hours", ""))],
            ["recommended_action", str(plan.get("recommended_action", ""))],
            ["parts_needed", ", ".join(plan.get("parts_needed", []))],
        ]
        table = Table(plan_rows, colWidths=[1.5 * inch, 5.0 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))

        story.append(Paragraph("Top Faults", styles["Heading2"]))
        faults = plan.get("top_faults", [])
        if faults:
            for fault in faults:
                story.append(
                    Paragraph(
                        f"<b>{escape(str(fault.get('code', '')))}</b> "
                        f"confidence={escape(str(fault.get('confidence', '')))}<br/>"
                        f"{escape(str(fault.get('evidence', ''))[:500])}",
                        styles["Normal"],
                    )
                )
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No faults reported.", styles["Normal"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Warnings", styles["Heading2"]))
        warnings = result.get("warnings", [])
        if warnings:
            for warning in warnings[:20]:
                story.append(Paragraph(f"- {escape(str(warning))}", styles["Normal"]))
        else:
            story.append(Paragraph("None.", styles["Normal"]))

        story.append(Spacer(1, 12))
        story.append(Paragraph("Audit Trail", styles["Heading2"]))
        events = result.get("events", [])[:40]
        for ev in events:
            story.append(
                Paragraph(
                    f"{escape(str(ev.get('timestamp', '')))} | "
                    f"{escape(str(ev.get('node', '')))} | "
                    f"{escape(str(ev.get('status', '')))} | "
                    f"{escape(str(ev.get('message', ''))[:300])}",
                    styles["Normal"],
                )
            )

        doc.build(story)
        return buf.getvalue()
    except Exception:
        return None


def execute_pipeline(
    status: Any,
    scenario_key: str,
    uploaded_csv: Any,
    uploaded_photos: Any,
    include_demo_photo: bool,
    machine_id_input: str,
    strictness: float,
    dry_run: bool,
    api_key: str,
) -> None:
    t0 = time.perf_counter()
    run_id = (
        "RUN-"
        + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        + "-"
        + hashlib.sha256(f"{scenario_key}{time.time()}".encode("utf-8")).hexdigest()[:6]
    )

    events: List[Dict[str, Any]] = []
    tool_calls: List[Dict[str, Any]] = []
    warnings: List[str] = []
    node_timings: Dict[str, int] = {}

    machine_id = sanitize_identifier(machine_id_input, SCENARIOS[scenario_key]["machine_id"])
    st.session_state.dry_run_current = dry_run
    st.session_state.strictness_current = strictness

    status.update(label="Node 1/7 ingest: loading sensor CSV", state="running")
    tn = time.perf_counter()
    df, ingest_warnings, machine_id = load_input_dataframe(scenario_key, uploaded_csv, machine_id)
    warnings.extend(ingest_warnings)
    log_event(events, run_id, "ingest", f"Sensor window loaded rows={len(df)} machine={machine_id}", "ok")
    node_timings["ingest"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 2/7 perceive: machine photos + defect detector", state="running")
    tn = time.perf_counter()
    photos = load_photos(scenario_key, uploaded_photos, include_demo_photo, warnings)
    defect_results: List[DefectResult] = []

    for ph in photos:
        res = run_defect_detector(ph["bytes"], ph["name"])
        ph["defect"] = res
        defect_results.append(res)
        log_tool_call(
            tool_calls,
            "run_defect_detector",
            {"image_path": ph["name"]},
            res,
            readonly=True,
            side_effect=False,
            timeout_ms=8000,
            error_type=res.error,
        )
        status.write(f"Vision: {ph['name']} -> {len(res.boxes)} defect(s), glare {res.glare_score:.2f}")

    log_event(events, run_id, "perceive", f"Processed {len(photos)} photo(s)", "ok")
    node_timings["perceive"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 3/7 analyze: sensor window + evidence extraction", state="running")
    tn = time.perf_counter()
    sensor = read_sensor_window(df, machine_id, None, None, strictness)
    log_tool_call(
        tool_calls,
        "read_sensor_window",
        {"machine_id": machine_id, "start": None, "end": None, "strictness": strictness},
        sensor,
        readonly=True,
        side_effect=False,
        timeout_ms=3000,
        error_type=sensor.error,
    )
    status.write(
        f"Sensor: flags={sensor.anomaly_flags}, sustained_vibration={sensor.sustained_vibration:.2f}"
    )

    codes = infer_fault_codes(sensor, defect_results)
    manuals: List[ManualSnippet] = []
    for code in codes:
        man = lookup_manual(code)
        manuals.append(man)
        log_tool_call(
            tool_calls,
            "lookup_manual",
            {"fault_code": code},
            man,
            readonly=True,
            side_effect=False,
            timeout_ms=1000,
            error_type=man.error,
        )

    log_event(events, run_id, "analyze", f"Fault candidates={codes}", "ok")
    node_timings["analyze"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 4/7 plan: structured maintenance plan", state="running")
    tn = time.perf_counter()
    evidence_ids: List[str] = []
    try:
        plan, plan_warnings, evidence_ids = mock_gemini_plan(machine_id, sensor, defect_results, manuals, strictness)
        warnings.extend(plan_warnings)
    except ValidationError as exc:
        warnings.append(f"Schema validation failed: {exc}. Using degraded fallback.")
        plan = degraded_plan(machine_id, sensor, defect_results)
    except Exception as exc:
        warnings.append(f"Planner exception: {exc}. Using degraded fallback.")
        plan = degraded_plan(machine_id, sensor, defect_results)

    log_event(events, run_id, "plan", f"Plan urgency={plan.urgency} health={plan.health_score}", "ok")
    node_timings["plan"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 5/7 act: spare parts check + ticket draft", state="running")
    tn = time.perf_counter()
    verified_parts: List[str] = []
    for part in plan.parts_needed:
        stock = check_spare_parts(part)
        log_tool_call(
            tool_calls,
            "check_spare_parts",
            {"part_no": part},
            stock,
            readonly=True,
            side_effect=False,
            timeout_ms=2000,
            error_type=stock.error,
        )
        if stock.ok:
            verified_parts.append(part)
            if stock.stock <= 0:
                warnings.append(f"Part {part} out of stock; lead time {stock.lead_time_days} days.")
        else:
            warnings.append(f"Removed unknown/hallucinated part number: {part}")

    plan.parts_needed = verified_parts
    try:
        plan = MaintenancePlan(**plan.model_dump())
    except Exception:
        plan = degraded_plan(machine_id, sensor, defect_results)

    pending_actions: List[Dict[str, Any]] = []
    max_conf = max((fault.confidence for fault in plan.top_faults), default=0.0)

    if should_create_ticket(plan, strictness):
        fault_codes = ", ".join(fault.code for fault in plan.top_faults) or "REVIEW"
        summary = f"[{plan.urgency}] {fault_codes}: {plan.recommended_action}"[:240]
        key = make_idempotency_key(machine_id, plan.urgency, summary, plan.parts_needed, evidence_ids)
        draft = TicketDraft(
            machine_id=machine_id,
            severity=plan.urgency,
            summary=summary,
            parts_needed=plan.parts_needed,
            evidence_ids=evidence_ids,
            confidence=max_conf,
            idempotency_key=key,
        ).model_dump()
        pending_actions.append(draft)
        log_event(
            events,
            run_id,
            "human_gate",
            "Ticket draft created and held for human approval. No CMMS write performed.",
            "pending",
            {"idempotency_key": key},
        )
        status.write("Ticket draft pending human gate; no side effect executed yet.")
    else:
        log_event(events, run_id, "act", "No ticket draft: evidence insufficient or healthy condition.", "ok")
        status.write("No ticket draft required.")

    node_timings["act"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 6/7 human gate", state="running")
    tn = time.perf_counter()
    log_event(events, run_id, "human_gate", f"Pending actions={len(pending_actions)}", "info")
    node_timings["human_gate"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Node 7/7 finalize: metrics + trace", state="running")
    tn = time.perf_counter()
    latency_ms = int((time.perf_counter() - t0) * 1000)
    metrics = estimate_metrics(df, photos, plan)
    metrics.update(
        {
            "latency_ms": latency_ms,
            "node_timings_ms": node_timings,
            "scenario": scenario_key,
            "dry_run": dry_run,
            "strictness": strictness,
            "api_key_present": bool(api_key),
        }
    )

    exit_code = 2 if pending_actions else 0
    degraded = any("degraded" in str(w).lower() for w in warnings)

    result = {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "status": "needs_human" if pending_actions else "ok",
        "exit_code": exit_code,
        "degraded": degraded,
        "warnings": warnings,
        "plan": plan.model_dump(),
        "pending_actions": pending_actions,
        "created_tickets": [],
        "events": events,
        "tool_calls": tool_calls,
        "metrics": metrics,
    }

    node_timings["finalize"] = int((time.perf_counter() - tn) * 1000)

    status.update(label="Generating live video artifact", state="running")
    video_bytes, video_mime = generate_demo_video(result, df, photos, scenario_key)

    st.session_state.result = result
    st.session_state.df = df
    st.session_state.photos = photos
    st.session_state.video_bytes = video_bytes
    st.session_state.video_mime = video_mime
    st.session_state.ticket_store = st.session_state.get("ticket_store", [])

    status.update(label=f"Ready — exit code {exit_code}", state="complete")


def build_fallback_result(scenario_key: str, error_message: str) -> Dict[str, Any]:
    machine_id = SCENARIOS[scenario_key]["machine_id"]
    plan = MaintenancePlan(
        machine_id=machine_id,
        health_score=75,
        top_faults=[],
        recommended_action="Degraded mode: pipeline error occurred. Manual review required before any maintenance action.",
        urgency="P3",
        est_downtime_hours=1.0,
        parts_needed=[],
    )
    run_id = "RUN-FALLBACK-" + short_hash(error_message + scenario_key)
    return {
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "status": "degraded",
        "exit_code": 3,
        "degraded": True,
        "warnings": [f"Fallback active: {error_message}"],
        "plan": plan.model_dump(),
        "pending_actions": [],
        "created_tickets": [],
        "events": [
            {
                "timestamp": utc_now_iso(),
                "run_id": run_id,
                "node": "fallback",
                "message": "Pipeline failed; degraded result generated.",
                "status": "error",
                "details": {"error": error_message},
            }
        ],
        "tool_calls": [],
        "metrics": {
            "latency_ms": 0,
            "node_timings_ms": {},
            "scenario": scenario_key,
            "dry_run": False,
            "strictness": 0.65,
            "api_key_present": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        },
    }


def init_session_state() -> None:
    defaults = {
        "has_run": False,
        "result": None,
        "df": None,
        "photos": [],
        "video_bytes": None,
        "video_mime": "video/mp4",
        "ticket_store": [],
        "last_error": None,
        "dry_run_current": False,
        "strictness_current": 0.65,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def refresh_result_status() -> None:
    result = st.session_state.get("result")
    if not result:
        return
    pending = result.get("pending_actions", [])
    result["status"] = "needs_human" if pending else "ok"
    result["exit_code"] = 2 if pending else 0


def approve_ticket(key: str, summary: str, severity: str) -> None:
    result = st.session_state.get("result")
    if not result:
        return

    draft = None
    for item in result.get("pending_actions", []):
        if item.get("idempotency_key") == key:
            draft = item
            break

    if draft is None:
        return

    dry_run = bool(st.session_state.get("dry_run_current", False))
    store = st.session_state.setdefault("ticket_store", [])

    ticket, duplicate, _ = create_ticket(
        machine_id=draft["machine_id"],
        severity=severity,
        summary=summary,
        parts_needed=draft.get("parts_needed", []),
        evidence_ids=draft.get("evidence_ids", []),
        dry_run=dry_run,
        store=store,
        idempotency_key=key,
    )

    result.setdefault("created_tickets", []).append(ticket)
    result["pending_actions"] = [
        item for item in result.get("pending_actions", []) if item.get("idempotency_key") != key
    ]

    log_tool_call(
        result.setdefault("tool_calls", []),
        "create_ticket",
        {
            "machine_id": draft["machine_id"],
            "severity": severity,
            "summary": summary[:120],
            "parts_needed": draft.get("parts_needed", []),
            "dry_run": dry_run,
        },
        ticket,
        readonly=False,
        side_effect=not dry_run,
        timeout_ms=5000,
        error_type="duplicate_suppressed" if duplicate else None,
    )

    message = (
        f"Duplicate suppressed for idempotency key {key}; returned existing ticket {ticket['ticket_id']}."
        if duplicate
        else f"Human approved. Ticket {'dry-run ' if dry_run else ''}filed: {ticket['ticket_id']}."
    )
    result.setdefault("events", []).append(
        {
            "timestamp": utc_now_iso(),
            "run_id": result.get("run_id", ""),
            "node": "human_gate",
            "message": message,
            "status": "approved",
            "details": {"ticket_id": ticket["ticket_id"], "duplicate": duplicate, "dry_run": dry_run},
        }
    )

    for widget_key in [f"sum_{key}", f"sev_{key}", f"approve_{key}", f"reject_{key}"]:
        st.session_state.pop(widget_key, None)

    refresh_result_status()
    st.rerun()


def reject_ticket(key: str) -> None:
    result = st.session_state.get("result")
    if not result:
        return

    draft = None
    for item in result.get("pending_actions", []):
        if item.get("idempotency_key") == key:
            draft = item
            break

    if draft is None:
        return

    result["pending_actions"] = [
        item for item in result.get("pending_actions", []) if item.get("idempotency_key") != key
    ]

    result.setdefault("events", []).append(
        {
            "timestamp": utc_now_iso(),
            "run_id": result.get("run_id", ""),
            "node": "human_gate",
            "message": f"Human rejected ticket draft {key}. No CMMS write performed.",
            "status": "rejected",
            "details": {"idempotency_key": key},
        }
    )

    for widget_key in [f"sum_{key}", f"sev_{key}", f"approve_{key}", f"reject_{key}"]:
        st.session_state.pop(widget_key, None)

    refresh_result_status()
    st.rerun()


def render_ticket_queue() -> None:
    result = st.session_state.get("result")
    if not result:
        return

    pending = result.get("pending_actions", [])
    created = result.get("created_tickets", [])

    if not pending and not created:
        st.info("No ticket actions for this run.")
        return

    if pending:
        st.subheader("Human gate — pending approvals")
        st.caption("Nothing with side effects is written to the CMMS until a human approves.")
        for draft in pending:
            key = draft.get("idempotency_key", short_hash(draft))
            sum_key = f"sum_{key}"
            sev_key = f"sev_{key}"

            if sum_key not in st.session_state:
                st.session_state[sum_key] = draft.get("summary", "")
            if sev_key not in st.session_state:
                st.session_state[sev_key] = draft.get("severity", "P3")

            with st.container():
                st.markdown("---")
                st.markdown(
                    f"**{draft.get('machine_id', '')} • {st.session_state[sev_key]} • "
                    f"confidence {draft.get('confidence', 0.0):.2f}**"
                )
                st.caption("Evidence: " + ", ".join(draft.get("evidence_ids", [])[:6]))

                summary = st.text_area("Editable summary", key=sum_key, height=72)
                severity = st.selectbox("Severity", ["P1", "P2", "P3", "P4"], key=sev_key)

                c1, c2, c3 = st.columns([1, 1, 2])
                if c1.button("Approve & file", key=f"approve_{key}", type="primary"):
                    approve_ticket(key, summary, severity)
                if c2.button("Reject", key=f"reject_{key}"):
                    reject_ticket(key)
                c3.code(json.dumps(draft, indent=2, default=str), language="json")

    if created:
        st.subheader("Created / dry-run tickets")
        st.dataframe(pd.DataFrame(created), use_container_width=True, hide_index=True)


def render_live_run() -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("Waiting for first run...")
        return

    if st.session_state.get("last_error"):
        with st.expander("Pipeline error — degraded mode active"):
            st.code(st.session_state["last_error"])

    video_bytes = st.session_state.get("video_bytes")
    video_mime = st.session_state.get("video_mime", "video/mp4")

    if video_bytes:
        if video_mime.startswith("video"):
            st.video(video_bytes, format=video_mime, loop=True, autoplay=True, muted=True)
            file_name = f"{result.get('run_id', 'run')}_demo.mp4"
        else:
            st.image(video_bytes, use_container_width=True)
            file_name = f"{result.get('run_id', 'run')}_demo.gif"

        st.download_button(
            "Download live demo video",
            video_bytes,
            file_name=file_name,
            mime=video_mime,
        )
    else:
        st.info("Video artifact unavailable; tabular results are still shown below.")

    if result.get("warnings"):
        st.subheader("Warnings")
        for warning in result["warnings"][:8]:
            st.warning(warning)

    plan = result.get("plan", {})
    pending = result.get("pending_actions", [])
    created = result.get("created_tickets", [])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Open / pending tickets", len(pending) + len(created))
    c2.metric("Health score", f"{plan.get('health_score', '?')}/100")
    c3.metric("Predicted downtime", f"{plan.get('est_downtime_hours', '?')} h")
    c4.metric("Latency", f"{result.get('metrics', {}).get('latency_ms', '?')} ms")

    st.subheader("Sensor trend")
    df = st.session_state.get("df")
    strictness = float(result.get("metrics", {}).get("strictness", 0.65))
    st.image(make_trend_chart_image(df, strictness), use_container_width=True)

    photos = st.session_state.get("photos", [])
    if photos:
        st.subheader("Photo evidence")
        cols = st.columns(min(3, len(photos)))
        for col, ph in zip(cols, photos):
            with col:
                st.image(make_overlay_image(ph), caption=ph.get("name", "photo"), use_container_width=True)
                defect = ph.get("defect")
                if defect:
                    st.caption(f"glare {defect.glare_score:.2f}, boxes {len(defect.boxes)}")

    render_ticket_queue()

    st.subheader("Recent agent trace")
    st.dataframe(events_dataframe(result.get("events", [])[-12:]), use_container_width=True, hide_index=True)


def render_results() -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("Run the agent to see results.")
        return

    st.subheader("Prioritized maintenance plan")
    plan = result.get("plan", {})
    st.json(plan)

    try:
        MaintenancePlan(**plan)
        st.success("Output schema valid.")
    except Exception as exc:
        st.error(f"Output schema invalid: {exc}")

    st.subheader("Run metrics")
    metrics = result.get("metrics", {})
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True, hide_index=True)

    st.subheader("Tool calls")
    st.dataframe(tool_calls_dataframe(result.get("tool_calls", [])), use_container_width=True, hide_index=True)

    st.subheader("Downloads")
    d1, d2, d3, d4 = st.columns(4)

    json_bytes = json.dumps(result, indent=2, default=str).encode("utf-8")
    d1.download_button(
        "Download JSON trace",
        json_bytes,
        file_name=f"{result.get('run_id', 'run')}.json",
        mime="application/json",
    )

    events_csv = events_dataframe(result.get("events", [])).to_csv(index=False).encode("utf-8")
    d2.download_button(
        "Download audit CSV",
        events_csv,
        file_name=f"{result.get('run_id', 'run')}_events.csv",
        mime="text/csv",
    )

    pdf = build_pdf_report(result)
    if pdf:
        d3.download_button(
            "Download PDF report",
            pdf,
            file_name=f"{result.get('run_id', 'run')}.pdf",
            mime="application/pdf",
        )
    else:
        d3.info("PDF renderer unavailable in this environment.")

    video_bytes = st.session_state.get("video_bytes")
    video_mime = st.session_state.get("video_mime", "video/mp4")
    if video_bytes:
        ext = "mp4" if video_mime.startswith("video") else "gif"
        d4.download_button(
            "Download demo video",
            video_bytes,
            file_name=f"{result.get('run_id', 'run')}_demo.{ext}",
            mime=video_mime,
        )
    else:
        d4.info("No video artifact.")


def render_audit() -> None:
    result = st.session_state.get("result")
    if not result:
        st.info("Run the agent to populate the audit log.")
        return

    st.subheader("Audit log")
    df = events_dataframe(result.get("events", []))
    if df.empty:
        st.info("No events.")
        return

    filter_text = st.text_input("Filter events")
    status_options = sorted(set(df.get("status", pd.Series(dtype=str)).dropna().astype(str)))
    status_filter = st.multiselect("Status filter", options=status_options)

    filtered = df.copy()
    if filter_text:
        needle = filter_text.lower()
        filtered = filtered[
            filtered.apply(lambda row: needle in " ".join(str(v) for v in row.values).lower(), axis=1)
        ]
    if status_filter:
        filtered = filtered[filtered["status"].astype(str).isin(status_filter)]

    st.dataframe(filtered, use_container_width=True, hide_index=True)

    st.subheader("Tool-call ledger")
    st.dataframe(tool_calls_dataframe(result.get("tool_calls", [])), use_container_width=True, hide_index=True)


def render_about() -> None:
    runtime = get_runtime()

    st.subheader("What this demo does")
    st.markdown(
        """
This single-file Streamlit app is an offline, deterministic demonstration of a Factory Maintenance Agent.

It reads a sensor CSV and machine photos, runs a small perception/analysis pipeline, produces a schema-validated
maintenance plan, and holds any ticket-creating side effect behind a human gate.
"""
    )

    st.subheader("Architecture strip")
    st.code(
        """
CSV/photos -> ingest -> perceive -> analyze -> plan -> act -> human gate -> finalize
                |          |          |         |        |          |           |
                v          v          v         v        v          v           v
           sensor DF          defect boxes   rules    JSON plan  draft    approve/reject  trace/video
        """,
        language="text",
    )

    st.subheader("Runtime")
    st.json(runtime)

    st.subheader("Guardrails")
    st.markdown(
        """
- The agent can create maintenance tickets only after human approval.
- P1 safety findings always remain behind the human gate.
- The agent cannot change PLC setpoints; no such tool exists in this app.
- Every tool call is logged with inputs, outputs, timeout metadata, and side-effect flags.
- Unknown part numbers are blocked by the spare-parts catalog.
- Duplicate ticket creation is suppressed by idempotency keys.
- Isolated sensor spikes are treated as noise unless corroborated by sustained evidence.
"""
    )

    st.subheader("Honest limitations")
    st.markdown(
        """
- This demo uses an offline mock instead of live Gemini 2.0 Flash, YOLOv8, or LangGraph calls.
- The vision model is a deterministic color/edge heuristic suitable for bundled synthetic photos.
- The planner is rule-based and grounded in tool evidence; it is not a trained maintenance model.
- Real deployments require CMMS authentication, RBAC, secret management, and production evaluation.
"""
    )


init_session_state()

st.title("🛠️ Factory Maintenance Agent")
st.caption("Offline demo: sensor CSV + machine photos → prioritized maintenance plan with human-gated ticket creation.")

runtime = get_runtime()

with st.sidebar:
    st.header("⚙️ Controls")

    scenario_keys = list(SCENARIOS.keys())
    scenario = st.selectbox("Sample scenario", scenario_keys, index=0)

    machine_default = SCENARIOS[scenario]["machine_id"]
    machine_id = st.text_input("Machine ID", value=machine_default)

    uploaded_csv = st.file_uploader("Sensor CSV (optional)", type=["csv"])
    uploaded_photos = st.file_uploader(
        "Machine photos (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    include_demo_photo = st.checkbox("Use demo photo if none uploaded", value=True)

    strictness = st.slider("Strictness / ticket confidence threshold", 0.0, 1.0, 0.65, 0.05)
    dry_run = st.checkbox("Dry run (no real CMMS writes)", value=False)

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    env_key = os.getenv("GEMINI_API_KEY", "")
    default_key = env_key or secret_key
    api_key = st.text_input(
        "Gemini API key (optional; offline demo does not need it)",
        value=default_key,
        type="password",
    )

    st.session_state.dry_run_current = dry_run
    st.session_state.strictness_current = strictness

    col1, col2 = st.sidebar.columns(2)
    if col1.button("Run agent", type="primary"):
        st.session_state.has_run = False
        st.rerun()

    if col2.button("Reset"):
        for key in list(st.session_state.keys()):
            if (
                key
                in {
                    "has_run",
                    "result",
                    "df",
                    "photos",
                    "video_bytes",
                    "video_mime",
                    "ticket_store",
                    "last_error",
                }
                or str(key).startswith("sum_")
                or str(key).startswith("sev_")
                or str(key).startswith("approve_")
                or str(key).startswith("reject_")
            ):
                del st.session_state[key]
        st.rerun()

    st.sidebar.caption(f"Detector: {runtime['detector']}")
    st.sidebar.caption(f"Planner: {runtime['gemini']}")

tabs = st.tabs(["Live Run", "Results", "Audit Log", "About"])

with tabs[0]:
    if not st.session_state.has_run:
        try:
            with st.status("Auto-starting realistic demo...", expanded=True) as status:
                execute_pipeline(
                    status=status,
                    scenario_key=scenario,
                    uploaded_csv=uploaded_csv,
                    uploaded_photos=uploaded_photos,
                    include_demo_photo=include_demo_photo,
                    machine_id_input=machine_id,
                    strictness=strictness,
                    dry_run=dry_run,
                    api_key=api_key,
                )
        except Exception as exc:
            st.session_state.last_error = traceback.format_exc()
            try:
                fallback_result = build_fallback_result(scenario, str(exc))
                df = make_demo_dataframe(scenario)
                warnings: List[str] = []
                photos = load_photos(scenario, None, True, warnings)
                video_bytes, video_mime = generate_demo_video(fallback_result, df, photos, scenario)
                st.session_state.result = fallback_result
                st.session_state.df = df
                st.session_state.photos = photos
                st.session_state.video_bytes = video_bytes
                st.session_state.video_mime = video_mime
            except Exception:
                st.session_state.result = build_fallback_result(scenario, str(exc))
                st.session_state.video_bytes = None
                st.session_state.video_mime = "video/mp4"
        finally:
            st.session_state.has_run = True

    render_live_run()

with tabs[1]:
    render_results()

with tabs[2]:
    render_audit()

with tabs[3]:
    render_about()