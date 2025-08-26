"""
Vehicle / Prohelion (CAN0) decoders — engineering values.
Exports:
- decode_vehicle(ts, can_id, d) -> List[dict] | None
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import struct

# ---- Optional scaling for counts -> engineering units ----
APPLY_SCALING = False  # set True if your FW sends integer counts, not floats

SCALING = {
    "MC_BUS":        {"bus_current_A": 0.1,   "bus_voltage_V": 0.1},
    "MC_VELOCITY":   {"vehicle_velocity_mps": 0.01, "motor_velocity_rpm": 1.0},
    "MC_PHASE":      {"phase_c_current_arms": 0.1,  "phase_b_current_arms": 0.1},
    "MC_V_VECTOR":   {"Vd": 0.1,  "Vq": 0.1},
    "MC_I_VECTOR":   {"Id": 0.1,  "Iq": 0.1},
    "MC_BEMF_VECTOR":{"BEMFd": 0.1, "BEMFq": 0.1},
    "MC_TEMP1":      {"heatsink_temp_C": 0.1, "motor_temp_C": 0.1},
    "MC_TEMP2":      {"dsp_temp_C": 0.1},
    "MC_CUMULATIVE": {"dc_bus_Ah": 0.1, "odometer_m": 1.0},
    "MC_RAIL1":      {"rail_15V": 0.01},
    "MC_RAIL2":      {"rail_3v3": 0.01, "rail_1v9": 0.01},
    "DC_DRIVE":      {"shunt_current_cmd_pct": 100.0, "motor_velocity_rpm": 1.0},
    "DC_POWER":      {"bus_current_cmd_pct": 100.0},
    "BP_VMAX":       {"max_cell_voltage_V": 0.0001},   # mV -> V
    "BP_VMIN":       {"min_cell_voltage_V": 0.0001},
    "BP_TMAX":       {"max_temp_C": 0.1},
    "BP_ISH":        {"shunt_current_A": 0.1, "state_of_charge_pct": 0.5},
    "BP_PVSS":       {"pack_voltage_V": 0.1, "shunt_sum": 0.1},
}

def _scale(msg: str, field: str, value: float) -> float:
    if not APPLY_SCALING:
        return value
    return value * SCALING.get(msg, {}).get(field, 1.0)

# ---- helpers (local) ----
def u16_le(b0, b1): return (b0 | (b1 << 8)) & 0xFFFF
def s16_le(b0, b1):
    v = u16_le(b0, b1)
    return v - 0x10000 if v & 0x8000 else v
def u32_le(b0,b1,b2,b3): return (b0 | (b1<<8) | (b2<<16) | (b3<<24)) & 0xFFFFFFFF

def split_words_le(d: bytes):
    return (u16_le(d[0],d[1]), u16_le(d[2],d[3]), u16_le(d[4],d[5]), u16_le(d[6],d[7]))

def expand_bits_word(word: int, prefix: str) -> Dict[str, int]:
    return {f"{prefix}_bit{b}": 1 if (word >> b) & 1 else 0 for b in range(16)}

def floats_le(d: bytes) -> tuple[float, float]:
    """
    Prohelion frames carry two little-endian 32-bit floats.
    Return (hi32, lo32): the value documented first is stored in bits 63..32.
    """
    lo, hi = struct.unpack("<ff", d)
    return hi, lo

# ---------- ID bases & offsets ----------
MC_CAN_BASE1 = 0x400
MC_CAN_BASE2 = 0x420
DC_CAN_BASE  = 0x500
STW_CAN_BASE = 0x540
BP_CAN_BASE  = 0x580
MPPT_CAN_BASE = 0x600
MPPT_CAN_ONOFF = MPPT_CAN_BASE + 0x10

MC_OFFSETS = {
    0x01: "MC_LIMITS",
    0x02: "MC_BUS",
    0x03: "MC_VELOCITY",
    0x04: "MC_PHASE",
    0x05: "MC_V_VECTOR",
    0x06: "MC_I_VECTOR",
    0x07: "MC_BEMF_VECTOR",
    0x08: "MC_RAIL1",
    0x09: "MC_RAIL2",
    0x0B: "MC_TEMP1",
    0x0C: "MC_TEMP2",
    0x0E: "MC_CUMULATIVE",
    0x17: "MC_SLIPSPEED",
}
DC_OFFSETS = {0x01:"DC_DRIVE", 0x02:"DC_POWER", 0x03:"DC_RESET", 0x04:"DC_SWITCH"}
STW_OFFSETS = {0x01:"STW_SWITCH"}
BP_OFFSETS  = {0x01:"BP_VMAX",0x02:"BP_VMIN",0x03:"BP_TMAX",0x04:"BP_PCDONE",0x05:"BP_ISH",0x06:"BP_PVSS",0x07:"BP_RESET"}
STW_FLAGS   = {0:"HORN",1:"IND_L",2:"IND_R",3:"REGEN",4:"CRUISE"}

def decode_mc(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    base = MC_CAN_BASE1 if (MC_CAN_BASE1 <= can_id < MC_CAN_BASE1+0x20) else MC_CAN_BASE2
    offset = can_id - base
    name = MC_OFFSETS.get(offset, f"MC_UNKNOWN_{offset:02X}")
    row = dict(timestamp=ts, id_hex=hex(can_id), message=name, base_hex=hex(base), offset=offset)

    if name == "MC_BUS":                 # A, V
        f_hi, f_lo = floats_le(d)
        row.update(bus_current_A=_scale("MC_BUS", "bus_current_A", f_hi), 
                   bus_voltage_V=_scale("MC_BUS", "bus_voltage_V", f_lo))
    elif name == "MC_VELOCITY":         # m/s, rpm
        f_hi, f_lo = floats_le(d)
        row.update(vehicle_velocity_mps=_scale("MC_VELOCITY", "vehicle_velocity_mps", f_hi), 
                   motor_velocity_rpm=_scale("MC_VELOCITY", "motor_velocity_rpm", f_lo))
    elif name == "MC_PHASE":            # Arms, Arms
        f_hi, f_lo = floats_le(d)
        row.update(phase_c_current_arms=_scale("MC_PHASE", "phase_c_current_arms", f_hi), 
                   phase_b_current_arms=_scale("MC_PHASE", "phase_b_current_arms", f_lo))
    elif name == "MC_V_VECTOR":
        f_hi, f_lo = floats_le(d)
        row.update(Vd=_scale("MC_V_VECTOR", "Vd", f_hi), 
                   Vq=_scale("MC_V_VECTOR", "Vq", f_lo))
    elif name == "MC_I_VECTOR":
        f_hi, f_lo = floats_le(d)
        row.update(Id=_scale("MC_I_VECTOR", "Id", f_hi), 
                   Iq=_scale("MC_I_VECTOR", "Iq", f_lo))
    elif name == "MC_BEMF_VECTOR":
        f_hi, f_lo = floats_le(d)
        row.update(BEMFd=_scale("MC_BEMF_VECTOR", "BEMFd", f_hi), 
                   BEMFq=_scale("MC_BEMF_VECTOR", "BEMFq", f_lo))
    elif name == "MC_TEMP1":            # °C
        f_hi, f_lo = floats_le(d)
        row.update(heatsink_temp_C=_scale("MC_TEMP1", "heatsink_temp_C", f_hi), 
                   motor_temp_C=_scale("MC_TEMP1", "motor_temp_C", f_lo))
    elif name == "MC_TEMP2":            # °C (only one documented)
        f_hi, f_lo = floats_le(d)
        row.update(dsp_temp_C=f_lo)
    elif name == "MC_CUMULATIVE":       # Ah, m
        f_hi, f_lo = floats_le(d)
        row.update(dc_bus_Ah=_scale("MC_CUMULATIVE", "dc_bus_Ah", f_hi), 
                   odometer_m=_scale("MC_CUMULATIVE", "odometer_m", f_lo))
    elif name == "MC_RAIL1":
        f_hi, _ = floats_le(d)
        row.update(rail_15V=_scale("MC_RAIL1", "rail_15V", f_hi))
    elif name == "MC_RAIL2":
        f_hi, f_lo = floats_le(d)
        row.update(rail_3v3=_scale("MC_RAIL2", "rail_3v3", f_hi), 
                   rail_1v9=_scale("MC_RAIL2", "rail_1v9", f_lo))
    elif name == "MC_LIMITS":
        w0,w1,w2,w3 = split_words_le(d)
        row.update(expand_bits_word(w1, "mc_limit"))
    else:
        # keep raw words for unknowns
        w0,w1,w2,w3 = split_words_le(d)
        row.update(word0=w0, word1=w1, word2=w2, word3=w3)

    return [row]

def decode_dc(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    offset = can_id - DC_CAN_BASE
    name = DC_OFFSETS.get(offset, f"DC_UNKNOWN_{offset:02X}")
    row = dict(timestamp=ts, id_hex=hex(can_id), message=name, offset=offset)
    if name == "DC_DRIVE":
        f_hi, f_lo = floats_le(d)  # typically pct (0..1) and rpm — adjust if your FW differs
        row.update(shunt_current_cmd_pct=_scale("DC_DRIVE", "shunt_current_cmd_pct", f_hi*100.0), 
                   motor_velocity_rpm=_scale("DC_DRIVE", "motor_velocity_rpm", f_lo))
    elif name == "DC_POWER":
        f_hi, _ = floats_le(d)     # typically bus current command pct
        row.update(bus_current_cmd_pct=_scale("DC_POWER", "bus_current_cmd_pct", f_hi*100.0))
    elif name == "DC_SWITCH":
        w0,w1,w2,w3 = split_words_le(d)
        row.update(expand_bits_word(w0, "dc_switch_pos"))
        row.update(expand_bits_word(w1, "dc_switch_change"))
    else:
        w0,w1,w2,w3 = split_words_le(d)
        row.update(word0=w0, word1=w1, word2=w2, word3=w3)
    return [row]

def decode_stw(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    offset = can_id - STW_CAN_BASE
    name = STW_OFFSETS.get(offset, f"STW_UNKNOWN_{offset:02X}")
    row = dict(timestamp=ts, id_hex=hex(can_id), message=name)
    if name == "STW_SWITCH":
        w0,w1,w2,w3 = split_words_le(d)
        row.update(expand_bits_word(w0, "stw_pos"))
        row.update(expand_bits_word(w1, "stw_change"))
        # convenience flags
        for bit, label in {0:"horn",1:"ind_l",2:"ind_r",3:"regen",4:"cruise"}.items():
            row[f"stw_pos_{label}"] = 1 if (w0 >> bit) & 1 else 0
    else:
        w0,w1,w2,w3 = split_words_le(d)
        row.update(word0=w0, word1=w1, word2=w2, word3=w3)
    return [row]

def decode_bp(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    offset = can_id - BP_CAN_BASE
    name = BP_OFFSETS.get(offset, f"BP_UNKNOWN_{offset:02X}")
    row = dict(timestamp=ts, id_hex=hex(can_id), message=name, offset=offset)
    f_hi, f_lo = floats_le(d)
    if name == "BP_VMAX":
        row.update(max_cell_voltage_V=_scale("BP_VMAX", "max_cell_voltage_V", f_hi), 
                   max_cell_id=int(round(f_lo)))
    elif name == "BP_VMIN":
        row.update(min_cell_voltage_V=_scale("BP_VMIN", "min_cell_voltage_V", f_hi), 
                   min_cell_id=int(round(f_lo)))
    elif name == "BP_TMAX":
        row.update(max_temp_C=_scale("BP_TMAX", "max_temp_C", f_hi), 
                   max_temp_cell=int(round(f_lo)))
    elif name == "BP_ISH":
        row.update(shunt_current_A=_scale("BP_ISH", "shunt_current_A", f_hi), 
                   state_of_charge_pct=_scale("BP_ISH", "state_of_charge_pct", f_lo))
    elif name == "BP_PVSS":
        row.update(pack_voltage_V=_scale("BP_PVSS", "pack_voltage_V", f_lo), 
                   shunt_sum=_scale("BP_PVSS", "shunt_sum", f_hi))
    return [row]

def decode_mppt(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    sub_id = can_id - MPPT_CAN_BASE
    w0,w1,w2,w3 = split_words_le(d)
    row = dict(timestamp=ts, id_hex=hex(can_id), message="MPPT",
               sub_id=sub_id, word0=w0, word1=w1, word2=w2, word3=w3)
    if can_id == MPPT_CAN_ONOFF:
        row["mppt_onoff_cmd"] = d[0]
    return [row]

# Dynamic router for CAN0 ranges
def decode_vehicle(ts, can_id, d: bytes):
    if (MC_CAN_BASE1 <= can_id < MC_CAN_BASE1+0x20) or (MC_CAN_BASE2 <= can_id < MC_CAN_BASE2+0x20):
        return decode_mc(ts, can_id, d)
    if (DC_CAN_BASE <= can_id < DC_CAN_BASE+0x10):
        return decode_dc(ts, can_id, d)
    if (STW_CAN_BASE <= can_id < STW_CAN_BASE+0x10):
        return decode_stw(ts, can_id, d)
    if (BP_CAN_BASE <= can_id < BP_CAN_BASE+0x10):
        return decode_bp(ts, can_id, d)
    if (MPPT_CAN_BASE <= can_id < MPPT_CAN_BASE+0x20):
        return decode_mppt(ts, can_id, d)
    return None
