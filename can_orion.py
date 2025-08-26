"""
Orion BMS + Thermistor (CAN1) decoders — engineering values.
Exports:
- decode_orion(ts, can_id, d) -> List[dict] | None
- CAN1_MESSAGE_NAMES: set[str]  (used to heuristically split if 'channel' is absent)
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional

# ---- helpers (local) ----
def u16_le(b0, b1): return (b0 | (b1 << 8)) & 0xFFFF
def s16_le(b0, b1):
    v = u16_le(b0, b1)
    return v - 0x10000 if v & 0x8000 else v
def u16_be(b0, b1): return ((b0 << 8) | b1) & 0xFFFF
def u32_le(b0,b1,b2,b3): return (b0 | (b1<<8) | (b2<<16) | (b3<<24)) & 0xFFFFFFFF
def s8(v): return v - 256 if v >= 128 else v

def compute_checksum(broadcast_id: int, data: bytes) -> int:
    # Orion cell-broadcast checksum = LSB( ID + 8 + sum(bytes[0..6]) )
    return (broadcast_id + 8 + sum(data[:7])) & 0xFF

# ---------- CAN1 IDs ----------
OR_CAN_BASE = 0x6B0
OR_SVOLT = OR_CAN_BASE + 0x00
OR_PACK  = OR_CAN_BASE + 0x01
OR_STEMP = OR_CAN_BASE + 0x02
OR_FLAGS = OR_CAN_BASE + 0x03
OR_AVOLT = OR_CAN_BASE + 0x04
OR_ATEMP = OR_CAN_BASE + 0x05

THERM_GENERAL   = 0x1838F380
THERM_ADDRCLAIM = 0x18EEFF80

def decode_or_svolt(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    hi_v = u16_le(d[0], d[1]) * 1e-4
    hi_id = d[2]
    lo_v = u16_le(d[4], d[5]) * 1e-4
    lo_id = d[6]
    return [dict(timestamp=ts, id_hex=hex(can_id), message="OR_SVOLT",
                 high_cell_voltage_V=round(hi_v,4), high_cell_id=hi_id,
                 low_cell_voltage_V=round(lo_v,4), low_cell_id=lo_id)]

def decode_or_pack(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    pack_current_A = s16_le(d[0], d[1]) / 10.0
    pack_voltage_V = u16_le(d[2], d[3]) / 10.0
    soc_pct        = u16_le(d[4], d[5]) * 0.5
    num_cells      = u16_le(d[6], d[7])
    return [dict(timestamp=ts, id_hex=hex(can_id), message="OR_PACK",
                 pack_current_A_orion=pack_current_A,
                 pack_voltage_V_orion=pack_voltage_V,
                 state_of_charge_pct_orion=soc_pct,
                 num_cells=num_cells)]

def decode_or_stemp(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    hi_temp_C = s16_le(d[0], d[1]); hi_id = u16_le(d[2], d[3])
    lo_temp_C = s16_le(d[4], d[5]); lo_id = u16_le(d[6], d[7])
    return [dict(timestamp=ts, id_hex=hex(can_id), message="OR_STEMP",
                 high_temp_C_orion=hi_temp_C, high_temp_id=hi_id,
                 low_temp_C_orion=lo_temp_C, low_temp_id=lo_id)]

def decode_or_flags(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    bit_names = [
        "Low Cell Voltage","High Cell Voltage","Over Temp","Open Wiring",
        "Internal Comm","Charge Enable Relay","Discharge Relay","Blank"
    ]
    flags0 = d[0]
    fields = {f"flag0_{n.replace(' ','_').lower()}": bool(flags0 & (1<<i))
              for i,n in enumerate(bit_names)}
    fields["raw_flags_hex"] = d.hex()
    return [dict(timestamp=ts, id_hex=hex(can_id), message="OR_FLAGS", **fields)]

def decode_or_avolt(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    cell_id        = d[0]
    inst_voltage_V = u16_be(d[1], d[2]) * 1e-4
    shunting       = bool(d[3] & 0x80)
    res_raw        = ((d[3] & 0x7F) << 8) | d[4]
    internal_mOhm  = res_raw * 0.01
    ocv_V          = u16_be(d[5], d[6]) * 1e-4
    checksum       = d[7]
    checksum_ok    = (checksum == compute_checksum(can_id, d))
    return [dict(timestamp=ts, id_hex=hex(can_id), message="OR_AVOLT",
                 cell_id=cell_id,
                 inst_voltage_V=round(inst_voltage_V,4),
                 internal_resistance_mOhm=round(internal_mOhm,2),
                 shunting=shunting,
                 open_circuit_voltage_V=round(ocv_V,4),
                 checksum=checksum, checksum_ok=checksum_ok)]

def decode_or_atemp(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    rows = []
    for i in range(0,8,2):
        tid, temp = d[i], s8(d[i+1])
        if tid==0 and temp==0 and i>0:
            continue
        rows.append(dict(timestamp=ts, id_hex=hex(can_id), message="OR_ATEMP",
                         thermistor_local_id=tid, temp_C=temp))
    return rows

def decode_therm_general(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    return [dict(timestamp=ts, id_hex=hex(can_id), message="THERM_GENERAL",
                 thermistor_global_id=d[0], temp_C=s8(d[1]),
                 thermistor_local_id=(d[2] & 0x7F), local_fault=bool(d[2] & 0x80),
                 lowest_temp_C=s8(d[3]), highest_temp_C=s8(d[4]),
                 module_highest_id=d[5], module_lowest_id=d[6], reserved=d[7])]

def decode_therm_addrclaim(ts, can_id, d: bytes) -> List[Dict[str,Any]]:
    return [dict(timestamp=ts, id_hex=hex(can_id), message="THERM_ADDR_CLAIM",
                 unique_id_hex=d[0:3].hex(),
                 bms_target_addr=d[3], module_number_shifted=d[4],
                 const_5=d[5], const_6=d[6], const_7=d[7])]

# Exact-decoder map
_DECODERS_EXACT = {
    OR_SVOLT: decode_or_svolt,
    OR_PACK:  decode_or_pack,
    OR_STEMP: decode_or_stemp,
    OR_FLAGS: decode_or_flags,
    OR_AVOLT: decode_or_avolt,
    OR_ATEMP: decode_or_atemp,
    THERM_GENERAL: decode_therm_general,
    THERM_ADDRCLAIM: decode_therm_addrclaim,
}

# exported set of message names (used for splitting if channel is missing)
CAN1_MESSAGE_NAMES = {
    "OR_SVOLT","OR_PACK","OR_STEMP","OR_FLAGS",
    "OR_AVOLT","OR_ATEMP","THERM_GENERAL","THERM_ADDR_CLAIM"
}

def decode_orion(ts, can_id, d: bytes):
    dec = _DECODERS_EXACT.get(can_id)
    if dec:
        return dec(ts, can_id, d)
    return None
