"""
Core streaming decoder + file splitting and Excel-safe CSV writer.

Exports:
- decode_csv_one(...)
- decode_csv_split(...)
- EXCEL_XLSX_MAX_ROWS / EXCEL_XLS_MAX_ROWS
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple, Callable
import pandas as pd
import os

import can_orion
import can_vehicle

# ---------- Excel/Calc row caps ----------
EXCEL_XLSX_MAX_ROWS = 1_048_576
EXCEL_XLS_MAX_ROWS = 65_536

# ---------- streaming parameters ----------
CHUNK_ROWS = 300_000      # rows to read per chunk
FLUSH_EVERY = 20_000      # rows to buffer before writing to disk

# ---------- helpers ----------
def parse_can_id(val) -> int:
    s = str(val).strip()
    if s.lower().startswith("0x"):
        return int(s, 16)
    return int(s, 16)

def expand_bits_bytes(data: bytes, prefix: str="b") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for i, b in enumerate(data):
        out[f"{prefix}{i}"] = b
        out[f"{prefix}{i}_hex"] = f"{b:02X}"
        for bit in range(8):
            out[f"{prefix}{i}_bit{bit}"] = 1 if ((b >> bit) & 1) else 0
    return out

def known_columns(include_bits: bool) -> list:
    """
    A unified schema so headers are consistent across rolled files.
    This list includes both vehicle (CAN0) and Orion (CAN1) fields.
    """
    base = [
        "timestamp","id_hex","message","channel","raw_data",
        "base_hex","offset","sub_id",
        # ---- Vehicle / Prohelion (CAN0) engineering values ----
        "bus_current_A","bus_voltage_V",
        "vehicle_velocity_mps","motor_velocity_rpm",
        "phase_c_current_arms","phase_b_current_arms",
        "Vd","Vq","Id","Iq","BEMFd","BEMFq",
        "heatsink_temp_C","motor_temp_C","dsp_temp_C",
        "dc_bus_Ah","odometer_m","rail_15V","rail_3v3","rail_1v9",
        *[f"mc_limit_bit{i}" for i in range(16)],
        *[f"dc_switch_pos_bit{i}" for i in range(16)],
        *[f"dc_switch_change_bit{i}" for i in range(16)],
        *[f"stw_pos_bit{i}" for i in range(16)],
        *[f"stw_change_bit{i}" for i in range(16)],
        "stw_pos_horn","stw_pos_ind_l","stw_pos_ind_r","stw_pos_regen","stw_pos_cruise",
        "max_cell_voltage_V","max_cell_id",
        "min_cell_voltage_V","min_cell_id",
        "max_temp_C","max_temp_cell",
        "shunt_current_A","state_of_charge_pct",
        "pack_voltage_V","shunt_sum",
        "mppt_onoff_cmd",
        # ---- Orion (CAN1) ----
        "high_cell_voltage_V","high_cell_id","low_cell_voltage_V","low_cell_id",
        "pack_current_A_orion","pack_voltage_V_orion","state_of_charge_pct_orion","num_cells",
        "high_temp_C_orion","high_temp_id","low_temp_C_orion","low_temp_id",
        "flag0_low_cell_voltage","flag0_high_cell_voltage","flag0_over_temp","flag0_open_wiring",
        "flag0_internal_comm","flag0_charge_enable_relay","flag0_discharge_relay","flag0_blank",
        "raw_flags_hex",
        "cell_id","inst_voltage_V","internal_resistance_mOhm","shunting",
        "open_circuit_voltage_V","checksum","checksum_ok",
        "thermistor_local_id","temp_C","thermistor_global_id","local_fault","lowest_temp_C",
        "highest_temp_C","module_highest_id","module_lowest_id",
        "unique_id_hex","bms_target_addr","module_number_shifted","const_5","const_6","const_7",
    ]
    if include_bits:
        for i in range(8):
            base += [f"b{i}", f"b{i}_hex", *[f"b{i}_bit{b}" for b in range(8)]]
    return base

class PartitionedCSVWriter:
    """Writes DataFrames to CSV, rolling files so each stays <= row_limit rows."""
    def __init__(self, out_path: str, row_limit: int, columns: list):
        self.base, self.ext = os.path.splitext(out_path)
        self.row_limit = int(row_limit)
        self.columns = columns
        self.part = 1
        self.count_in_part = 0
        self.paths: List[str] = []
        self.header_written = False

    def _path(self) -> str:
        if self.part == 1:
            return f"{self.base}{self.ext}"
        return f"{self.base}_part{self.part}{self.ext}"

    def _roll(self):
        self.part += 1
        self.count_in_part = 0
        self.header_written = False

    def write_df(self, df: pd.DataFrame):
        if df.empty:
            return
        df = df.reindex(columns=self.columns)
        start = 0
        n = len(df)
        while start < n:
            room = self.row_limit - self.count_in_part
            if room <= 0:
                self._roll()
            take = min(room, n - start)
            chunk = df.iloc[start:start+take]
            path = self._path()
            if path not in self.paths:
                self.paths.append(path)
            mode = "a" if self.header_written else "w"
            chunk.to_csv(path, index=False, header=not self.header_written, mode=mode)
            self.header_written = True
            self.count_in_part += len(chunk)
            start += take

# ---------- core streaming engine ----------
def _stream_decode(
    input_csv: str,
    include_bits: bool,
    channel: Optional[str],
    split: bool,
    out_one: Optional[str],
    out_can0: Optional[str],
    out_can1: Optional[str],
    row_limit: int,
    progress_cb: Optional[Callable[[float, Optional[str]], None]] = None,
) -> List[str]:
    """
    Memory-safe streaming decode. Returns list of output file paths.
    """
    outputs: List[str] = []
    cols = known_columns(include_bits)

    # columns to actually read
    need = {"id","data","timestamp","channel"}
    usecols = lambda c: c.strip().lower() in need

    # ---- PASS 1: count rows we will process (after optional channel filter)
    total_rows = 0
    for chunk in pd.read_csv(
        input_csv, chunksize=CHUNK_ROWS, dtype=str, keep_default_na=False, na_filter=False,
        on_bad_lines="skip", usecols=usecols, low_memory=True
    ):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if channel and "channel" in chunk.columns:
            chunk = chunk[chunk["channel"].str.lower() == channel.lower()]
        if "id" in chunk.columns and "data" in chunk.columns:
            total_rows += len(chunk)

    total_rows = max(total_rows, 1)
    if progress_cb:
        progress_cb(0.0, f"Rows to process: {total_rows:,}")

    # ---- prepare writers ----
    writer_one = writer_can0 = writer_can1 = None
    if split:
        writer_can0 = PartitionedCSVWriter(out_can0, row_limit, cols)  # vehicle
        writer_can1 = PartitionedCSVWriter(out_can1, row_limit, cols)  # orion
    else:
        writer_one = PartitionedCSVWriter(out_one, row_limit, cols)

    processed = 0

    # ---- PASS 2: decode and write incrementally ----
    def flush(buf: List[Dict[str,Any]]):
        if not buf:
            return
        df = pd.DataFrame(buf)
        if split:
            if "channel" in df.columns:
                ch = df["channel"].astype(str).str.lower()
                df0 = df[ch == "can0"].copy()
                df1 = df[ch == "can1"].copy()
                other = df[~ch.isin(["can0","can1"])].copy()

                if not df0.empty: writer_can0.write_df(df0)
                if not df1.empty: writer_can1.write_df(df1)

                if not other.empty:
                    # Heuristic: treat Orion-named messages as can1
                    is_can1 = other["message"].isin(can_orion.CAN1_MESSAGE_NAMES)
                    writer_can1.write_df(other[is_can1])
                    writer_can0.write_df(other[~is_can1])
            else:
                is_can1 = df["message"].isin(can_orion.CAN1_MESSAGE_NAMES)
                writer_can1.write_df(df[is_can1].copy())
                writer_can0.write_df(df[~is_can1].copy())
        else:
            writer_one.write_df(df)

    # real pass
    buf: List[Dict[str,Any]] = []
    for chunk in pd.read_csv(
        input_csv, chunksize=CHUNK_ROWS, dtype=str, keep_default_na=False, na_filter=False,
        on_bad_lines="skip", usecols=usecols, low_memory=True
    ):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        if channel and "channel" in chunk.columns:
            chunk = chunk[chunk["channel"].str.lower() == channel.lower()]
        if "id" not in chunk.columns or "data" not in chunk.columns:
            continue

        for _, r in chunk.iterrows():
            sid = r.get("id","").strip()
            if not sid:
                continue
            try:
                can_id = parse_can_id(sid)
            except Exception:
                continue

            data_s = str(r.get("data","")).replace(" ","").strip()
            data_s = (data_s[:16]).ljust(16, "0")
            try:
                data_b = bytes.fromhex(data_s)
            except Exception:
                continue

            ts   = r.get("timestamp","")
            chan = r.get("channel", None)

            # Try Orion first, then Vehicle
            recs = can_orion.decode_orion(ts, can_id, data_b)
            if recs is None:
                recs = can_vehicle.decode_vehicle(ts, can_id, data_b)

            if recs is None:
                row = dict(timestamp=ts, id_hex=hex(can_id), message="RAW", raw_data=data_s)
                if include_bits: row.update(expand_bits_bytes(data_b))
                if chan is not None: row["channel"] = chan
                buf.append(row)
            else:
                for rr in recs:
                    if include_bits:
                        rr.update(expand_bits_bytes(data_b))
                    if chan is not None:
                        rr["channel"] = chan
                    buf.append(rr)

            if len(buf) >= FLUSH_EVERY:
                flush(buf); buf.clear()

        flush(buf); buf.clear()
        processed += len(chunk)
        if progress_cb:
            progress_cb(min(processed/total_rows, 1.0), f"Processed {processed:,}/{total_rows:,}")

    # collect paths
    if split:
        outputs.extend(writer_can0.paths)
        outputs.extend(writer_can1.paths)
    else:
        outputs.extend(writer_one.paths)
    return outputs

# ---------- public API ----------
def decode_csv_one(
    input_csv: str,
    output_csv: str,
    include_bits: bool=False,
    channel: Optional[str]=None,
    row_limit: int = EXCEL_XLSX_MAX_ROWS,
    progress_cb: Optional[Callable[[float, Optional[str]], None]] = None,
) -> List[str]:
    return _stream_decode(
        input_csv=input_csv,
        include_bits=include_bits,
        channel=channel,
        split=False,
        out_one=output_csv,
        out_can0=None,
        out_can1=None,
        row_limit=row_limit,
        progress_cb=progress_cb,
    )

def decode_csv_split(
    input_csv: str,
    out_can0_csv: str,
    out_can1_csv: str,
    include_bits: bool=False,
    channel: Optional[str]=None,
    row_limit: int = EXCEL_XLSX_MAX_ROWS,
    progress_cb: Optional[Callable[[float, Optional[str]], None]] = None,
) -> List[str]:
    return _stream_decode(
        input_csv=input_csv,
        include_bits=include_bits,
        channel=channel,
        split=True,
        out_one=None,
        out_can0=out_can0_csv,
        out_can1=out_can1_csv,
        row_limit=row_limit,
        progress_cb=progress_cb,
    )

if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser(description="CAN CSV decoder (GUI or CLI).")
    ap.add_argument("input_csv", nargs="?", help="CSV to decode (omit to open GUI)")
    ap.add_argument("-o", "--output", default="decoded.csv")
    ap.add_argument("--split", action="store_true", help="write <base>_can0.csv and <base>_can1.csv")
    ap.add_argument("--channel", choices=["can0","can1"])
    ap.add_argument("--bits", action="store_true", help="expand raw bytes & bitfields")
    ap.add_argument("--xls", action="store_true", help="use old .xls row limit (65,536)")
    ap.add_argument("--gui", action="store_true", help="force GUI even if input_csv is provided")
    args = ap.parse_args()

    # Centralized start:
    if args.gui or not args.input_csv:
        from can_gui_decode import launch_gui
        launch_gui()
    else:
        limit = EXCEL_XLSX_MAX_ROWS if not args.xls else EXCEL_XLS_MAX_ROWS
        if args.split:
            base, ext = os.path.splitext(args.output)
            out0 = f"{base}_can0.csv"
            out1 = f"{base}_can1.csv"
            paths = decode_csv_split(
                args.input_csv, out0, out1,
                include_bits=args.bits, channel=args.channel, row_limit=limit
            )
        else:
            paths = decode_csv_one(
                args.input_csv, args.output,
                include_bits=args.bits, channel=args.channel, row_limit=limit
            )
        print("Wrote:")
        for p in paths:
            print("  ", p)