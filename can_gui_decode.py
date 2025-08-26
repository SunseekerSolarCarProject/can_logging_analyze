"""
GUI front-end (CustomTkinter) for selecting CSV and running the streaming decoder.

Run:
    python can_gui_decode.py
"""

from __future__ import annotations
import threading
from typing import List, Optional

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox

from main import (
    decode_csv_one, decode_csv_split,
    EXCEL_XLSX_MAX_ROWS, EXCEL_XLS_MAX_ROWS
)

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CAN Decoder — Modern")
        self.geometry("840x480")
        self.minsize(820, 440)

        # ---- String / Boolean vars ----
        self.in_path = ctk.StringVar(value="")
        self.out_path = ctk.StringVar(value="decoded.csv")
        self.channel = ctk.StringVar(value="all")           # all | can0 | can1
        self.expand_bits = ctk.BooleanVar(value=False)
        self.split_subsystems = ctk.BooleanVar(value=True)  # split into can0/can1 files
        self.xls_limit = ctk.BooleanVar(value=False)        # old .xls row cap

        # theme controls
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
        self.appearance_var = ctk.StringVar(value="System")
        self.color_var = ctk.StringVar(value="blue")

        # layout
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        # Left column
        self.left = ctk.CTkFrame(self, corner_radius=16)
        self.left.grid(row=0, column=0, sticky="nsew", padx=(16, 8), pady=16)
        self.left.columnconfigure(1, weight=1)

        # Right column
        self.right = ctk.CTkFrame(self, corner_radius=16)
        self.right.grid(row=0, column=1, sticky="nsew", padx=(8, 16), pady=16)
        self.right.columnconfigure(0, weight=1)
        self.right.rowconfigure(3, weight=1)

        # ---- Left: file pickers ----
        ctk.CTkLabel(self.left, text="Input CSV", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, padx=12, pady=(16, 4), sticky="w")
        entry_in = ctk.CTkEntry(self.left, textvariable=self.in_path, placeholder_text="Select CAN log CSV…")
        entry_in.grid(row=0, column=1, padx=(0,12), pady=(16,4), sticky="ew")
        ctk.CTkButton(self.left, text="Browse…", command=self.browse_in).grid(row=0, column=2, padx=(0,12), pady=(16,4))

        ctk.CTkLabel(self.left, text="Output file / base", font=("Segoe UI", 12, "bold")).grid(row=1, column=0, padx=12, pady=4, sticky="w")
        entry_out = ctk.CTkEntry(self.left, textvariable=self.out_path, placeholder_text="decoded.csv")
        entry_out.grid(row=1, column=1, padx=(0,12), pady=4, sticky="ew")
        ctk.CTkButton(self.left, text="Browse…", command=self.browse_out).grid(row=1, column=2, padx=(0,12), pady=4)

        ctk.CTkLabel(self.left, text="Channel", font=("Segoe UI", 12, "bold")).grid(row=2, column=0, padx=12, pady=(10,4), sticky="w")
        ctk.CTkOptionMenu(self.left, values=["all", "can0", "can1"], variable=self.channel, width=120).grid(row=2, column=1, padx=(0,12), pady=(10,4), sticky="w")

        self.switch_split = ctk.CTkSwitch(self.left, text="Split into CAN0 / CAN1 files", variable=self.split_subsystems)
        self.switch_split.grid(row=3, column=0, columnspan=3, padx=12, pady=(10,4), sticky="w")

        self.switch_bits = ctk.CTkSwitch(self.left, text="Expand raw bytes and bits", variable=self.expand_bits)
        self.switch_bits.grid(row=4, column=0, columnspan=3, padx=12, pady=(4,6), sticky="w")

        self.switch_xls = ctk.CTkSwitch(self.left, text="Old Excel .xls row limit (65,536)", variable=self.xls_limit)
        self.switch_xls.grid(row=5, column=0, columnspan=3, padx=12, pady=(4,16), sticky="w")

        # Decode buttons row
        btn_row = ctk.CTkFrame(self.left, fg_color="transparent")
        btn_row.grid(row=6, column=0, columnspan=3, sticky="ew", padx=12, pady=(0,16))
        btn_row.columnconfigure(0, weight=1)
        ctk.CTkButton(btn_row, text="Decode", height=40, command=self.on_decode).grid(row=0, column=0, sticky="ew", padx=(0,6))
        ctk.CTkButton(btn_row, text="Quit", height=40, fg_color=("gray85", "gray20"), command=self.destroy).grid(row=0, column=1, sticky="ew", padx=(6,0))

        # ---- Right: Appearance, progress, status ----
        theme_row = ctk.CTkFrame(self.right, fg_color="transparent")
        theme_row.grid(row=0, column=0, sticky="ew", padx=12, pady=(16,4))
        theme_row.columnconfigure(2, weight=1)
        ctk.CTkLabel(theme_row, text="Appearance", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkOptionMenu(theme_row, values=["System", "Light", "Dark"], variable=self.appearance_var,
                          command=self.on_appearance_change, width=110).grid(row=0, column=1, padx=(8,0))
        ctk.CTkOptionMenu(theme_row, values=["blue", "green", "dark-blue"], variable=self.color_var,
                          command=self.on_color_change, width=120).grid(row=0, column=2, padx=(8,0), sticky="w")

        # Determinate progress bar + label
        self.progress = ctk.CTkProgressBar(self.right)
        self.progress.grid(row=1, column=0, sticky="ew", padx=16, pady=(16,6))
        self.progress.set(0)
        self.progress_lbl = ctk.CTkLabel(self.right, text="0%")
        self.progress_lbl.grid(row=1, column=0, sticky="e", padx=16, pady=(16,6))

        ctk.CTkLabel(self.right, text="Status", font=("Segoe UI", 12, "bold")).grid(row=2, column=0, sticky="w", padx=16, pady=(4,2))
        self.status = ctk.CTkTextbox(self.right, wrap="word", height=220)
        self.status.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0,16))
        self.append_status("Ready.\n")

        foot = ctk.CTkLabel(self.right, text="Orion CAN1 + Vehicle CAN0 Decoder", text_color=("gray30","gray70"))
        foot.grid(row=4, column=0, sticky="e", padx=16, pady=(0,12))

    # ---- UI helpers ----
    def _set_progress_ui(self, frac: float, note: str | None = None):
        frac = max(0.0, min(1.0, float(frac)))
        self.progress.set(frac)
        self.progress_lbl.configure(text=f"{int(round(frac*100))}%")
        if note:
            self.append_status(note + "\n")

    def set_progress(self, frac: float, note: str | None = None):
        self.after(0, lambda: self._set_progress_ui(frac, note))

    def append_status(self, text: str):
        self.status.insert("end", text)
        self.status.see("end")

    def browse_in(self):
        p = filedialog.askopenfilename(title="Select CAN CSV", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if p:
            self.in_path.set(p)

    def browse_out(self):
        p = filedialog.asksaveasfilename(title="Save decoded CSV as…", defaultextension=".csv", filetypes=[("CSV files","*.csv")])
        if p:
            self.out_path.set(p)

    def on_appearance_change(self, mode: str):
        ctk.set_appearance_mode(mode)

    def on_color_change(self, theme: str):
        ctk.set_default_color_theme(theme)

    # ---- Decode button handler ----
    def on_decode(self):
        ip = self.in_path.get().strip()
        op = self.out_path.get().strip() or "decoded.csv"
        if not ip:
            messagebox.showerror("Missing input", "Please choose an input CSV file.")
            return

        split = self.split_subsystems.get()
        include_bits = self.expand_bits.get()
        ch = self.channel.get()
        channel = None if ch == "all" else ch
        row_limit = EXCEL_XLSX_MAX_ROWS if not self.xls_limit.get() else EXCEL_XLS_MAX_ROWS

        self.append_status(
            f"\nDecoding:\n  input: {ip}\n  output: {op}\n  channel: {ch}\n"
            f"  split: {split}\n  expand_bits: {include_bits}\n"
            f"  row_limit: {row_limit}\n"
        )
        self.set_progress(0.0, "Starting…")

        def work():
            try:
                outputs: List[str] = []

                def progress_cb(fr: float, note: str | None = None):
                    self.set_progress(fr, note)

                if split:
                    base = op[:-4] if op.lower().endswith(".csv") else op
                    out0 = base + "_can0.csv"
                    out1 = base + "_can1.csv"
                    outputs = decode_csv_split(
                        ip, out0, out1,
                        include_bits=include_bits, channel=channel,
                        row_limit=row_limit, progress_cb=progress_cb
                    )
                else:
                    outputs = decode_csv_one(
                        ip, op,
                        include_bits=include_bits, channel=channel,
                        row_limit=row_limit, progress_cb=progress_cb
                    )

                self.set_progress(1.0, "Finished.")
                self.append_status("Done:\n  " + "\n  ".join(outputs) + "\n")
                messagebox.showinfo("Done", "Wrote:\n" + "\n".join(outputs))
            except Exception as e:
                self.set_progress(0.0)
                self.append_status(f"Error: {e}\n")
                messagebox.showerror("Error", str(e))

        threading.Thread(target=work, daemon=True).start()

def launch_gui():
    App().mainloop()
