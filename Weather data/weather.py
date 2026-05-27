"""
Visualization of 5-Location Weather & Soil Data
================================================
Generates 3 separate figures:
  1. Monthly Max & Min Temperature per location
  2. Monthly Rainfall per location
  3. Soil Properties heatmap across locations

Dependencies: pandas, matplotlib, seaborn, openpyxl
Install: pip install pandas matplotlib seaborn openpyxl
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
FILE = "5 location weather data.xlsx"
LOCATIONS = ["Godagari", "Shyampur", "Nachole", "Sapaher", "Iswardi"]
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

# Excel serial → datetime
def excel_to_date(serial):
    # If pandas already parsed it, return as-is
    if isinstance(serial, datetime):
        return serial
    # Handle empty/NaN cells
    if pd.isna(serial):
        return pd.NaT
    # Convert Excel serial number (float/int) to Python datetime
    return datetime.fromordinal(datetime(1899, 12, 30).toordinal() + int(float(serial)))

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
xf = pd.ExcelFile(FILE)
raw = xf.parse("Sheet1", header=None)

# ── Temperature block (rows 0–13) ──
temp_header_row = 0   # "Temperature", locations
temp_col_row    = 1   # "Months/25", Max, Min ...
temp_data_rows  = slice(2, 14)

temp_raw = raw.iloc[temp_data_rows, :].copy()
date_col = temp_raw.iloc[:, 0].apply(excel_to_date)

temp_data = {}
for i, loc in enumerate(LOCATIONS):
    col_max = 1 + i * 2
    col_min = 2 + i * 2
    temp_data[loc] = pd.DataFrame({
        "Date": date_col.values,
        "Max":  pd.to_numeric(temp_raw.iloc[:, col_max].values, errors="coerce"),
        "Min":  pd.to_numeric(temp_raw.iloc[:, col_min].values, errors="coerce"),
    })

# ── Rainfall block (rows 15–28) ──
rain_start = None
for i, val in enumerate(raw.iloc[:, 0]):
    if str(val).strip().lower().startswith("rainfall"):
        rain_start = i
        break

rain_raw = raw.iloc[rain_start + 1 : rain_start + 13, :].copy()
rain_date_col = rain_raw.iloc[:, 0].apply(excel_to_date)

rain_data = pd.DataFrame({"Date": rain_date_col.values})
for i, loc in enumerate(LOCATIONS):
    rain_data[loc] = pd.to_numeric(rain_raw.iloc[:, i + 1].values, errors="coerce")

# ── Soil properties block ──
soil_props  = ["Soil pH", "Organic matter (%)", "Total nitrogen (%)",
               "P (µg/g)", "K (meq/100g)", "S (µg/g)", "Zn (µg/g)", "B (µg/g)"]
soil_values = [
    [6.7,  7.7,  5.5,  4.9,  8.2 ],   # pH
    [1.45, 1.29, 1.57, 1.87, 1.88],   # OM
    [0.08, 0.09, 0.09, 0.08, 0.15],   # N
    [14.5, 32.4, 7.4,  28.5, 9.0 ],   # P
    [0.14, 0.29, 0.23, 0.15, 0.25],   # K
    [63.75,25.6, 22.43,25.4, 25.1],   # S
    [1.13, 1.27, 2.61, 0.85, 0.81],   # Zn
    [0.45, 0.85, 0.27, 0.25, 0.18],   # B
]
soil_df = pd.DataFrame(soil_values, index=soil_props, columns=LOCATIONS)


# ═════════════════════════════════════════════
# FIGURE 1 — Monthly Temperature (Max & Min)
# ═════════════════════════════════════════════
fig1, axes1 = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
fig1.suptitle("Monthly Max & Min Temperature by Location (2024–2025)",
              fontsize=15, fontweight="bold", y=0.98)

for ax, (loc, color) in zip(axes1, zip(LOCATIONS, PALETTE)):
    df = temp_data[loc]
    ax.plot(df["Date"], df["Max"], "o-", color=color, lw=2, ms=5, label="Max Temp")
    ax.plot(df["Date"], df["Min"], "s--", color=color, lw=1.5, ms=5,
            alpha=0.65, label="Min Temp")
    ax.fill_between(df["Date"], df["Min"], df["Max"], color=color, alpha=0.10)
    ax.set_ylabel("°C", fontsize=10)
    ax.set_title(loc, fontsize=11, fontweight="bold", loc="left", pad=4)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(df["Date"].min(), df["Date"].max())

axes1[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes1[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(axes1[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
fig1.tight_layout(rect=[0, 0, 1, 0.97])
fig1.savefig("fig1_temperature.png", dpi=150, bbox_inches="tight")
print("✔  Saved: fig1_temperature.png")


# ═════════════════════════════════════════════
# FIGURE 2 — Monthly Rainfall
# ═════════════════════════════════════════════
fig2, axes2 = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
fig2.suptitle("Monthly Rainfall by Location (2024–2025)",
              fontsize=15, fontweight="bold", y=0.98)

for ax, (loc, color) in zip(axes2, zip(LOCATIONS, PALETTE)):
    ax.bar(rain_data["Date"], rain_data[loc], color=color, alpha=0.80,
           width=20, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("mm", fontsize=10)
    ax.set_title(loc, fontsize=11, fontweight="bold", loc="left", pad=4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes2[-1].xaxis.set_major_locator(mdates.MonthLocator())
plt.setp(axes2[-1].xaxis.get_majorticklabels(), rotation=45, ha="right")
fig2.tight_layout(rect=[0, 0, 1, 0.97])
fig2.savefig("fig2_rainfall.png", dpi=150, bbox_inches="tight")
print("✔  Saved: fig2_rainfall.png")


# ═════════════════════════════════════════════
# FIGURE 3 — Soil Properties Heatmap
# ═════════════════════════════════════════════

# Row-wise z-score normalisation so different units are comparable
soil_norm = soil_df.apply(lambda row: (row - row.mean()) / row.std(), axis=1)

fig3, (ax_heat, ax_bar) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={"width_ratios": [2, 3]})
fig3.suptitle("Soil Properties Across 5 Locations", fontsize=14, fontweight="bold")

# Heatmap (normalised)
sns.heatmap(soil_norm, ax=ax_heat, cmap="RdYlGn", annot=False,
            linewidths=0.5, linecolor="white", cbar_kws={"label": "Z-score"})
ax_heat.set_title("Normalised Heatmap\n(row z-score)", fontsize=11)
ax_heat.set_xlabel("")
ax_heat.tick_params(axis="x", rotation=30)
ax_heat.tick_params(axis="y", rotation=0)

# Grouped bar — raw values for selected key nutrients
key_props = ["Soil pH", "P (µg/g)", "S (µg/g)", "Zn (µg/g)"]
key_df = soil_df.loc[key_props].T.reset_index()
key_df.columns = ["Location"] + key_props

x = np.arange(len(LOCATIONS))
width = 0.18
offsets = np.linspace(-(len(key_props) - 1) / 2, (len(key_props) - 1) / 2, len(key_props)) * width

for offset, prop, color in zip(offsets, key_props,
                                ["#4C72B0","#DD8452","#55A868","#C44E52"]):
    ax_bar.bar(x + offset, key_df[prop], width, label=prop,
               color=color, alpha=0.85, edgecolor="white")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(LOCATIONS, rotation=20, ha="right")
ax_bar.set_ylabel("Value (native units)")
ax_bar.set_title("Key Soil Nutrients by Location", fontsize=11)
ax_bar.legend(fontsize=9, loc="upper right")
ax_bar.yaxis.grid(True, linestyle="--", alpha=0.5)
ax_bar.set_axisbelow(True)

fig3.tight_layout()
fig3.savefig("fig3_soil_properties.png", dpi=150, bbox_inches="tight")
print("✔  Saved: fig3_soil_properties.png")

plt.show()
print("\nDone. All 3 figures saved.")