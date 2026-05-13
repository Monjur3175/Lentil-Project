import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

# ---------------------------------------------------
# WEATHER DATA PARSER (Same as before)
# ---------------------------------------------------

def parse_weather_data(file_path):
    """Parse messy Excel weather sheet containing side-by-side monthly tables."""
    
    try:
        df_raw = pd.read_excel(file_path, header=None)
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        return pd.DataFrame()

    records = []
    month_headers = []

    month_map = {
        'jan': '01', 'january': '01', 'feb': '02', 'february': '02',
        'mar': '03', 'march': '03', 'apr': '04', 'april': '04',
        'may': '05', 'jun': '06', 'june': '06', 'jul': '07', 'july': '07',
        'aug': '08', 'august': '08', 'sep': '09', 'september': '09',
        'oct': '10', 'october': '10', 'nov': '11', 'november': '11',
        'dec': '12', 'december': '12'
    }

    for row_idx, row in df_raw.iterrows():
        # Detect Month Headers
        for col_idx, cell in enumerate(row):
            if pd.notna(cell):
                txt = str(cell).strip()
                match = re.search(r'Month:\s*([A-Za-z]+)\s*/\s*(\d{4})', txt, re.IGNORECASE)
                if match:
                    month_name = match.group(1).lower()
                    year = match.group(2)
                    if month_name in month_map:
                        month_num = month_map[month_name]
                        month_headers.append((col_idx, month_num, year))

        # Detect Left and Right Tables
        for start_col in [0, 11]:
            if start_col >= len(row): continue
            val = row.iloc[start_col]
            if pd.isna(val): continue
            try:
                day = int(float(val))
            except: continue
            if not (1 <= day <= 31): continue

            applicable = None
            for h_col, h_month, h_year in month_headers:
                if h_col <= start_col:
                    applicable = (h_month, h_year)
            if applicable is None: continue
            month, year = applicable
            if start_col + 8 >= len(row): continue

            data = row.iloc[start_col + 1:start_col + 9]
            vals = []
            for item in data:
                if pd.isna(item):
                    vals.append(np.nan)
                else:
                    txt = str(item).replace(',', '').strip()
                    try:
                        vals.append(float(txt))
                    except:
                        vals.append(np.nan)

            if len(vals) == 8:
                records.append({
                    'Day': day, 'Month': month, 'Year': year,
                    'Max_Temp': vals[0], 'Min_Temp': vals[1],
                    'Hum_6am': vals[2], 'Hum_6pm': vals[3],
                    'Rainfall': vals[4], 'Sunshine': vals[5],
                    'ET_Day': vals[6], 'ET_Night': vals[7]
                })

    if not records:
        print("❌ No records found.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    cols = ['Max_Temp', 'Min_Temp', 'Hum_6am', 'Hum_6pm', 'Rainfall', 'Sunshine', 'ET_Day', 'ET_Night']
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace([-1, -2], np.nan)

    df['Date'] = pd.to_datetime(df['Year'] + '-' + df['Month'] + '-' + df['Day'].astype(str), errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    df['Month_Period'] = df['Date'].dt.to_period('M')
    return df


# ---------------------------------------------------
# SEPARATE PLOTTING FUNCTIONS
# ---------------------------------------------------

def plot_temperature_monthly(df, save_path="01_temperature_monthly.png"):
    """Plot monthly average temperature (Max & Min)"""
    monthly = df.groupby('Month_Period').agg({'Max_Temp': 'mean', 'Min_Temp': 'mean'}).reset_index()
    monthly['Label'] = monthly['Month_Period'].astype(str)
    x = np.arange(len(monthly))
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, monthly['Max_Temp'], 'r-o', label='Avg Max Temp (°C)', linewidth=2, markersize=6)
    plt.plot(x, monthly['Min_Temp'], 'b-s', label='Avg Min Temp (°C)', linewidth=2, markersize=6)
    plt.fill_between(x, monthly['Min_Temp'], monthly['Max_Temp'], color='gray', alpha=0.2)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Temperature (°C)', fontweight='bold')
    plt.title('Monthly Average Temperature (July 2024 - June 2025)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.xticks(x, monthly['Label'], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")


def plot_humidity_monthly(df, save_path="02_humidity_monthly.png"):
    """Plot monthly average humidity (6 AM & 6 PM)"""
    monthly = df.groupby('Month_Period').agg({'Hum_6am': 'mean', 'Hum_6pm': 'mean'}).reset_index()
    monthly['Label'] = monthly['Month_Period'].astype(str)
    x = np.arange(len(monthly))
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, monthly['Hum_6am'], 'g-o', label='Avg 6:00 AM Humidity (%)', linewidth=2, markersize=6)
    plt.plot(x, monthly['Hum_6pm'], 'orange', marker='s', label='Avg 6:00 PM Humidity (%)', linewidth=2, markersize=6)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Humidity (%)', fontweight='bold')
    plt.title('Monthly Average Relative Humidity (July 2024 - June 2025)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left')
    plt.ylim(0, 105)
    plt.xticks(x, monthly['Label'], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")


def plot_rainfall_monthly(df, save_path="03_rainfall_monthly.png"):
    """Plot monthly average daily rainfall"""
    monthly = df.groupby('Month_Period')['Rainfall'].mean().reset_index()
    monthly['Label'] = monthly['Month_Period'].astype(str)
    x = np.arange(len(monthly))
    
    plt.figure(figsize=(12, 5))
    bars = plt.bar(x, monthly['Rainfall'], color='#4A90E2', edgecolor='#2C5282', alpha=0.8)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Avg Daily Rainfall (mm)', fontweight='bold')
    plt.title('Monthly Average Daily Rainfall (July 2024 - June 2025)', fontsize=14, fontweight='bold')
    plt.xticks(x, monthly['Label'], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")


def plot_sunshine_monthly(df, save_path="04_sunshine_monthly.png"):
    """Plot monthly average daily sunshine hours"""
    monthly = df.groupby('Month_Period')['Sunshine'].mean().reset_index()
    monthly['Label'] = monthly['Month_Period'].astype(str)
    x = np.arange(len(monthly))
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, monthly['Sunshine'], 'D-', color='#D4AF37', linewidth=2, markersize=8, markerfacecolor='gold')
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Avg Daily Sunshine (Hours)', fontweight='bold')
    plt.title('Monthly Average Daily Sunshine Hours (July 2024 - June 2025)', fontsize=14, fontweight='bold')
    plt.xticks(x, monthly['Label'], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")


def plot_et_monthly(df, save_path="05_evapotranspiration_monthly.png"):
    """Plot monthly average evapotranspiration (Day & Night)"""
    monthly = df.groupby('Month_Period').agg({'ET_Day': 'mean', 'ET_Night': 'mean'}).reset_index()
    monthly['Label'] = monthly['Month_Period'].astype(str)
    x = np.arange(len(monthly))
    
    plt.figure(figsize=(12, 5))
    plt.plot(x, monthly['ET_Day'], 'r-o', label='Avg Day ET (mm)', linewidth=2, markersize=6)
    plt.plot(x, monthly['ET_Night'], 'b-s', label='Avg Night ET (mm)', linewidth=2, markersize=6)
    plt.fill_between(x, monthly['ET_Night'], monthly['ET_Day'], color='purple', alpha=0.2)
    plt.xlabel('Month', fontweight='bold')
    plt.ylabel('Evapotranspiration (mm)', fontweight='bold')
    plt.title('Monthly Average Evapotranspiration (July 2024 - June 2025)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.xticks(x, monthly['Label'], rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Saved: {save_path}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":
    FILE_PATH = "Weather data 24-25.xlsx"
    
    print("📊 Loading weather data...")
    df = parse_weather_data(FILE_PATH)
    
    if not df.empty:
        print(f"✅ Parsed {len(df)} daily records")
        print("\n📈 Generating 5 separate monthly plots...\n")
        
        plot_temperature_monthly(df)
        plot_humidity_monthly(df)
        plot_rainfall_monthly(df)
        plot_sunshine_monthly(df)
        plot_et_monthly(df)
        
        print("\n🎉 All plots saved successfully!")
        print("Files created:")
        print("  • 01_temperature_monthly.png")
        print("  • 02_humidity_monthly.png")
        print("  • 03_rainfall_monthly.png")
        print("  • 04_sunshine_monthly.png")
        print("  • 05_evapotranspiration_monthly.png")
    else:
        print("❌ Parsing failed - check file path and format")