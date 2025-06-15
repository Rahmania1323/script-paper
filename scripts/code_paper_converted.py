#!/usr/bin/env python
# coding: utf-8

# # RESULT

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Tampilkan semua kolom
pd.set_option('display.max_columns', None)

# Baca file Excel dari drive F: dan sheet 'kumpulan total'
df = pd.read_excel("F:\Data Penerbangan\DATA DELAY1.xlsx")

# Tampilkan 20 baris pertama
df.head(133)


# In[3]:


import pandas as pd
import statsmodels.api as sm

# 1. Baca file Excel dan gabungkan semua sheet
file_path = 'F:\\Data Penerbangan\\DATA DELAY1.xlsx'
xls = pd.ExcelFile(file_path)
all_data = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)

# 2. Pilih kolom yang akan dianalisis
# (pastikan nama kolom sama persis seperti di file-mu)
cuaca_cols = ['Temperature', 'Pressure', 'Precipitation', 'Relative Humidity (RH)', 'Wind Speed at 2 m', 'Wind Speed at 10 m']
target_col = 'Delay'

# 3. Hapus baris yang ada NaN di kolom penting
data = all_data[cuaca_cols + [target_col]].dropna()

# 4. Siapkan X dan y
X = data[cuaca_cols]
y = data[target_col]

# Tambahkan konstanta ke X
X = sm.add_constant(X)

# 5. Regresi linear OLS
model = sm.OLS(y, X).fit()

# 6. Tampilkan hasil regresi
print(model.summary())


# In[4]:


# Prediksi dari model
y_pred = model.fittedvalues

# Hitung SS
SS_total = np.sum((y - np.mean(y))**2)                    # SST
SS_residual = np.sum((y - y_pred)**2)                     # SSR
SS_regression = np.sum((y_pred - np.mean(y))**2)          # SSM

# Hitung MS
df_model = model.df_model
df_resid = model.df_resid

MS_regression = SS_regression / df_model
MS_residual = SS_residual / df_resid

# Cetak hasil
print(f"SS Total      (SST): {SS_total:.2f}")
print(f"SS Regression (SSR): {SS_regression:.2f}")
print(f"SS Residual   (SSE): {SS_residual:.2f}")
print(f"MS Regression       : {MS_regression:.2f}")
print(f"MS Residual         : {MS_residual:.2f}")


# In[5]:


import pandas as pd

# Baca semua sheet dan gabungkan
file_path = 'F:/Data Penerbangan/DATA DELAY1.xlsx'
xls = pd.ExcelFile(file_path)
df = pd.concat([xls.parse(sheet) for sheet in xls.sheet_names], ignore_index=True)

# Bersihkan spasi dari nama kolom
df.columns = df.columns.str.strip()


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns

# Pastikan kolom tidak punya spasi aneh
df.columns = df.columns.str.strip()

# Ganti nama kolom ke Bahasa Inggris (jika perlu)
df = df.rename(columns={
    'Suhu': 'Temperature',
    'Tekanan': 'Pressure',
    'Curah Hujan': 'Precipitation',
    'RH': 'Relative Humidity (RH)',
    'Kecepatan Angin 2m': 'Wind Speed at 2 m',
    'Kecepatan Angin 10m': 'Wind Speed at 10 m'
})

# Daftar kolom cuaca yang dipakai
weather_columns = ['Temperature', 'Pressure', 'Precipitation',
                   'Relative Humidity (RH)', 'Wind Speed at 2 m', 'Wind Speed at 10 m']

# Gaya visual dan ukuran plot
sns.set(style="whitegrid")
plt.figure(figsize=(15, 9))

# Plot per variabel
for i, col in enumerate(weather_columns):
    plt.subplot(2, 3, i+1)
    sns.regplot(x=col, y='Delay', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title(f'Delay vs {col}')
    plt.xlabel(col)
    plt.ylabel('Delay')

plt.tight_layout()
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Pastikan nama kolom bersih
df.columns = df.columns.str.strip()

# Rename kolom ke Bahasa Inggris jika perlu
df = df.rename(columns={
    'Suhu': 'Temperature',
    'Tekanan': 'Pressure',
    'Curah Hujan': 'Precipitation',
    'RH': 'Relative Humidity (RH)',
    'Kecepatan Angin 2m': 'Wind Speed at 2 m',
    'Kecepatan Angin 10m': 'Wind Speed at 10 m'
})

# Kolom cuaca
weather_columns = ['Temperature', 'Pressure', 'Precipitation',
                   'Relative Humidity (RH)', 'Wind Speed at 2 m', 'Wind Speed at 10 m']

# Gaya visual
sns.set(style="whitegrid")
plt.figure(figsize=(15, 9))

# Loop plotting
for i, col in enumerate(weather_columns):
    plt.subplot(2, 3, i+1)

    # Hitung R²
    X = sm.add_constant(df[col])
    y = df['Delay']
    model = sm.OLS(y, X).fit()
    r2 = model.rsquared

    # Plot regresi
    sns.regplot(x=col, y='Delay', data=df, scatter_kws={'alpha': 0.6}, line_kws={'color': 'red'})

    # Tambahkan teks R² di kanan atas dalam koordinat axes
    plt.text(0.95, 0.90, f'$R^2$ = {r2:.3f}',
             horizontalalignment='right',
             verticalalignment='top',
             transform=plt.gca().transAxes,
             fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    # Label
    plt.title(f'Delay vs {col}')
    plt.xlabel(col)
    plt.ylabel('Delay')

plt.tight_layout()
plt.show()


# In[9]:


import statsmodels.api as sm

# Inisialisasi list kolom cuaca
weather_columns = ['Temperature', 'Pressure', 'Precipitation',
                   'Relative Humidity (RH)', 'Wind Speed at 2 m', 'Wind Speed at 10 m']

# Loop untuk menghitung dan mencetak R²
for col in weather_columns:
    X = sm.add_constant(df[col])  # Tambah intersep
    y = df['Delay']
    model = sm.OLS(y, X).fit()
    r2 = model.rsquared
    print(f'R² for Delay vs {col}: {r2:.3f}')


# In[11]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Tampilkan semua kolom
pd.set_option('display.max_columns', None)

# Baca file Excel dari drive F: dan sheet 'kumpulan total'
df = pd.read_excel("F:\Data Penerbangan\ot_delaycause1_DL\Formatted_Weather_Averages.xlsx", sheet_name='All_Seasons')

# Tampilkan 20 baris pertama
df.head(119)


# In[12]:


import plotly.express as px
import plotly.colors as pc

# Get unique airline names
unique_carriers = df['carrier_name'].unique()

# Create a manual color map using Plotly's qualitative palette
palette = pc.qualitative.Plotly  # You can also try 'D3', 'Pastel', etc.
color_map = {carrier: palette[i % len(palette)] for i, carrier in enumerate(unique_carriers)}

# Function to plot top 10 airlines per season
def plot_per_season(season_name):
    df_season = df[df['season'] == season_name].sort_values(by='arr_del15', ascending=False).head(10)
    fig = px.bar(
        df_season,
        x='carrier_name',
        y='arr_del15',
        color='carrier_name',
        title=f'Top 10 Airlines with Arrival Delays >15 Minutes – Season: {season_name}',
        labels={'carrier_name': 'Airline', 'arr_del15': 'Number of Arrival Delays (>15 min)'},
        color_discrete_map=color_map
    )

    # Tambahkan sumber data di bawah legenda
    fig.add_annotation(
        text="Source: Bureau of Transportation Statistics (2013–2023)",
        xref="paper", yref="paper",
        x=1.02, y=-0.1,  # posisi bawah dari legend
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="left"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            title='Airlines',
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(r=350, b=100),  # tambah margin kanan dan bawah
        height=500
    )

    fig.show()


# Display plots for each season
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    plot_per_season(season)


# In[13]:


import plotly.express as px
import plotly.colors as pc

# Get unique list of airlines
unique_carriers = df['carrier_name'].unique()

# Create a manual color map for each airline (using Plotly's qualitative color palette)
palette = pc.qualitative.Plotly  # can be changed to 'D3', 'Pastel', etc.
color_map = {carrier: palette[i % len(palette)] for i, carrier in enumerate(unique_carriers)}

# Function to plot top 10 airlines by weather-related delay count per season
def plot_by_season(season_name):
    df_season = df[df['season'] == season_name].sort_values(by='weather_ct', ascending=False).head(10)

    fig = px.bar(
        df_season,
        x='carrier_name',
        y='weather_ct',
        color='carrier_name',
        title=f'Top 10 Airlines by Weather-Related Delay Count – Season: {season_name}',
        labels={'carrier_name': 'Airline', 'weather_ct': 'Number of Flights with Weather Delay Cause'},
        color_discrete_map=color_map
    )

    # Add data source below the legend
    fig.add_annotation(
        text="Source: Bureau of Transportation Statistics (2013–2023)",
        xref="paper", yref="paper",
        x=1.02, y=-0.1,  # position under the legend
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="left"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(
            title='Airlines',
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(r=350, b=100),  # increase right and bottom margin
        height=500
    )

    fig.show()

# Display plots for all four seasons
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    plot_by_season(season)


# In[14]:


import plotly.express as px
import plotly.colors as pc

# Get unique list of airlines
unique_carriers = df['carrier_name'].unique()

# Create a manual color map for each airline using Plotly's qualitative color palette
palette = pc.qualitative.Plotly  # Can be replaced with 'D3', 'Pastel', etc.
color_map = {carrier: palette[i % len(palette)] for i, carrier in enumerate(unique_carriers)}

# Function to plot top 10 airlines by weather delay for a given season
def plot_by_season(season_name):
    df_season = df[df['season'] == season_name].sort_values(by='weather_delay', ascending=False).head(10)

    fig = px.bar(
        df_season,
        x='carrier_name',
        y='weather_delay',
        color='carrier_name',
        title=f'Top 10 Airlines by Weather Delay – Season: {season_name}',
        labels={
            'carrier_name': 'Airline',
            'weather_delay': ' Total Weather Delay Time (minutes) '
        },
        color_discrete_map=color_map
    )

    # Add data source below the legend
    fig.add_annotation(
        text="Source: Bureau of Transportation Statistics (2013–2023)",
        xref="paper", yref="paper",
        x=1.02, y=-0.1,
        showarrow=False,
        font=dict(size=11, color="gray"),
        xanchor="left"
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        plot_bgcolor= 'white',
        showlegend=True,
        legend=dict(
            title='Airlines',
            orientation='v',
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            traceorder='normal',
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0)',
            bordercolor='lightgray',
            borderwidth=1
        ),
        margin=dict(r=350, b=100),
        height=500
    )

    fig.show()

# Display the chart for each season
for season in ['DJF', 'MAM', 'JJA', 'SON']:
    plot_by_season(season)

Add converted notebook script

