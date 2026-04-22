"""
Hyperliquid × Fear/Greed Sentiment Analysis
Full pipeline: data prep → analysis → charts → model → outputs
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings, os

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE   = os.path.dirname(os.path.abspath(__file__))
RAW    = os.path.join(BASE, 'data', 'raw')
PROC   = os.path.join(BASE, 'data', 'processed')
CHARTS = os.path.join(BASE, 'charts')
OUT    = os.path.join(BASE, 'outputs')

for d in [PROC, CHARTS, OUT]:
    os.makedirs(d, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style='darkgrid', palette='deep')
plt.rcParams.update({
    'figure.dpi': 140,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
})

FEAR_COLOR  = '#e74c3c'
GREED_COLOR = '#27ae60'
NEUTRAL_COLOR = '#f39c12'
PALETTE = {
    'Extreme Fear': '#c0392b',
    'Fear':         '#e74c3c',
    'Neutral':      '#f39c12',
    'Greed':        '#27ae60',
    'Extreme Greed':'#1a7a3e',
}
BINARY_PALETTE = {'Fear': FEAR_COLOR, 'Greed': GREED_COLOR}

print("=" * 65)
print("  HYPERLIQUID × FEAR/GREED SENTIMENT ANALYSIS")
print("=" * 65)

# ╔══════════════════════════════════════════════════════════════╗
# ║  PART A — DATA PREPARATION                                   ║
# ╚══════════════════════════════════════════════════════════════╝

# ── A1 · Load datasets ───────────────────────────────────────────
print("\n── A1  Loading datasets ──────────────────────────────────────")

fg = pd.read_csv(os.path.join(RAW, 'fear_greed.csv'))
hl = pd.read_csv(os.path.join(RAW, 'hyperliquid_trades.csv'))

print(f"Fear/Greed  : {fg.shape[0]:,} rows × {fg.shape[1]} cols  →  {fg.columns.tolist()}")
print(f"Hyperliquid : {hl.shape[0]:,} rows × {hl.shape[1]} cols  →  {hl.columns.tolist()}")

print("\nFear/Greed missing values :")
print(fg.isnull().sum().to_string())
print(f"Fear/Greed duplicates     : {fg.duplicated().sum()}")

print("\nHyperliquid missing values :")
print(hl.isnull().sum().to_string())
print(f"Hyperliquid duplicates    : {hl.duplicated().sum()}")

# ── A2 · Clean & align by date ──────────────────────────────────
print("\n── A2  Timestamp parsing & date alignment ────────────────────")

# Fear/Greed: 'date' column already in YYYY-MM-DD format
fg['date'] = pd.to_datetime(fg['date'])
fg = fg.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
fg['classification'] = fg['classification'].str.strip()

# Binary mapping: Extreme Fear/Fear → Fear; Neutral/Greed/Extreme Greed → Greed
# We keep Neutral as its own bucket for 5-way analysis but binary excludes it
fg['sentiment_5'] = fg['classification']   # full 5-class label
fg['sentiment_binary'] = fg['classification'].map({
    'Extreme Fear': 'Fear',
    'Fear':         'Fear',
    'Neutral':      'Neutral',
    'Greed':        'Greed',
    'Extreme Greed':'Greed',
})

print(f"Fear/Greed date range  : {fg['date'].min().date()} → {fg['date'].max().date()}")
print("Sentiment distribution :")
print(fg['sentiment_5'].value_counts().to_string())

# Hyperliquid: 'Timestamp IST' is 'DD-MM-YYYY HH:MM'
hl['date'] = pd.to_datetime(hl['Timestamp IST'], format='%d-%m-%Y %H:%M', dayfirst=True)
hl['date_only'] = hl['date'].dt.normalize()

print(f"\nHyperliquid date range : {hl['date_only'].min().date()} → {hl['date_only'].max().date()}")
print(f"Unique trading days    : {hl['date_only'].nunique()}")

# Overlap
fg_dates = set(fg['date'])
hl_dates = set(hl['date_only'])
overlap  = fg_dates & hl_dates
print(f"Overlapping dates      : {len(overlap)}")

# ── A3 · Feature engineering ─────────────────────────────────────
print("\n── A3  Engineering key metrics ───────────────────────────────")

# Classify Direction into clean long/short/close buckets
open_long  = hl['Direction'].isin(['Open Long', 'Buy'])
open_short = hl['Direction'].isin(['Open Short', 'Sell'])
close_long = hl['Direction'] == 'Close Long'
close_short= hl['Direction'] == 'Close Short'
flip_ls    = hl['Direction'] == 'Long > Short'
flip_sl    = hl['Direction'] == 'Short > Long'

hl['is_long']  = (open_long  | flip_sl).astype(int)
hl['is_short'] = (open_short | flip_ls).astype(int)
hl['is_close'] = (close_long | close_short).astype(int)
hl['is_win']   = (hl['Closed PnL'] > 0).astype(int)

# ── Daily metrics per trader ──────────────────────────────────────
daily_trader = (
    hl.groupby(['Account', 'date_only'])
    .agg(
        daily_pnl     = ('Closed PnL',  'sum'),
        trade_count   = ('Closed PnL',  'count'),
        win_count     = ('is_win',       'sum'),
        long_trades   = ('is_long',      'sum'),
        short_trades  = ('is_short',     'sum'),
        close_trades  = ('is_close',     'sum'),
        avg_size_usd  = ('Size USD',     'mean'),
        total_size_usd= ('Size USD',     'sum'),
        avg_exec_price= ('Execution Price', 'mean'),
        total_fee     = ('Fee',          'sum'),
    )
    .reset_index()
    .rename(columns={'date_only': 'date'})
)

daily_trader['win_rate']   = daily_trader['win_count']   / daily_trader['trade_count']
daily_trader['long_ratio'] = daily_trader['long_trades'] / daily_trader['trade_count'].replace(0, np.nan)
daily_trader['net_pnl_after_fee'] = daily_trader['daily_pnl'] - daily_trader['total_fee']

# Merge sentiment
daily_trader = daily_trader.merge(
    fg[['date', 'sentiment_5', 'sentiment_binary', 'value']],
    on='date', how='inner'
)

# ── Daily market-wide metrics ─────────────────────────────────────
daily_market = (
    hl.groupby('date_only')
    .agg(
        total_pnl    = ('Closed PnL',  'sum'),
        trade_count  = ('Closed PnL',  'count'),
        unique_traders=('Account',      'nunique'),
        avg_size_usd = ('Size USD',     'mean'),
        total_vol_usd= ('Size USD',     'sum'),
        win_rate     = ('is_win',       'mean'),
        long_ratio   = ('is_long',      'mean'),
        total_fee    = ('Fee',          'sum'),
    )
    .reset_index()
    .rename(columns={'date_only': 'date'})
)
daily_market = daily_market.merge(
    fg[['date', 'sentiment_5', 'sentiment_binary', 'value']],
    on='date', how='inner'
)

print(f"daily_trader rows  : {len(daily_trader):,}   (account × day pairs)")
print(f"daily_market rows  : {len(daily_market):,}   (unique trading days)")
print(f"Unique accounts    : {daily_trader['Account'].nunique():,}")

# ── Trader profile (all-time aggregate) ──────────────────────────
trader_profile = (
    daily_trader.groupby('Account')
    .agg(
        total_pnl    = ('daily_pnl',    'sum'),
        avg_daily_pnl= ('daily_pnl',    'mean'),
        pnl_std      = ('daily_pnl',    'std'),
        total_trades = ('trade_count',  'sum'),
        trading_days = ('date',         'nunique'),
        avg_win_rate = ('win_rate',     'mean'),
        avg_size_usd = ('avg_size_usd', 'mean'),
        avg_long_ratio=('long_ratio',   'mean'),
    )
    .reset_index()
    .fillna(0)
)

# Segments
trade_med = trader_profile['total_trades'].median()
size_med  = trader_profile['avg_size_usd'].median()
trader_profile['freq_segment']  = np.where(trader_profile['total_trades'] >= trade_med, 'Frequent', 'Infrequent')
trader_profile['size_segment']  = np.where(trader_profile['avg_size_usd']  >= size_med,  'Large Size', 'Small Size')
trader_profile['winner_segment']= np.where(
    (trader_profile['avg_daily_pnl'] > 0) & (trader_profile['avg_win_rate'] >= 0.5),
    'Consistent Winner', 'Others'
)
# Sharpe-like consistency
trader_profile['sharpe'] = trader_profile['avg_daily_pnl'] / (trader_profile['pnl_std'] + 1e-9)
trader_profile['consistency'] = pd.qcut(trader_profile['sharpe'], 3, labels=['Inconsistent', 'Moderate', 'Consistent'])

print("\nTrader segments :")
print("  Freq :", trader_profile['freq_segment'].value_counts().to_dict())
print("  Size :", trader_profile['size_segment'].value_counts().to_dict())
print("  Win  :", trader_profile['winner_segment'].value_counts().to_dict())

# Merge segment info back to daily_trader
daily_trader = daily_trader.merge(
    trader_profile[['Account', 'freq_segment', 'size_segment', 'winner_segment', 'consistency']],
    on='Account', how='left'
)

# ── Save processed data ───────────────────────────────────────────
daily_trader.to_csv(os.path.join(PROC, 'daily_trader.csv'), index=False)
daily_market.to_csv(os.path.join(PROC, 'daily_market.csv'), index=False)
trader_profile.to_csv(os.path.join(PROC, 'trader_profile.csv'), index=False)
print("\n✅ Processed CSVs saved to data/processed/")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PART B — ANALYSIS                                           ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n\n── B1  Performance: Fear vs Greed ────────────────────────────")

# Binary only (exclude Neutral for cleaner comparison)
dt_bin = daily_trader[daily_trader['sentiment_binary'].isin(['Fear', 'Greed'])]
dm_bin = daily_market[daily_market['sentiment_binary'].isin(['Fear', 'Greed'])]

perf = dt_bin.groupby('sentiment_binary').agg(
    median_pnl  = ('daily_pnl',   'median'),
    mean_pnl    = ('daily_pnl',   'mean'),
    mean_winrate= ('win_rate',    'mean'),
    pnl_std     = ('daily_pnl',   'std'),
    n_obs       = ('daily_pnl',   'count'),
).round(4)
print(perf.to_string())

# ── CHART 1 · PnL & Win Rate by Sentiment ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Chart 1 — Performance: Fear vs Greed Days', fontsize=14, fontweight='bold', y=1.01)

# 1a Median PnL bar
cats = ['Fear', 'Greed']
vals = [perf.loc[c, 'median_pnl'] for c in cats]
bars = axes[0].bar(cats, vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor='black', linewidth=0.8, width=0.5)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_title('Median Daily PnL per Trader')
axes[0].set_ylabel('PnL (USD)')
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width()/2, v + (0.5 if v >= 0 else -0.5),
                 f'${v:.2f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=10, fontweight='bold')

# 1b Win rate bar
wr_vals = [perf.loc[c, 'mean_winrate'] for c in cats]
bars2 = axes[1].bar(cats, wr_vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor='black', linewidth=0.8, width=0.5)
axes[1].axhline(0.5, color='navy', linestyle='--', linewidth=1, label='50% baseline')
axes[1].set_title('Average Win Rate')
axes[1].set_ylabel('Win Rate')
axes[1].set_ylim(0, 1)
axes[1].legend()
for b, v in zip(bars2, wr_vals):
    axes[1].text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.1%}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# 1c PnL distribution violin
dt_bin_clip = dt_bin.copy()
dt_bin_clip['daily_pnl_clipped'] = dt_bin_clip['daily_pnl'].clip(-3000, 3000)
sns.violinplot(data=dt_bin_clip, x='sentiment_binary', y='daily_pnl_clipped',
               palette=BINARY_PALETTE, ax=axes[2], inner='quartile', cut=0)
axes[2].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[2].set_title('PnL Distribution (clipped ±$3k)')
axes[2].set_ylabel('Daily PnL (USD)')
axes[2].set_xlabel('')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart1_pnl_winrate_sentiment.png'), bbox_inches='tight')
plt.close()
print("  → chart1_pnl_winrate_sentiment.png saved")


print("\n── B2  Trader Behavior by Sentiment ──────────────────────────")

behavior = dt_bin.groupby('sentiment_binary').agg(
    avg_trade_count   = ('trade_count',   'mean'),
    avg_size_usd      = ('avg_size_usd',  'mean'),
    avg_long_ratio    = ('long_ratio',    'mean'),
    avg_total_vol     = ('total_size_usd','mean'),
).round(3)
print(behavior.to_string())

# ── CHART 2 · Behavior by Sentiment ──────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle('Chart 2 — Trader Behavior: Fear vs Greed Days', fontsize=14, fontweight='bold', y=1.01)

metrics = [
    ('avg_trade_count',  'Avg Trades per Day',   ''),
    ('avg_size_usd',     'Avg Trade Size (USD)',  '$'),
    ('avg_long_ratio',   'Avg Long Ratio',        ''),
    ('avg_total_vol',    'Avg Daily Volume (USD)','$'),
]
for ax, (col, title, prefix) in zip(axes, metrics):
    vals = [behavior.loc[c, col] for c in cats]
    bars = ax.bar(cats, vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor='black', linewidth=0.8, width=0.5)
    ax.set_title(title)
    ax.set_xlabel('')
    for b, v in zip(bars, vals):
        label = f'{prefix}{v:,.1f}' if prefix == '$' else f'{v:.3f}'
        ax.text(b.get_x() + b.get_width()/2, v * 1.02, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart2_behavior_sentiment.png'), bbox_inches='tight')
plt.close()
print("  → chart2_behavior_sentiment.png saved")


print("\n── B3  Trader Segmentation ───────────────────────────────────")

# Merge segments into binary daily data
dt_seg = dt_bin.copy()

seg_perf = {}
for seg_col in ['freq_segment', 'size_segment', 'winner_segment']:
    grp = dt_seg.groupby([seg_col, 'sentiment_binary'])['daily_pnl'].median().unstack()
    seg_perf[seg_col] = grp
    print(f"\n  {seg_col}:")
    print(grp.round(2).to_string())

# ── CHART 3 · Segmentation ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Chart 3 — Segment PnL: Fear vs Greed', fontsize=14, fontweight='bold', y=1.01)

for ax, (seg_col, title) in zip(axes, [
    ('freq_segment',   'Frequent vs Infrequent Traders'),
    ('size_segment',   'Large vs Small Position Traders'),
    ('winner_segment', 'Consistent Winners vs Others'),
]):
    grp = seg_perf[seg_col]
    x   = np.arange(len(grp.index))
    w   = 0.35
    for i, (sentiment, color) in enumerate([('Fear', FEAR_COLOR), ('Greed', GREED_COLOR)]):
        if sentiment in grp.columns:
            bars = ax.bar(x + i*w - w/2, grp[sentiment], w, label=sentiment,
                          color=color, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=15, ha='right')
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.set_title(title, fontsize=11)
    ax.set_ylabel('Median Daily PnL (USD)')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart3_segments.png'), bbox_inches='tight')
plt.close()
print("  → chart3_segments.png saved")


# ── CHART 4 · Sentiment value over time vs Market PnL ────────────
print("\n── B4  Insight: Sentiment score vs market PnL over time ─────")

dm_sorted = daily_market.sort_values('date')

fig, ax1 = plt.subplots(figsize=(16, 5))
fig.suptitle('Chart 4 — Market PnL & Sentiment Score Over Time', fontsize=14, fontweight='bold')

color_bars = dm_sorted['sentiment_binary'].map({'Fear': FEAR_COLOR, 'Greed': GREED_COLOR, 'Neutral': NEUTRAL_COLOR})
ax1.bar(dm_sorted['date'], dm_sorted['total_pnl'], color=color_bars, alpha=0.6, width=0.8, label='Total PnL')
ax1.set_ylabel('Total Market PnL (USD)', color='black')
ax1.axhline(0, color='black', linewidth=0.8)

ax2 = ax1.twinx()
ax2.plot(dm_sorted['date'], dm_sorted['value'], color='royalblue', linewidth=1.5, label='Fear/Greed Score', alpha=0.85)
ax2.set_ylabel('Fear/Greed Index (0–100)', color='royalblue')
ax2.tick_params(axis='y', labelcolor='royalblue')
ax2.set_ylim(0, 100)

patches = [
    mpatches.Patch(color=FEAR_COLOR,  label='Fear days'),
    mpatches.Patch(color=GREED_COLOR, label='Greed days'),
    mpatches.Patch(color='royalblue', label='F/G Score'),
]
ax1.legend(handles=patches, loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart4_pnl_vs_score_time.png'), bbox_inches='tight')
plt.close()
print("  → chart4_pnl_vs_score_time.png saved")


# ── CHART 5 · Long ratio heatmap by sentiment × day-of-week ──────
print("\n── B5  Insight: Long bias & trade volume by sentiment ────────")

dm_sorted['dayofweek'] = pd.to_datetime(dm_sorted['date']).dt.day_name()
dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
hmap_data = dm_sorted.groupby(['sentiment_5', 'dayofweek'])['long_ratio'].mean().unstack()
hmap_data = hmap_data.reindex(columns=[d for d in dow_order if d in hmap_data.columns])
hmap_data = hmap_data.reindex(['Extreme Fear','Fear','Neutral','Greed','Extreme Greed'])

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Chart 5 — Long Ratio & Trade Volume by Sentiment', fontsize=14, fontweight='bold')

# 5a heatmap
sns.heatmap(hmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
            ax=axes[0], linewidths=0.4, cbar_kws={'label': 'Long Ratio'}, vmin=0.3, vmax=0.7)
axes[0].set_title('Avg Long Ratio: Sentiment × Day of Week')
axes[0].set_xlabel('')
axes[0].set_ylabel('')

# 5b Volume bars by sentiment_5
vol_by_sent = dm_sorted.groupby('sentiment_5')['total_vol_usd'].mean()
vol_by_sent = vol_by_sent.reindex(['Extreme Fear','Fear','Neutral','Greed','Extreme Greed'])
colors_5 = [PALETTE.get(s, '#999') for s in vol_by_sent.index]
vol_by_sent.plot(kind='bar', ax=axes[1], color=colors_5, edgecolor='black', linewidth=0.7)
axes[1].set_title('Avg Daily Trading Volume by Sentiment')
axes[1].set_ylabel('Avg Volume USD')
axes[1].tick_params(axis='x', rotation=20)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart5_long_ratio_volume.png'), bbox_inches='tight')
plt.close()
print("  → chart5_long_ratio_volume.png saved")


# ── CHART 6 · Win rate by sentiment × trader segment ─────────────
print("\n── B6  Insight: Win rate heatmap across segments ─────────────")

heat_wr = dt_bin.groupby(['winner_segment', 'sentiment_binary'])['win_rate'].mean().unstack()
heat_freq= dt_bin.groupby(['freq_segment',  'sentiment_binary'])['win_rate'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Chart 6 — Win Rate: Segment × Sentiment', fontsize=14, fontweight='bold')

for ax, data, title in [
    (axes[0], heat_wr,   'Consistent Winners vs Others'),
    (axes[1], heat_freq, 'Frequent vs Infrequent Traders'),
]:
    sns.heatmap(data, annot=True, fmt='.2%', cmap='RdYlGn', center=0.5,
                ax=ax, linewidths=0.5, cbar_kws={'label': 'Win Rate'}, vmin=0.3, vmax=0.7)
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart6_winrate_heatmap.png'), bbox_inches='tight')
plt.close()
print("  → chart6_winrate_heatmap.png saved")


# ── CHART 7 · Drawdown proxy: cumulative PnL by sentiment ─────────
print("\n── B7  Insight: Cumulative PnL & drawdown by sentiment ───────")

dm_fear  = daily_market[daily_market['sentiment_binary'] == 'Fear'].sort_values('date')
dm_greed = daily_market[daily_market['sentiment_binary'] == 'Greed'].sort_values('date')

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Chart 7 — Cumulative Market PnL on Fear vs Greed Days', fontsize=14, fontweight='bold')

for ax, dm, color, label in [
    (axes[0], dm_fear,  FEAR_COLOR,  'Fear Days'),
    (axes[1], dm_greed, GREED_COLOR, 'Greed Days'),
]:
    cum = dm['total_pnl'].cumsum()
    roll_max = cum.cummax()
    drawdown = cum - roll_max

    ax.fill_between(range(len(cum)), cum, alpha=0.3, color=color)
    ax.plot(cum.values, color=color, linewidth=2, label='Cumulative PnL')
    ax.fill_between(range(len(drawdown)), drawdown, alpha=0.4, color='grey', label='Drawdown')
    ax.set_title(f'{label} — Cumulative PnL')
    ax.set_xlabel('Trading Day Index')
    ax.set_ylabel('Cumulative PnL (USD)')
    ax.axhline(0, color='black', linewidth=0.7, linestyle='--')
    ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart7_cumulative_drawdown.png'), bbox_inches='tight')
plt.close()
print("  → chart7_cumulative_drawdown.png saved")


# ╔══════════════════════════════════════════════════════════════╗
# ║  BONUS — PREDICTIVE MODEL                                    ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n── BONUS  Predictive Model ────────────────────────────────────")

model_df = daily_trader.dropna(subset=['daily_pnl', 'trade_count', 'avg_size_usd', 'long_ratio', 'win_rate']).copy()
model_df['target']        = (model_df['daily_pnl'] > 0).astype(int)
model_df['sentiment_enc'] = (model_df['sentiment_binary'] == 'Greed').astype(int)
model_df['fg_score']      = model_df['value']

features = ['trade_count', 'avg_size_usd', 'long_ratio', 'total_size_usd', 'sentiment_enc', 'fg_score']
model_df  = model_df.dropna(subset=features)

X = model_df[features]
y = model_df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler  = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
clf.fit(X_train_s, y_train)
y_pred = clf.predict(X_test_s)
y_prob = clf.predict_proba(X_test_s)[:, 1]

roc = roc_auc_score(y_test, y_prob)
print(f"\nRandom Forest AUC-ROC: {roc:.4f}")
print(classification_report(y_test, y_pred, target_names=['Loss Day', 'Profit Day']))

# ── CHART 8 · Feature Importance ─────────────────────────────────
feat_labels = ['Trade Count', 'Avg Size USD', 'Long Ratio', 'Total Volume', 'Sentiment (Greed=1)', 'F/G Score']
importances = pd.Series(clf.feature_importances_, index=feat_labels).sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle('Chart 8 — Feature Importance: Predicting Trader Profitability', fontsize=14, fontweight='bold')

colors = ['#3498db' if i < len(importances)-2 else '#e67e22' for i in range(len(importances))]
importances.plot(kind='barh', ax=ax, color=colors[::-1], edgecolor='black', linewidth=0.6)
ax.set_xlabel('Feature Importance (Mean Decrease Impurity)')
ax.set_title(f'Random Forest  |  AUC-ROC: {roc:.3f}  |  n_estimators=200')

for i, (v, name) in enumerate(zip(importances.values, importances.index)):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart8_feature_importance.png'), bbox_inches='tight')
plt.close()
print("  → chart8_feature_importance.png saved")


# ╔══════════════════════════════════════════════════════════════╗
# ║  SUMMARY TABLES                                              ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n── Summary Tables ─────────────────────────────────────────────")

summary_perf = dt_bin.groupby('sentiment_binary').agg(
    observations     = ('daily_pnl',    'count'),
    median_pnl       = ('daily_pnl',    'median'),
    mean_pnl         = ('daily_pnl',    'mean'),
    pnl_std          = ('daily_pnl',    'std'),
    mean_win_rate    = ('win_rate',     'mean'),
    mean_trade_count = ('trade_count',  'mean'),
    mean_size_usd    = ('avg_size_usd', 'mean'),
    mean_long_ratio  = ('long_ratio',   'mean'),
    mean_volume      = ('total_size_usd','mean'),
).round(3)

summary_5class = daily_market.groupby('sentiment_5').agg(
    days             = ('total_pnl',     'count'),
    median_pnl       = ('total_pnl',     'median'),
    mean_pnl         = ('total_pnl',     'mean'),
    mean_win_rate    = ('win_rate',      'mean'),
    mean_long_ratio  = ('long_ratio',    'mean'),
    mean_trade_count = ('trade_count',   'mean'),
    mean_volume      = ('total_vol_usd', 'mean'),
).round(2)
order_5 = ['Extreme Fear','Fear','Neutral','Greed','Extreme Greed']
summary_5class = summary_5class.reindex([r for r in order_5 if r in summary_5class.index])

print("\nBinary Summary (Fear vs Greed):")
print(summary_perf.to_string())
print("\n5-Class Summary:")
print(summary_5class.to_string())

summary_perf.to_csv(os.path.join(OUT, 'summary_binary.csv'))
summary_5class.to_csv(os.path.join(OUT, 'summary_5class.csv'))
print("\n✅ Summary tables saved to outputs/")


print("\n\n" + "=" * 65)
print("  ALL DONE — 8 charts + 2 tables + 3 processed CSVs generated")
print("=" * 65)
