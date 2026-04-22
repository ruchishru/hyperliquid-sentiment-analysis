"""
Clustering traders into 4 behavioral archetypes using KMeans + PCA.
k=4 forced for analytically meaningful archetypes (32 traders).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings, os

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid')
plt.rcParams.update({'figure.dpi': 140, 'axes.titlesize': 12, 'axes.labelsize': 11})

BASE   = os.path.dirname(os.path.abspath(__file__))
PROC   = os.path.join(BASE, 'data', 'processed')
CHARTS = os.path.join(BASE, 'charts')

# ── Load ─────────────────────────────────────────────────────────
tp = pd.read_csv(os.path.join(PROC, 'trader_profile.csv'))
dt = pd.read_csv(os.path.join(PROC, 'daily_trader.csv'))
print(f"Traders: {len(tp)}")

# ── Per-sentiment features ────────────────────────────────────────
for sent, prefix in [('Fear','fear'), ('Greed','greed')]:
    stats = dt[dt['sentiment_binary']==sent].groupby('Account').agg(
        **{f'{prefix}_pnl':        ('daily_pnl',    'mean'),
           f'{prefix}_trades':     ('trade_count',  'mean'),
           f'{prefix}_win_rate':   ('win_rate',     'mean'),
           f'{prefix}_long_ratio': ('long_ratio',   'mean'),
           f'{prefix}_size':       ('avg_size_usd', 'mean')}
    ).reset_index()
    tp = tp.merge(stats, on='Account', how='left')
tp = tp.fillna(0)

tp['pnl_greed_lift']  = tp['greed_pnl'] - tp['fear_pnl']
tp['direction_shift'] = tp['fear_long_ratio'] - tp['greed_long_ratio']
tp['wr_greed_lift']   = tp['greed_win_rate'] - tp['fear_win_rate']

FEATURES = ['avg_daily_pnl','avg_size_usd','avg_win_rate','avg_long_ratio',
            'total_trades','sharpe','pnl_greed_lift','direction_shift','wr_greed_lift']
X = tp[FEATURES].copy()
for col in X.columns:
    lo, hi = X[col].quantile(0.02), X[col].quantile(0.98)
    X[col] = X[col].clip(lo, hi)
X_scaled = StandardScaler().fit_transform(X)

# ── Elbow + Silhouette ───────────────────────────────────────────
sil_scores, inertias = {}, {}
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)
    inertias[k]   = km.inertia_
CHOSEN_K = 4
print(f"Silhouette scores: {sil_scores}")
print(f"Using k={CHOSEN_K}")

# ── Final clustering ─────────────────────────────────────────────
tp['cluster'] = KMeans(n_clusters=CHOSEN_K, random_state=42, n_init=50).fit_predict(X_scaled)

# ── PCA ──────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)
tp['pca1'], tp['pca2'] = coords[:,0], coords[:,1]
ev = pca.explained_variance_ratio_
print(f"PCA: PC1={ev[0]:.1%}  PC2={ev[1]:.1%}")

# ── Cluster stats ─────────────────────────────────────────────────
cluster_stats = tp.groupby('cluster').agg(
    n              = ('Account',         'count'),
    avg_pnl        = ('avg_daily_pnl',   'mean'),
    avg_size       = ('avg_size_usd',    'mean'),
    avg_win_rate   = ('avg_win_rate',    'mean'),
    avg_long_ratio = ('avg_long_ratio',  'mean'),
    avg_trades     = ('total_trades',    'mean'),
    avg_sharpe     = ('sharpe',          'mean'),
    fear_pnl       = ('fear_pnl',        'mean'),
    greed_pnl      = ('greed_pnl',       'mean'),
    pnl_greed_lift = ('pnl_greed_lift',  'mean'),
    dir_shift      = ('direction_shift', 'mean'),
).round(2)

# ── Hand-craft archetype names from cluster inspection ───────────
# Cluster 0 (n=1):  dominant Fear earner, massive size     -> Fear Dominator
# Cluster 1 (n=9):  large positions, inconsistent, flat    -> High-Stakes Speculator
# Cluster 2 (n=12): highest win-rate, active, best Sharpe  -> Disciplined Grinder
# Cluster 3 (n=10): almost all PnL on Greed, low win-rate  -> Greed Momentum Rider

ARCHETYPE_MAP = {
    0: "Fear Dominator",
    1: "High-Stakes Speculator",
    2: "Disciplined Grinder",
    3: "Greed Momentum Rider",
}
ARCHETYPE_DEFS = {
    "Fear Dominator": {
        "desc": "Earns the most during Fear days. Dominant player with large positions. Acts as a market stabiliser.",
        "color": "#8e44ad", "emoji": "👑"
    },
    "High-Stakes Speculator": {
        "desc": "Large position sizes with volatile, inconsistent PnL. Performs erratically across both sentiment regimes.",
        "color": "#e67e22", "emoji": "🎲"
    },
    "Disciplined Grinder": {
        "desc": "Highest win-rate and best risk-adjusted returns (Sharpe). High trade volume. Consistent across regimes.",
        "color": "#27ae60", "emoji": "⚙️"
    },
    "Greed Momentum Rider": {
        "desc": "Earns almost exclusively on Greed days. Low win-rate but large wins when sentiment is positive.",
        "color": "#e74c3c", "emoji": "📈"
    },
}
ACOLORS = {a: ARCHETYPE_DEFS[a]['color'] for a in ARCHETYPE_DEFS}

tp['archetype'] = tp['cluster'].map(ARCHETYPE_MAP)
cluster_stats['archetype'] = cluster_stats.index.map(ARCHETYPE_MAP)

print("\nArchetype summary:")
print(f"{'Cluster':>8}  {'Archetype':<26}  {'N':>3}  {'AvgPnL':>9}  {'WinRate':>8}  {'Sharpe':>7}  {'FearPnL':>9}  {'GreedPnL':>9}")
print("-"*90)
for c, row in cluster_stats.iterrows():
    print(f"{c:>8}  {ARCHETYPE_MAP[c]:<26}  {int(row['n']):>3}  {row['avg_pnl']:>9.0f}  {row['avg_win_rate']:>8.1%}  {row['avg_sharpe']:>7.3f}  {row['fear_pnl']:>9.0f}  {row['greed_pnl']:>9.0f}")

# ═══════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════

# ── CHART 9 · Elbow + Silhouette ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle('Chart 9 — Optimal Number of Clusters', fontsize=13, fontweight='bold')

ks = list(inertias.keys())
axes[0].plot(ks, list(inertias.values()), 'o-', color='steelblue', linewidth=2.2, markersize=7)
axes[0].axvline(CHOSEN_K, color='#e74c3c', linestyle='--', linewidth=1.5, label=f'Chosen k={CHOSEN_K}')
axes[0].set_title('Elbow Method (Inertia)'); axes[0].set_xlabel('k'); axes[0].set_ylabel('Inertia'); axes[0].legend()

bars = axes[1].bar(list(sil_scores.keys()), list(sil_scores.values()),
                   color=['#e74c3c' if k==CHOSEN_K else '#95a5a6' for k in sil_scores],
                   edgecolor='black', linewidth=0.6)
axes[1].set_title('Silhouette Score by k'); axes[1].set_xlabel('k'); axes[1].set_ylabel('Score')
for k, v in sil_scores.items():
    axes[1].text(k, v+0.002, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart9_cluster_optimal_k.png'), bbox_inches='tight')
plt.close()
print("\n  -> chart9 saved")

# ── CHART 10 · PCA scatter ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))
fig.suptitle('Chart 10 — Trader Archetypes in PCA Space\n(k-Means Clustering, k=4)', fontsize=13, fontweight='bold')

for arch in tp['archetype'].unique():
    sub   = tp[tp['archetype']==arch]
    color = ACOLORS[arch]
    emoji = ARCHETYPE_DEFS[arch]['emoji']
    ax.scatter(sub['pca1'], sub['pca2'],
               label=f"{emoji} {arch} (n={len(sub)})",
               color=color, s=260, edgecolors='white', linewidths=1.8, alpha=0.92, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(row['Account'][:6]+'…', (row['pca1'], row['pca2']),
                    textcoords='offset points', xytext=(7, 4), fontsize=7.5, color='#444')

ax.set_xlabel(f'PC1 ({ev[0]:.1%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({ev[1]:.1%} variance)', fontsize=11)
ax.axhline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.6)
ax.axvline(0, color='grey', linewidth=0.5, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart10_pca_archetypes.png'), bbox_inches='tight')
plt.close()
print("  -> chart10 saved")

# ── CHART 11 · Radar profiles ────────────────────────────────────
radar_cols   = ['avg_pnl','avg_size','avg_win_rate','avg_trades','avg_sharpe','greed_pnl','fear_pnl']
radar_labels = ['Avg PnL','Pos. Size','Win Rate','Trade Vol','Sharpe','Greed PnL','Fear PnL']
N = len(radar_cols)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]

radar_df = cluster_stats[radar_cols].copy()
for col in radar_cols:
    lo, hi = radar_df[col].min(), radar_df[col].max()
    radar_df[col] = (radar_df[col]-lo)/(hi-lo+1e-9)

fig, axes = plt.subplots(1, CHOSEN_K, figsize=(5.5*CHOSEN_K, 6), subplot_kw=dict(polar=True))
fig.suptitle('Chart 11 — Archetype Behavioural Radar Profiles', fontsize=14, fontweight='bold', y=1.01)

for ax, (cid, row) in zip(axes, radar_df.iterrows()):
    arch  = ARCHETYPE_MAP[cid]
    color = ACOLORS[arch]
    vals  = row.tolist() + [row.tolist()[0]]
    ax.plot(angles, vals, color=color, linewidth=2.5)
    ax.fill(angles, vals, alpha=0.2, color=color)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(radar_labels, size=9)
    ax.set_yticklabels([]); ax.set_ylim(0, 1); ax.grid(True, alpha=0.3)
    n = int(cluster_stats.loc[cid,'n'])
    ax.set_title(f"{ARCHETYPE_DEFS[arch]['emoji']} {arch}\nn={n}", size=10, pad=18, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart11_archetype_radar.png'), bbox_inches='tight')
plt.close()
print("  -> chart11 saved")

# ── CHART 12 · Archetype PnL + Win Rate by Sentiment ────────────
dt2 = dt.merge(tp[['Account','archetype']], on='Account', how='left')
dt2_bin = dt2[dt2['sentiment_binary'].isin(['Fear','Greed'])]

arch_pnl = dt2_bin.groupby(['archetype','sentiment_binary'])['daily_pnl'].median().unstack()
arch_wr  = dt2_bin.groupby(['archetype','sentiment_binary'])['win_rate'].mean().unstack()

fig, axes = plt.subplots(1, 2, figsize=(17, 6))
fig.suptitle('Chart 12 — Archetype Performance: Fear vs Greed', fontsize=14, fontweight='bold')

archs = arch_pnl.index.tolist()
x = np.arange(len(archs)); w = 0.35
for i, (sent, color) in enumerate([('Fear','#e74c3c'),('Greed','#27ae60')]):
    if sent in arch_pnl.columns:
        vals = arch_pnl[sent].values
        bars = axes[0].bar(x+i*w-w/2, vals, w, label=sent, color=color, edgecolor='black', linewidth=0.6)
        for b, v in zip(bars, vals):
            axes[0].text(b.get_x()+b.get_width()/2, v+(3 if v>=0 else -3),
                         f'${v:,.0f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=8.5, fontweight='bold')

arch_labels = [f"{ARCHETYPE_DEFS[a]['emoji']} {a}" for a in archs]
axes[0].set_xticks(x); axes[0].set_xticklabels(arch_labels, rotation=12, ha='right', fontsize=9)
axes[0].axhline(0, color='black', linewidth=0.8, linestyle='--')
axes[0].set_ylabel('Median Daily PnL (USD)'); axes[0].legend()
axes[0].set_title('Median Daily PnL by Archetype & Sentiment')

# Win rate heatmap
arch_wr_display = arch_wr.copy()
arch_wr_display.index = [f"{ARCHETYPE_DEFS[a]['emoji']} {a}" for a in arch_wr.index]
sns.heatmap(arch_wr_display, annot=True, fmt='.1%', cmap='RdYlGn', center=0.35,
            ax=axes[1], linewidths=0.6, cbar_kws={'label':'Win Rate'}, vmin=0.1, vmax=0.65)
axes[1].set_title('Avg Win Rate: Archetype × Sentiment')
axes[1].set_xlabel(''); axes[1].set_ylabel(''); axes[1].tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHARTS, 'chart12_archetype_sentiment_pnl.png'), bbox_inches='tight')
plt.close()
print("  -> chart12 saved")

# ── Save ─────────────────────────────────────────────────────────
tp.to_csv(os.path.join(PROC, 'trader_profile_clustered.csv'), index=False)
cluster_stats.to_csv(os.path.join(PROC, 'cluster_summary.csv'))
print("\nAll clustering outputs saved.")
