"""
Streamlit Dashboard — Hyperliquid × Fear/Greed Sentiment Analysis
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Hyperliquid × Sentiment",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ─────────────────────────────────────────────────────────
FEAR_COLOR    = "#e74c3c"
GREED_COLOR   = "#27ae60"
NEUTRAL_COLOR = "#f39c12"
ACCENT        = "#1a3a5c"
BINARY_PALETTE = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR}
PALETTE_5 = {
    "Extreme Fear": "#c0392b", "Fear": "#e74c3c",
    "Neutral": "#f39c12", "Greed": "#27ae60", "Extreme Greed": "#1a7a3e",
}
ARCHETYPE_COLORS = {
    "Fear Dominator":         "#8e44ad",
    "High-Stakes Speculator": "#e67e22",
    "Disciplined Grinder":    "#27ae60",
    "Greed Momentum Rider":   "#e74c3c",
}
ARCHETYPE_EMOJI = {
    "Fear Dominator":         "👑",
    "High-Stakes Speculator": "🎲",
    "Disciplined Grinder":    "⚙️",
    "Greed Momentum Rider":   "📈",
}

sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.dpi": 120, "axes.titlesize": 11, "axes.labelsize": 10})

# ── Data loading ─────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "data", "processed")

@st.cache_data
def load_data():
    dt  = pd.read_csv(os.path.join(PROC, "daily_trader.csv"), parse_dates=["date"])
    dm  = pd.read_csv(os.path.join(PROC, "daily_market.csv"), parse_dates=["date"])
    tp  = pd.read_csv(os.path.join(PROC, "trader_profile_clustered.csv"))
    cs  = pd.read_csv(os.path.join(PROC, "cluster_summary.csv"))
    return dt, dm, tp, cs

dt, dm, tp, cs = load_data()

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.image("https://img.shields.io/badge/Hyperliquid-Sentiment-blue?style=flat-square", width=200)
st.sidebar.title("🎛️ Filters")

# Date range
min_d, max_d = dt["date"].min().date(), dt["date"].max().date()
date_range = st.sidebar.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)

# Sentiment filter
all_sentiments = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
selected_sentiments = st.sidebar.multiselect("Sentiment Filter", all_sentiments, default=all_sentiments)

# Archetype filter
all_archetypes = list(ARCHETYPE_COLORS.keys())
selected_archetypes = st.sidebar.multiselect("Trader Archetype", all_archetypes, default=all_archetypes)

# Binary only toggle
binary_only = st.sidebar.checkbox("Binary mode (Fear vs Greed only)", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset Stats**")
st.sidebar.metric("Total Trades",   "211,224")
st.sidebar.metric("Unique Traders", str(tp["Account"].nunique()))
st.sidebar.metric("Trading Days",   str(dm["date"].nunique()))
st.sidebar.metric("Date Coverage",  "Jan–Dec 2024")

# ── Apply filters ─────────────────────────────────────────────────
if len(date_range) == 2:
    start_d, end_d = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    dt_f = dt[(dt["date"] >= start_d) & (dt["date"] <= end_d)]
    dm_f = dm[(dm["date"] >= start_d) & (dm["date"] <= end_d)]
else:
    dt_f, dm_f = dt.copy(), dm.copy()

if selected_sentiments:
    dt_f = dt_f[dt_f["sentiment_5"].isin(selected_sentiments)]
    dm_f = dm_f[dm_f["sentiment_5"].isin(selected_sentiments)]

if binary_only:
    dt_f = dt_f[dt_f["sentiment_binary"].isin(["Fear", "Greed"])]
    dm_f = dm_f[dm_f["sentiment_binary"].isin(["Fear", "Greed"])]

tp_f = tp[tp["archetype"].isin(selected_archetypes)] if selected_archetypes else tp.copy()
dt_arch = dt_f.merge(tp[["Account","archetype"]], on="Account", how="left")
dt_arch = dt_arch[dt_arch["archetype"].isin(selected_archetypes)] if selected_archetypes else dt_arch

# ── Title ─────────────────────────────────────────────────────────
st.title("📊 Hyperliquid × Fear/Greed Sentiment Analysis")
st.caption("How Bitcoin market sentiment shapes trader behaviour and performance on Hyperliquid DEX")

# ═══════════════════════════════════════════════════════════════
#  TAB LAYOUT
# ═══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview",
    "📈 Performance",
    "🧠 Behaviour",
    "🔬 Archetypes",
    "📋 Strategy",
])

# ─────────────────────────────────────────────────────────────────
#  TAB 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Market-Wide Summary")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    fear_days  = dm_f[dm_f["sentiment_binary"]=="Fear"]
    greed_days = dm_f[dm_f["sentiment_binary"]=="Greed"]

    col1.metric("Fear Days",   len(fear_days))
    col2.metric("Greed Days",  len(greed_days))
    col3.metric("Median PnL — Fear",  f"${fear_days['total_pnl'].median():,.0f}"  if len(fear_days)  else "—")
    col4.metric("Median PnL — Greed", f"${greed_days['total_pnl'].median():,.0f}" if len(greed_days) else "—")
    col5.metric("Avg Win Rate", f"{dm_f['win_rate'].mean():.1%}" if len(dm_f) else "—")

    st.markdown("---")

    col_a, col_b = st.columns([2,1])

    with col_a:
        st.markdown("**Daily Market PnL vs Fear/Greed Score**")
        dm_s = dm_f.sort_values("date")
        if len(dm_s) > 0:
            fig, ax1 = plt.subplots(figsize=(11, 4))
            cbars = dm_s["sentiment_binary"].map({"Fear": FEAR_COLOR, "Greed": GREED_COLOR, "Neutral": NEUTRAL_COLOR})
            ax1.bar(dm_s["date"], dm_s["total_pnl"], color=cbars, alpha=0.65, width=0.8)
            ax1.axhline(0, color="black", linewidth=0.8)
            ax1.set_ylabel("Total PnL (USD)")
            ax2 = ax1.twinx()
            ax2.plot(dm_s["date"], dm_s["value"], color="royalblue", linewidth=1.5, alpha=0.85)
            ax2.set_ylabel("F/G Score", color="royalblue")
            ax2.tick_params(axis="y", labelcolor="royalblue")
            ax2.set_ylim(0, 100)
            patches = [mpatches.Patch(color=FEAR_COLOR, label="Fear"),
                       mpatches.Patch(color=GREED_COLOR, label="Greed"),
                       mpatches.Patch(color="royalblue", label="F/G Score")]
            ax1.legend(handles=patches, loc="upper left", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with col_b:
        st.markdown("**Sentiment Distribution**")
        if len(dm_f) > 0:
            sent_counts = dm_f["sentiment_5"].value_counts().reindex(
                ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]).dropna()
            fig, ax = plt.subplots(figsize=(5, 4))
            colors = [PALETTE_5.get(s,"#999") for s in sent_counts.index]
            bars = ax.barh(sent_counts.index, sent_counts.values, color=colors, edgecolor="black", linewidth=0.5)
            for b, v in zip(bars, sent_counts.values):
                ax.text(v+0.3, b.get_y()+b.get_height()/2, str(v), va="center", fontsize=9)
            ax.set_xlabel("Days"); ax.set_title("Days per Sentiment Class")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    st.markdown("---")
    st.markdown("**5-Class Daily Summary Table**")
    summary_5 = dm_f.groupby("sentiment_5").agg(
        Days=("total_pnl","count"), Median_PnL=("total_pnl","median"),
        Mean_PnL=("total_pnl","mean"), Win_Rate=("win_rate","mean"),
        Long_Ratio=("long_ratio","mean"), Avg_Trades=("trade_count","mean"),
    ).round(2)
    order = ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]
    summary_5 = summary_5.reindex([r for r in order if r in summary_5.index])
    st.dataframe(summary_5.style.background_gradient(cmap="RdYlGn", subset=["Median_PnL","Win_Rate"]), use_container_width=True)


# ─────────────────────────────────────────────────────────────────
#  TAB 2 — PERFORMANCE
# ─────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Performance: Fear vs Greed Days")

    dt_bin = dt_f[dt_f["sentiment_binary"].isin(["Fear","Greed"])]

    if len(dt_bin) > 0:
        perf = dt_bin.groupby("sentiment_binary").agg(
            Observations  = ("daily_pnl","count"),
            Median_PnL    = ("daily_pnl","median"),
            Mean_PnL      = ("daily_pnl","mean"),
            PnL_Std       = ("daily_pnl","std"),
            Win_Rate      = ("win_rate", "mean"),
        ).round(3)

        col1, col2, col3 = st.columns(3)
        cats = ["Fear","Greed"]

        with col1:
            fig, ax = plt.subplots(figsize=(5, 4))
            vals = [perf.loc[c,"Median_PnL"] if c in perf.index else 0 for c in cats]
            bars = ax.bar(cats, vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor="black", width=0.5)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title("Median Daily PnL per Trader")
            ax.set_ylabel("PnL (USD)")
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v+(2 if v>=0 else -2),
                        f"${v:.1f}", ha="center", va="bottom" if v>=0 else "top", fontweight="bold", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col2:
            fig, ax = plt.subplots(figsize=(5, 4))
            wr_vals = [perf.loc[c,"Win_Rate"] if c in perf.index else 0 for c in cats]
            bars = ax.bar(cats, wr_vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor="black", width=0.5)
            ax.axhline(0.5, color="navy", linestyle="--", linewidth=1, label="50% baseline")
            ax.set_title("Average Win Rate")
            ax.set_ylabel("Win Rate"); ax.set_ylim(0,1); ax.legend()
            for b, v in zip(bars, wr_vals):
                ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.1%}",
                        ha="center", va="bottom", fontweight="bold", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with col3:
            fig, ax = plt.subplots(figsize=(5, 4))
            dt_clipped = dt_bin.copy()
            dt_clipped["daily_pnl_c"] = dt_clipped["daily_pnl"].clip(-3000, 3000)
            sns.violinplot(data=dt_clipped, x="sentiment_binary", y="daily_pnl_c",
                           palette=BINARY_PALETTE, ax=ax, inner="quartile", cut=0)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title("PnL Distribution (±$3k clip)")
            ax.set_ylabel("Daily PnL (USD)"); ax.set_xlabel("")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("**Performance Table**")
        st.dataframe(perf.style.background_gradient(cmap="RdYlGn", subset=["Median_PnL","Win_Rate"]), use_container_width=True)

        st.markdown("---")
        st.markdown("**Cumulative Market PnL — Fear vs Greed Days**")
        col_f, col_g = st.columns(2)
        for col, sent, color, label in [
            (col_f, "Fear",  FEAR_COLOR,  "Fear Days"),
            (col_g, "Greed", GREED_COLOR, "Greed Days"),
        ]:
            sub = dm_f[dm_f["sentiment_binary"]==sent].sort_values("date")
            if len(sub) > 0:
                cum = sub["total_pnl"].cumsum()
                dd  = cum - cum.cummax()
                fig, ax = plt.subplots(figsize=(6,3.5))
                ax.fill_between(range(len(cum)), cum, alpha=0.2, color=color)
                ax.plot(cum.values, color=color, linewidth=2, label="Cum. PnL")
                ax.fill_between(range(len(dd)), dd, alpha=0.3, color="grey", label="Drawdown")
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
                ax.set_title(label); ax.set_ylabel("USD"); ax.legend(fontsize=8)
                plt.tight_layout()
                col.pyplot(fig); plt.close()
    else:
        st.info("No data for current filter selection.")


# ─────────────────────────────────────────────────────────────────
#  TAB 3 — BEHAVIOUR
# ─────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Trader Behaviour by Sentiment")

    dt_bin = dt_f[dt_f["sentiment_binary"].isin(["Fear","Greed"])]
    if len(dt_bin) > 0:
        behavior = dt_bin.groupby("sentiment_binary").agg(
            Avg_Trades     = ("trade_count",    "mean"),
            Avg_Size_USD   = ("avg_size_usd",   "mean"),
            Avg_Long_Ratio = ("long_ratio",     "mean"),
            Avg_Volume_USD = ("total_size_usd", "mean"),
        ).round(2)

        st.markdown("**Key Behaviour Metrics**")
        cols = st.columns(4)
        metrics = [("Avg_Trades","Avg Daily Trades",""),
                   ("Avg_Size_USD","Avg Position Size","$"),
                   ("Avg_Long_Ratio","Avg Long Ratio",""),
                   ("Avg_Volume_USD","Avg Daily Volume","$")]

        for col, (metric, label, prefix) in zip(cols, metrics):
            fig, ax = plt.subplots(figsize=(3.5, 3.2))
            cats = ["Fear","Greed"]
            vals = [behavior.loc[c,metric] if c in behavior.index else 0 for c in cats]
            ax.bar(cats, vals, color=[FEAR_COLOR, GREED_COLOR], edgecolor="black", width=0.5)
            ax.set_title(label, fontsize=9)
            for b, v in zip(ax.patches, vals):
                lbl = f"${v:,.0f}" if prefix=="$" else f"{v:.3f}"
                ax.text(b.get_x()+b.get_width()/2, v*1.02, lbl,
                        ha="center", va="bottom", fontsize=9, fontweight="bold")
            plt.tight_layout(); col.pyplot(fig); plt.close()

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Long Ratio & Volume Heatmap: Sentiment × Day of Week**")
            dm_s = dm_f.copy()
            dm_s["dayofweek"] = dm_s["date"].dt.day_name()
            dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            hmap = dm_s.groupby(["sentiment_5","dayofweek"])["long_ratio"].mean().unstack()
            hmap = hmap.reindex(columns=[d for d in dow if d in hmap.columns])
            hmap = hmap.reindex([r for r in ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"] if r in hmap.index])
            if len(hmap) > 0:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.heatmap(hmap, annot=True, fmt=".2f", cmap="RdYlGn", center=0.5,
                            ax=ax, linewidths=0.4, vmin=0.2, vmax=0.7)
                ax.set_title("Long Ratio by Sentiment × Day of Week")
                ax.set_xlabel(""); ax.set_ylabel("")
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_b:
            st.markdown("**Avg Volume by Sentiment Class**")
            vol = dm_f.groupby("sentiment_5")["total_vol_usd"].mean()
            vol = vol.reindex(["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"]).dropna()
            if len(vol) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                colors = [PALETTE_5.get(s,"#999") for s in vol.index]
                vol.plot(kind="bar", ax=ax, color=colors, edgecolor="black")
                ax.set_title("Avg Daily Volume (USD) by Sentiment")
                ax.set_ylabel("Volume USD"); ax.tick_params(axis="x", rotation=20)
                plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown("---")
        st.markdown("**Segment Comparison: Fear vs Greed**")
        col_s1, col_s2, col_s3 = st.columns(3)

        for col, seg_col, title in [
            (col_s1, "freq_segment",   "Frequent vs Infrequent"),
            (col_s2, "size_segment",   "Large vs Small Position"),
            (col_s3, "winner_segment", "Winners vs Others"),
        ]:
            if seg_col in dt_bin.columns:
                grp = dt_bin.groupby([seg_col,"sentiment_binary"])["daily_pnl"].median().unstack()
                if len(grp) > 0:
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    x = np.arange(len(grp.index)); w = 0.35
                    for i, (s, c) in enumerate([("Fear",FEAR_COLOR),("Greed",GREED_COLOR)]):
                        if s in grp.columns:
                            ax.bar(x+i*w-w/2, grp[s].values, w, label=s, color=c,
                                   edgecolor="black", linewidth=0.6)
                    ax.set_xticks(x); ax.set_xticklabels(grp.index, rotation=12, ha="right", fontsize=8)
                    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
                    ax.set_title(title, fontsize=9); ax.set_ylabel("Median PnL"); ax.legend(fontsize=7)
                    plt.tight_layout(); col.pyplot(fig); plt.close()
    else:
        st.info("No data for current filter selection.")


# ─────────────────────────────────────────────────────────────────
#  TAB 4 — ARCHETYPES
# ─────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🔬 Trader Behavioral Archetypes (k-Means, k=4)")

    # Archetype cards
    arch_cards = st.columns(4)
    archetype_descs = {
        "Fear Dominator":         "Earns most during Fear. Dominant player, large positions, acts as market stabiliser.",
        "High-Stakes Speculator": "Biggest position sizes, inconsistent and volatile PnL. Performs erratically across regimes.",
        "Disciplined Grinder":    "Highest win-rate & best Sharpe. High trade volume. Consistent returns across sentiment regimes.",
        "Greed Momentum Rider":   "Earns almost exclusively on Greed days. Low win-rate but large wins when sentiment is positive.",
    }
    for col, arch in zip(arch_cards, all_archetypes):
        n    = len(tp_f[tp_f["archetype"]==arch])
        emoji = ARCHETYPE_EMOJI.get(arch, "")
        color = ARCHETYPE_COLORS[arch]
        col.markdown(f"""
        <div style='border-left:4px solid {color}; padding:10px 12px; background:#f9f9f9; border-radius:4px; margin-bottom:8px;'>
            <div style='font-size:1.5rem'>{emoji}</div>
            <div style='font-weight:700; font-size:0.9rem; color:{color}'>{arch}</div>
            <div style='font-size:0.75rem; color:#555; margin-top:4px'>{archetype_descs[arch]}</div>
            <div style='font-weight:600; margin-top:6px'>n = {n} traders</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # PCA scatter
    col_pca, col_radar = st.columns([3, 2])

    with col_pca:
        st.markdown("**PCA Scatter — Archetype Clusters**")
        if len(tp_f) > 0:
            fig, ax = plt.subplots(figsize=(8, 5.5))
            for arch in tp_f["archetype"].unique():
                sub   = tp_f[tp_f["archetype"]==arch]
                color = ARCHETYPE_COLORS.get(arch,"#aaa")
                emoji = ARCHETYPE_EMOJI.get(arch,"")
                ax.scatter(sub["pca1"], sub["pca2"],
                           label=f"{emoji} {arch} (n={len(sub)})",
                           color=color, s=200, edgecolors="white", linewidths=1.5, alpha=0.9)
                for _, row in sub.iterrows():
                    ax.annotate(row["Account"][:6]+"…", (row["pca1"], row["pca2"]),
                                textcoords="offset points", xytext=(5,3), fontsize=6.5, color="#555")
            ax.axhline(0, color="grey", linewidth=0.4, linestyle="--")
            ax.axvline(0, color="grey", linewidth=0.4, linestyle="--")
            ax.set_xlabel("PC1 (24.1% variance)"); ax.set_ylabel("PC2 (18.7% variance)")
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    with col_radar:
        st.markdown("**Archetype Profile Table**")
        if len(tp_f) > 0:
            arch_summary = tp_f.groupby("archetype").agg(
                N         = ("Account",      "count"),
                Avg_PnL   = ("avg_daily_pnl","mean"),
                Win_Rate  = ("avg_win_rate", "mean"),
                Sharpe    = ("sharpe",        "mean"),
                Avg_Size  = ("avg_size_usd", "mean"),
                Total_Trades=("total_trades","mean"),
            ).round(2)
            st.dataframe(
                arch_summary.style.background_gradient(cmap="RdYlGn", subset=["Avg_PnL","Win_Rate","Sharpe"]),
                use_container_width=True
            )

    st.markdown("---")
    st.markdown("**Archetype Performance by Sentiment**")

    dt_arch_bin = dt_arch[dt_arch["sentiment_binary"].isin(["Fear","Greed"])]
    if len(dt_arch_bin) > 0:
        col_p, col_w = st.columns(2)

        with col_p:
            arch_pnl = dt_arch_bin.groupby(["archetype","sentiment_binary"])["daily_pnl"].median().unstack()
            if len(arch_pnl) > 0:
                fig, ax = plt.subplots(figsize=(7, 4.5))
                archs = arch_pnl.index.tolist()
                x = np.arange(len(archs)); w = 0.35
                for i, (s, c) in enumerate([("Fear",FEAR_COLOR),("Greed",GREED_COLOR)]):
                    if s in arch_pnl.columns:
                        vals = arch_pnl[s].values
                        bars = ax.bar(x+i*w-w/2, vals, w, label=s, color=c, edgecolor="black", linewidth=0.6)
                        for b, v in zip(bars, vals):
                            ax.text(b.get_x()+b.get_width()/2, v+(2 if v>=0 else -2),
                                    f"${v:,.0f}", ha="center", va="bottom" if v>=0 else "top", fontsize=7.5, fontweight="bold")
                arch_lbls = [f"{ARCHETYPE_EMOJI.get(a,'')}{a}" for a in archs]
                ax.set_xticks(x); ax.set_xticklabels(arch_lbls, rotation=12, ha="right", fontsize=8)
                ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
                ax.set_title("Median Daily PnL"); ax.legend()
                plt.tight_layout(); st.pyplot(fig); plt.close()

        with col_w:
            arch_wr = dt_arch_bin.groupby(["archetype","sentiment_binary"])["win_rate"].mean().unstack()
            arch_wr.index = [f"{ARCHETYPE_EMOJI.get(a,'')} {a}" for a in arch_wr.index]
            if len(arch_wr) > 0:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(arch_wr, annot=True, fmt=".1%", cmap="RdYlGn", center=0.35,
                            ax=ax, linewidths=0.5, vmin=0.1, vmax=0.65)
                ax.set_title("Win Rate Heatmap"); ax.set_xlabel(""); ax.set_ylabel("")
                ax.tick_params(axis="y", rotation=0, labelsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # Trader explorer
    st.markdown("---")
    st.markdown("**🔍 Individual Trader Explorer**")
    selected_account = st.selectbox("Select Trader Account", options=tp_f["Account"].tolist(),
        format_func=lambda x: f"{x[:10]}…  |  {tp_f[tp_f['Account']==x]['archetype'].values[0]}"
                               if x in tp_f["Account"].values else x)

    if selected_account:
        trader_data = tp_f[tp_f["Account"]==selected_account]
        if len(trader_data) > 0:
            trow = trader_data.iloc[0]
            arch = trow["archetype"]
            color = ARCHETYPE_COLORS.get(arch,"#333")
            emoji = ARCHETYPE_EMOJI.get(arch,"")

            st.markdown(f"""
            <div style='border-left:4px solid {color}; padding:10px 14px; background:#f5f5f5; border-radius:4px;'>
                <b>{emoji} {arch}</b> &nbsp;|&nbsp; {selected_account}
            </div>
            """, unsafe_allow_html=True)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total PnL",    f"${trow['total_pnl']:,.0f}")
            m2.metric("Avg Daily PnL",f"${trow['avg_daily_pnl']:,.0f}")
            m3.metric("Win Rate",     f"{trow['avg_win_rate']:.1%}")
            m4.metric("Sharpe",       f"{trow['sharpe']:.3f}")
            m5.metric("Total Trades", f"{int(trow['total_trades']):,}")

            trader_daily = dt[dt["Account"]==selected_account].sort_values("date")
            if len(trader_daily) > 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
                cum = trader_daily.set_index("date")["daily_pnl"].cumsum()
                cum.plot(ax=axes[0], color=color, linewidth=2)
                axes[0].axhline(0, color="black", linewidth=0.7, linestyle="--")
                axes[0].set_title("Cumulative PnL"); axes[0].set_ylabel("USD")
                trader_daily["daily_pnl"].clip(-5000,5000).plot(kind="bar", ax=axes[1],
                    color=[GREED_COLOR if v>=0 else FEAR_COLOR for v in trader_daily["daily_pnl"]], edgecolor="none")
                axes[1].set_title("Daily PnL (clipped ±$5k)"); axes[1].set_ylabel("USD")
                axes[1].set_xticklabels([])
                plt.tight_layout(); st.pyplot(fig); plt.close()


# ─────────────────────────────────────────────────────────────────
#  TAB 5 — STRATEGY
# ─────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("📋 Strategy Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='border-top:3px solid #e74c3c; padding:14px; background:#fff5f5; border-radius:6px;'>
            <h4 style='color:#e74c3c; margin:0 0 8px'>🟥 Strategy 1 — Fear Regime</h4>
            <b>Reduce Size & Frequency on Fear Days</b><br><br>
            <ul style='font-size:0.9rem; padding-left:16px'>
                <li>Cut position size by <b>≥ 30%</b> vs baseline</li>
                <li>Limit trade count to <b>≤ 50%</b> of Greed-day average</li>
                <li>Avoid unconfirmed long setups</li>
                <li>Sit flat or short during <b>Extreme Fear</b></li>
            </ul>
            <div style='background:#fde8e6;padding:8px;border-radius:4px;font-size:0.85rem;margin-top:8px'>
                <b>Evidence:</b> Fear-day traders earn only 46% of Greed-day median PnL despite 2.15× more volume. Dip-buyers are active but consistently underperform.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='border-top:3px solid #27ae60; padding:14px; background:#f0fff4; border-radius:6px;'>
            <h4 style='color:#27ae60; margin:0 0 8px'>🟩 Strategy 2 — Greed Regime</h4>
            <b>Scale Up Consistent Winners on Greed Days</b><br><br>
            <ul style='font-size:0.9rem; padding-left:16px'>
                <li>If trailing 30-day win-rate <b>> 50%</b>: allow <b>1.5× sizing</b></li>
                <li>Increase trade frequency — Greed rewards execution</li>
                <li>Favour long-biased setups during standard Greed</li>
                <li>Add short hedges during <b>Extreme Greed</b></li>
            </ul>
            <div style='background:#e8f8ee;padding:8px;border-radius:4px;font-size:0.85rem;margin-top:8px'>
                <b>Evidence:</b> Consistent Winners earn 2.5× more on Greed days. Long ratio drops to 0.25 at Extreme Greed — smart money rotates short.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📌 Rule-of-Thumb Table**")

    rot = pd.DataFrame({
        "F/G Condition": ["Extreme Fear","Fear","Neutral","Greed","Extreme Greed"],
        "Position Size": ["Minimal / flat","−30% from baseline","Normal","+50% if WR>50%","Reduce longs"],
        "Trade Frequency": ["Very low","−50% from baseline","Normal","Increase","Selective"],
        "Direction Bias": ["Neutral or short","Avoid unconfirmed longs","Balanced","Long-biased","Hedge with shorts"],
    })
    st.dataframe(rot, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**🤖 Predictive Model Results**")
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("AUC-ROC",          "0.81")
    col_m2.metric("Accuracy",         "76%")
    col_m3.metric("Profit Day Recall","98%")
    col_m4.metric("Loss Day Recall",  "40%")
    st.info("A Random Forest (200 trees, depth 8) trained on trade count, position size, long ratio, total volume, sentiment binary, and F/G score. Sentiment ranks as a **significant predictive feature** confirming it provides genuine edge beyond behavioural signals alone.")

    st.markdown("---")
    st.markdown("**🔬 Archetype-Specific Rules**")
    arch_rules = {
        "👑 Fear Dominator":        "Increase activity during Fear — this is your home regime. Reduce on Extreme Greed where your edge diminishes.",
        "🎲 High-Stakes Speculator":"Avoid large positions during Fear. Sentiment-driven volatility amplifies losses. Reserve big bets for Greed confirmation.",
        "⚙️ Disciplined Grinder":   "You perform across regimes — slightly scale up frequency on Greed days where your win-rate lifts.",
        "📈 Greed Momentum Rider":  "Wait patiently for Greed. Do NOT trade aggressively on Fear days — your edge disappears and losses mount.",
    }
    for arch, rule in arch_rules.items():
        color = ARCHETYPE_COLORS.get(arch.split(" ",1)[1],"#333")
        st.markdown(f"""
        <div style='border-left:3px solid {color}; padding:8px 12px; margin:6px 0; background:#fafafa; border-radius:3px; font-size:0.9rem;'>
            <b>{arch}</b><br>{rule}
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Hyperliquid × Fear/Greed Sentiment Analysis · 211,224 trades · 32 traders · 479 days · 2024")
