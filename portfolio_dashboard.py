"""
PATCH: Fix HTML rendering in portfolio_dashboard.py
====================================================

PROBLEM: Streamlit's markdown parser treats 4+ leading spaces as code blocks.
Multi-line f-strings with indented HTML get dumped as raw text after the first card.

FIX: Add _clean_html() helper that strips leading whitespace per line, then use it
on any HTML block passed to st.markdown().

INSTRUCTIONS:
1. Add the _clean_html() function at module level (anywhere above class Dashboard)
2. Replace render_bucket_cards() method in the Dashboard class
3. Replace render_positions() method in the Dashboard class
4. Replace render_health_strip() method in the Dashboard class
5. Save and rerun streamlit

Or simpler: copy this entire file's functions into your existing file, overwriting
the matching methods.
"""

# ============================================================================
# HELPER — add at module level above class Dashboard
# ============================================================================

def _clean_html(html: str) -> str:
    """
    Strip leading whitespace from every line to prevent Streamlit's markdown
    parser from treating indented HTML as code blocks.
    """
    return '\n'.join(line.strip() for line in html.split('\n') if line.strip())


# ============================================================================
# REPLACEMENT: render_health_strip
# ============================================================================

def render_health_strip(self, stats: dict, fx_rate: float):
    """Level 1 health metrics — always visible"""
    import streamlit as st

    pnl_class = "up" if stats['total_pnl'] >= 0 else "dn"
    pnl_arrow = "▲" if stats['total_pnl'] >= 0 else "▼"

    cash_pct = stats['cash_pct']
    cash_min = 2  # config.CASH_TARGET_MIN * 100
    cash_max = 10  # config.CASH_TARGET_MAX * 100
    if cash_pct > cash_max:
        cash_status = '<span class="dn">over band</span>'
    elif cash_pct < cash_min:
        cash_status = '<span class="dn">under band</span>'
    else:
        cash_status = '<span class="up">in band</span>'

    # Build HTML as single-line segments joined together (NO leading whitespace)
    parts = [
        '<div class="health-strip">',
        '<div class="health-metric">',
        '<div class="health-label">Net Value</div>',
        f'<div class="health-value">A${stats["total_mv"]:,.0f}</div>',
        f'<div class="health-sub">Capital A${stats["capital_injected"]:,.0f}</div>',
        '</div>',
        '<div class="health-metric">',
        '<div class="health-label">Total P&amp;L</div>',
        f'<div class="health-value {pnl_class}">{pnl_arrow} A${stats["total_pnl"]:,.0f}</div>',
        f'<div class="health-sub {pnl_class}">{stats["pnl_pct"]:+.2f}%</div>',
        '</div>',
        '<div class="health-metric">',
        '<div class="health-label">Stock P&amp;L</div>',
        f'<div class="health-value">{stats["stock_pnl_pct"]:+.2f}%</div>',
        f'<div class="health-sub">A${stats["stock_pnl"]:,.0f}</div>',
        '</div>',
        '<div class="health-metric">',
        '<div class="health-label">Cash</div>',
        f'<div class="health-value">{cash_pct:.1f}%</div>',
        f'<div class="health-sub">{cash_status} · target {cash_min}-{cash_max}%</div>',
        '</div>',
        '<div class="health-metric">',
        '<div class="health-label">Positions</div>',
        f'<div class="health-value">{stats["num_positions"]}</div>',
        f'<div class="health-sub"><span class="up">{stats["num_winners"]}W</span> · <span class="dn">{stats["num_losers"]}L</span></div>',
        '</div>',
        '<div class="health-metric">',
        '<div class="health-label">AUD/USD</div>',
        f'<div class="health-value">{fx_rate:.4f}</div>',
        '</div>',
        '</div>',
    ]
    st.markdown(''.join(parts), unsafe_allow_html=True)


# ============================================================================
# REPLACEMENT: render_bucket_cards
# ============================================================================

def render_bucket_cards(self, buckets):
    """Level 2 — framework bucket cards. CRITICAL: HTML must be single-line."""
    import streamlit as st

    # Reference to module-level config (adjust if your variable name differs)
    BUCKET_COLORS = {
        'Core':     '#1e40af',
        'Growth':   '#059669',
        'Tactical': '#d97706',
        'Cleanup':  '#dc2626',
    }
    BUCKET_LABELS = {
        'Core':     '核心 Core',
        'Growth':   '成長 Growth',
        'Tactical': '戰術 Tactical',
        'Cleanup':  '非框架 Cleanup',
    }
    action_colors = {
        'BUY':       '#059669',
        'TRIM':      '#d97706',
        'EXIT':      '#dc2626',
        'ON TARGET': '#1e40af',
        'CLEAN':     '#64748b',
    }

    if not buckets:
        st.info("⚠️ Add a 'Strategy_Role' column to your sheet: Core / Growth / Tactical / Cleanup")
        return

    st.markdown("## 🎯 Strategy Buckets")

    # Build ALL cards into ONE flat HTML string with NO line breaks between elements
    parts = ['<div class="bucket-grid">']

    for b in buckets:
        name = b['name']
        color = BUCKET_COLORS.get(name, '#64748b')
        label = BUCKET_LABELS.get(name, name)
        action_color = action_colors.get(b['action'], '#64748b')

        if name == 'Cleanup':
            fill_pct = min(100, b['current_pct'] * 5)
        else:
            fill_pct = min(100, (b['current_pct'] / b['target_pct'] * 100)) if b['target_pct'] > 0 else 0

        if name == 'Cleanup':
            gap_class = 'dn' if b['current_pct'] > 0 else 'up'
        else:
            gap_class = 'up' if b['gap_pct'] >= 0 else 'dn'

        gap_sign = '+' if b['gap_pct'] >= 0 else ''
        count_label = f"{b['count']} holding" + ('s' if b['count'] != 1 else '')

        # Everything as single-line concatenation — NO indentation
        card = (
            f'<div class="bucket-card" style="border-color:{color};">'
            f'<div class="bucket-name" style="color:{color};">{label}</div>'
            f'<div class="bucket-pct">{b["current_pct"]:.1f}%</div>'
            f'<div class="bucket-target">Target {b["target_pct"]:.0f}% · {count_label}</div>'
            f'<div class="bucket-progress">'
            f'<div class="bucket-fill" style="width:{fill_pct:.1f}%;background:{color};"></div>'
            f'</div>'
            f'<div class="bucket-footer">'
            f'<div class="bucket-row">'
            f'<span class="bucket-label-txt">Gap</span>'
            f'<span class="{gap_class}">{gap_sign}{b["gap_pct"]:.1f}%</span>'
            f'</div>'
            f'<div class="bucket-row">'
            f'<span class="bucket-label-txt">Value</span>'
            f'<span class="bucket-value-txt">A${b["mv_aud"]:,.0f}</span>'
            f'</div>'
            f'<div class="bucket-row" style="margin-top:8px;">'
            f'<span class="action-flag" style="background:{action_color};">{b["action"]}</span>'
            f'<span class="bucket-label-txt" style="font-size:0.7rem;">A${abs(b["gap_aud"]):,.0f}</span>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        parts.append(card)

    parts.append('</div>')

    # Join with NO separator — keep it as one continuous string
    st.markdown(''.join(parts), unsafe_allow_html=True)


# ============================================================================
# REPLACEMENT: render_positions
# ============================================================================

def render_positions(self, df, stats):
    """Level 3 — position cards filtered by bucket"""
    import streamlit as st

    BUCKET_COLORS = {
        'Core':     '#1e40af',
        'Growth':   '#059669',
        'Tactical': '#d97706',
        'Cleanup':  '#dc2626',
    }
    action_colors = {
        'BUY':       '#059669',
        'TRIM':      '#d97706',
        'EXIT':      '#dc2626',
        'ON TARGET': '#1e40af',
        '—':         '#64748b',
    }

    st.markdown("## 📋 Positions")

    # Bucket filter with query param persistence
    default_bucket = st.query_params.get('bucket', 'All')
    if isinstance(default_bucket, list):
        default_bucket = default_bucket[0] if default_bucket else 'All'

    bucket_options = ['All', 'Core', 'Growth', 'Tactical', 'Cleanup']
    if default_bucket not in bucket_options:
        default_bucket = 'All'

    selected = st.radio(
        "Filter by bucket",
        bucket_options,
        horizontal=True,
        label_visibility="collapsed",
        index=bucket_options.index(default_bucket),
        key="bucket_filter_radio"
    )

    if selected != default_bucket:
        st.query_params['bucket'] = selected

    pos_df = self.analytics.build_position_table(df, stats)
    if pos_df.empty:
        st.info("No positions found")
        return

    if selected != 'All':
        pos_df = pos_df[pos_df['Bucket'] == selected]

    if pos_df.empty:
        st.info(f"No positions in {selected} bucket")
        return

    # Build ALL position cards into ONE flat HTML string
    parts = []
    for _, r in pos_df.iterrows():
        bucket = r['Bucket']
        color = BUCKET_COLORS.get(bucket, '#64748b')
        action = r['Action']
        action_color = action_colors.get(action, '#64748b')

        pnl_class = 'up' if r['PnL_AUD'] >= 0 else 'dn'
        pnl_arrow = '▲' if r['PnL_AUD'] >= 0 else '▼'
        tag_bg = f"{color}22"

        if r.get('Target_Pct', 0) > 0:
            gap_str = f"{r['Gap_Pct']:+.1f}%"
        else:
            gap_str = '—'

        target_display = f"{r.get('Target_Pct', 0):.0f}%"

        card = (
            f'<div class="position-card" style="border-left-color:{color};">'
            f'<div class="pos-header">'
            f'<div>'
            f'<span class="pos-ticker">{r["Ticker"]}</span>'
            f'<span class="pos-tag" style="background:{tag_bg};color:{color};margin-left:8px;">{bucket}</span>'
            f'</div>'
            f'<span class="action-flag" style="background:{action_color};">{action}</span>'
            f'</div>'
            f'<div class="pos-grid">'
            f'<div>'
            f'<div class="pos-cell-label">Value</div>'
            f'<div class="pos-cell-value">A${r["MV_AUD"]:,.0f}</div>'
            f'</div>'
            f'<div>'
            f'<div class="pos-cell-label">Weight / Target</div>'
            f'<div class="pos-cell-value">{r["Weight_Pct"]:.1f}% / {target_display}</div>'
            f'</div>'
            f'<div>'
            f'<div class="pos-cell-label">P&amp;L</div>'
            f'<div class="pos-cell-value {pnl_class}">{pnl_arrow} {r["PnL_Pct"]:+.1f}%</div>'
            f'</div>'
            f'<div>'
            f'<div class="pos-cell-label">Gap</div>'
            f'<div class="pos-cell-value">{gap_str}</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        parts.append(card)

    st.markdown(''.join(parts), unsafe_allow_html=True)
