"""
Advocacy Fundraising Dashboard
Dash + Plotly. All data comes from the tracker view (jolly / warner / ohio_dems
unioned with a `source_campaign` discriminator). A campaign is picked via the
header dropdown; everything downstream (metrics, figures, layout content) is
rebuilt in the callback from the in-memory master DataFrame — no re-query.
"""
import json
import math
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from google.cloud import bigquery
from google.oauth2 import service_account

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from _auth import init_auth

# ── Credentials ────────────────────────────────────────────────────────────
# GOOGLE_APPLICATION_CREDENTIALS_JSON holds the raw service-account JSON
# (used on Render). When absent, the client falls back to ADC / gcloud
# auth application-default login (used locally).
_creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if _creds_json:
    _creds_info = json.loads(_creds_json)
    _credentials = service_account.Credentials.from_service_account_info(_creds_info)
else:
    _credentials = None

# ── Config ─────────────────────────────────────────────────────────────────
PROJECT_ID = "sys-93912543499576679810429295"
DATASET    = "tracker_views"
TABLE      = "fundraising_tracker_view"

# Per-campaign monthly goals. Placeholders — swap for real numbers.
CAMPAIGN_GOALS = {
    "jolly":     300_000,
    "warner":    500_000,
    "ohio_dems": 250_000,
}

# Display labels for the dropdown + the big header title.
CAMPAIGN_LABELS = {
    "jolly":     "David Jolly",
    "warner":    "Mark Warner",
    "ohio_dems": "Ohio Democratic Party",
}

DEFAULT_CAMPAIGN = "jolly"

# ── Client + one-time data load ────────────────────────────────────────────
# We pull the whole view once at startup and keep it resident. Every dropdown
# selection then filters this frame in memory — no BigQuery round-trip per
# switch. Trade-off: data is stale until the process restarts.
bq_client = bigquery.Client(project=PROJECT_ID, credentials=_credentials)
table_id  = f"{PROJECT_ID}.{DATASET}.{TABLE}"

raw_df = bq_client.query(f"""
    SELECT source_campaign, actblue_custom_date, amount, recurs
    FROM `{table_id}`
""").to_dataframe()

raw_df["actblue_custom_date"] = pd.to_datetime(
    raw_df["actblue_custom_date"], format="%Y-%m-%d %H:%M:%S"
)
raw_df["amount"] = pd.to_numeric(raw_df["amount"], errors="coerce")

# ══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════
BG          = "#f0f0ec"       # warm off-white page bg
SURFACE     = "#ffffff"       # pure white cards
SURFACE2    = "#f7f7f4"       # off-white inner surfaces
BORDER      = "#e2e2dc"       # soft warm grey border
ACCENT      = "#16a34a"       # deep emerald — positive metrics
ACCENT2     = "#0369a1"       # steel blue — secondary metrics
GREEN       = "#16a34a"
RED         = "#dc2626"
TEXT        = "#1c1917"
MUTED       = "#78716c"
TITLE_COLOR = "#292524"
LABEL_COLOR = "#57534e"
FONT_TITLE  = "'Bebas Neue', sans-serif"
FONT_BODY   = "'DM Sans', sans-serif"
FONT_MONO   = "'DM Mono', monospace"

GOOGLE_FONTS = "https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap"

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=FONT_BODY, color="#1c1917"),
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis=dict(showgrid=False, zeroline=False, color=MUTED, tickfont=dict(size=11, color=MUTED)),
    yaxis=dict(showgrid=True, gridcolor="#e2e2dc", zeroline=False, color=MUTED, tickfont=dict(size=11, color=MUTED)),
)

# ══════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ══════════════════════════════════════════════════════════════════════════
def stat_card(label, value, sub=None, accent_color=ACCENT):
    return html.Div([
        html.P(label, style={
            "margin": "0 0 6px 0",
            "fontSize": "11px",
            "letterSpacing": "2px",
            "textTransform": "uppercase",
            "color": LABEL_COLOR,
            "fontFamily": FONT_MONO,
        }),
        html.P(value, style={
            "margin": "0",
            "fontSize": "28px",
            "fontFamily": FONT_TITLE,
            "letterSpacing": "1px",
            "color": accent_color,
            "lineHeight": "1",
        }),
        html.P(sub, style={
            "margin": "6px 0 0 0",
            "fontSize": "12px",
            "color": LABEL_COLOR,
            "fontFamily": FONT_BODY,
        }) if sub else None,
    ], style={
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "20px 24px",
        "flex": "1",
        "minWidth": "160px",
    })

def section_card(title, children, style=None, tooltip=None):
    title_style = {
        "margin": "0 0 20px 0",
        "fontSize": "16px",
        "letterSpacing": "2px",
        "textTransform": "uppercase",
        "color": TITLE_COLOR,
        "fontFamily": FONT_MONO,
        "borderBottom": f"1px solid {BORDER}",
        "paddingBottom": "12px",
    }
    title_kwargs = {"style": title_style}
    if tooltip:
        title_kwargs["className"] = "tt"
        title_kwargs["title"] = tooltip
    return html.Div([
        html.P(title, **title_kwargs),
        *children,
    ], style={
        "background": SURFACE,
        "border": f"1px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "24px",
        **(style or {}),
    })

# ══════════════════════════════════════════════════════════════════════════
# DASHBOARD BUILDER
# Recomputes MTD metrics, the historical avg_curve, and the single retained
# chart (distribution-adjusted pace & projection) for a given campaign and
# goal, then returns the full content block (campaign banner + KPI row +
# chart). Called by the callback on any dropdown or goal change.
# ══════════════════════════════════════════════════════════════════════════
def build_dashboard(campaign: str, monthly_goal: int | None = None):
    # Fall back to the campaign's stored default if the caller didn't override.
    monthly_goal = int(monthly_goal) if monthly_goal else CAMPAIGN_GOALS[campaign]

    # ── Filter master frame to this campaign ─────────────────────────────
    df_all = raw_df[raw_df["source_campaign"] == campaign].copy()

    today               = pd.Timestamp.now()
    today_date          = today.date()
    days_in_month       = pd.Timestamp(today.year, today.month, 1).days_in_month
    day_of_month        = today.day
    current_month_start = pd.Timestamp(today.year, today.month, 1)

    # ── Current month (MTD) slice ────────────────────────────────────────
    df = df_all[df_all["actblue_custom_date"] >= current_month_start].copy()

    raised_mtd      = float(df["amount"].sum())
    daily_avg       = raised_mtd / day_of_month if day_of_month else 0.0
    pct_to_goal     = (raised_mtd / monthly_goal) * 100 if monthly_goal else 0.0
    # projected_total is computed after avg_curve is built (smart projection).

    df["date"] = df["actblue_custom_date"].dt.date
    daily_totals = (
        df.groupby("date")["amount"]
        .sum()
        .reset_index()
        .sort_values("date")
    )
    daily_totals["cumulative"] = daily_totals["amount"].cumsum()
    actual_dates    = list(daily_totals["date"])
    actual_cum_vals = list(daily_totals["cumulative"])

    # ── Historical daily aggregation (last 6 complete months) ────────────
    # This is the ONLY historical view we still compute — it feeds avg_curve,
    # which drives the smart pace curve on the retained chart.
    hist_source = df_all[df_all["actblue_custom_date"] < current_month_start].copy()

    if not hist_source.empty:
        hist_source["month"]     = hist_source["actblue_custom_date"].values.astype("datetime64[M]")
        hist_source["dt"]        = hist_source["actblue_custom_date"].dt.normalize()
        hist_source["month_end"] = hist_source["dt"] + pd.offsets.MonthEnd(0)
        latest_months = sorted(hist_source["month"].unique(), reverse=True)[:6]
        daily_source  = hist_source[hist_source["month"].isin(latest_months)].copy()
        daily_source["day"]           = daily_source["dt"].dt.day
        daily_source["days_in_month"] = daily_source["month_end"].dt.day

        daily_hist_df = (
            daily_source.groupby(["month", "day", "days_in_month"], as_index=False)["amount"]
            .sum()
            .rename(columns={"amount": "daily_total"})
            .sort_values(["month", "day"])
        )
    else:
        daily_hist_df = pd.DataFrame(columns=["month", "day", "days_in_month", "daily_total"])

    for _col in ("day", "days_in_month", "daily_total"):
        daily_hist_df[_col] = pd.to_numeric(daily_hist_df[_col], errors="coerce")

    # ── Build avg_curve: typical within-month cumulative shape ───────────
    # For each historical month, build a cumulative share curve on a
    # normalized fraction-of-month axis (t = day / days_in_month) anchored
    # at (0,0) and (1,1), interpolate onto a common 0..1 grid, then average.
    # avg_curve[i] answers: "by the moment i% of the month has elapsed, what
    # fraction of the total has historically been raised?"
    _COMMON_GRID = np.linspace(0.0, 1.0, 101)

    month_curves = {}
    for _month, _grp in daily_hist_df.groupby("month"):
        _grp = _grp.sort_values("day").copy()
        _days_in_month_h = int(_grp["days_in_month"].iloc[0])
        _grp["t"]       = _grp["day"] / _days_in_month_h
        _grp["cum_amt"] = _grp["daily_total"].cumsum()
        _month_total    = _grp["cum_amt"].iloc[-1]
        if _month_total <= 0:
            continue
        _grp["cum_share"] = _grp["cum_amt"] / _month_total
        _xs = np.concatenate([[0.0], _grp["t"].to_numpy(),       [1.0]])
        _ys = np.concatenate([[0.0], _grp["cum_share"].to_numpy(), [1.0]])
        month_curves[pd.Timestamp(_month)] = (_xs, _ys)

    interpolated_curves = {
        _month: np.interp(_COMMON_GRID, _xs, _ys)
        for _month, (_xs, _ys) in month_curves.items()
    }
    avg_curve = (
        np.mean(list(interpolated_curves.values()), axis=0)
        if interpolated_curves else _COMMON_GRID.copy()
    )

    # ── Smart EOM projection ─────────────────────────────────────────────
    # If by the moment t = day_of_month / days_in_month has elapsed the
    # historical avg_curve says share S of the monthly total is typically
    # in, then the implied full-month projection is raised_mtd / S. Falls
    # back to a naive linear extrapolation only when S is unreliably small
    # (e.g. day 1 of the month). When there's no history, avg_curve is
    # linear so this collapses to the naive projection automatically.
    _t_today           = day_of_month / days_in_month
    _smart_share_today = float(np.interp(_t_today, _COMMON_GRID, avg_curve))
    if _smart_share_today > 0.001:
        projected_total = raised_mtd / _smart_share_today
    else:
        projected_total = daily_avg * days_in_month

    # ══════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════
    month_start_date = today.replace(day=1).date()
    month_end_date   = today.replace(day=days_in_month).date()
    _goal_mid_date   = month_start_date + timedelta(days=days_in_month // 2)

    # Smart forecast — extends from today's raised_mtd along the historical
    # curve shape. For any future date f, projected cumulative =
    # raised_mtd * (avg_curve_share[f] / avg_curve_share[today]), so the
    # line starts at raised_mtd today and lands at `projected_total` on the
    # last day of the month. Reweights future days by their historical
    # contribution instead of assuming a flat daily rate.
    forecast_dates = [today_date + timedelta(days=i)
                      for i in range((month_end_date - today_date).days + 1)]
    if _smart_share_today > 0.001:
        _forecast_shares = np.interp(
            [d.day / days_in_month for d in forecast_dates],
            _COMMON_GRID, avg_curve,
        )
        _scale = raised_mtd / _smart_share_today
        smart_forecast = list(_forecast_shares * _scale)
    else:
        smart_forecast = [raised_mtd + daily_avg * i for i in range(len(forecast_dates))]

    # ── Smart pacing curve: distribution-adjusted reference ──────────────
    _smart_pace_t        = np.array([d / days_in_month for d in range(1, days_in_month + 1)])
    _smart_pace_share    = np.interp(_smart_pace_t, _COMMON_GRID, avg_curve)
    smart_pace_dollars   = _smart_pace_share * monthly_goal
    smart_pace_dates     = [pd.Timestamp(today.year, today.month, d).date() for d in range(1, days_in_month + 1)]
    smart_pace_at_actual = [smart_pace_dollars[d.day - 1] for d in actual_dates]

    min_envelope_smart = [min(a, p) for a, p in zip(actual_cum_vals, smart_pace_at_actual)]
    above_pace_y_smart = [max(a, p) for a, p in zip(actual_cum_vals, smart_pace_at_actual)]

    smart_spark_fig = go.Figure()

    smart_spark_fig.add_trace(go.Scatter(
        x=actual_dates, y=min_envelope_smart,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tozeroy", fillcolor="rgba(3,105,161,0.18)",
        hoverinfo="skip", showlegend=False,
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=actual_dates, y=smart_pace_at_actual,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=actual_dates, y=above_pace_y_smart,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(22,163,74,0.28)",
        hoverinfo="skip", showlegend=False,
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=actual_dates, y=smart_pace_at_actual,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip", showlegend=False,
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=actual_dates, y=min_envelope_smart,
        mode="lines", line=dict(color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(220,38,38,0.28)",
        hoverinfo="skip", showlegend=False,
    ))

    _smart_deltas     = [a - p for a, p in zip(actual_cum_vals, smart_pace_at_actual)]
    _smart_delta_strs = [
        f"+${d:,.0f}" if d >= 0 else f"-${abs(d):,.0f}"
        for d in _smart_deltas
    ]
    _smart_customdata = list(zip(smart_pace_at_actual, _smart_delta_strs))

    smart_spark_fig.add_trace(go.Scatter(
        x=daily_totals["date"], y=daily_totals["cumulative"],
        mode="lines", line=dict(color=ACCENT2, width=2.5),
        customdata=_smart_customdata,
        hovertemplate=(
            "<b>%{x|%b %-d}</b>"
            "<br>Raised: <b>$%{y:,.0f}</b>"
            "<br>Expected: $%{customdata[0]:,.0f}"
            "<br>Δ: <b>%{customdata[1]}</b>"
            "<extra></extra>"
        ),
        showlegend=True, name="Raised MTD",
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=smart_pace_dates, y=smart_pace_dollars,
        mode="lines",
        line=dict(color="#000000", width=2, dash="dash"),
        hovertemplate="<b>Goal pace</b>: $%{y:,.0f}<extra></extra>",
        showlegend=True, name="Goal Pace",
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=forecast_dates, y=smart_forecast,
        mode="lines", line=dict(color=ACCENT2, width=2, dash="dot"),
        hovertemplate="<b>Projected</b>: $%{y:,.0f}<extra></extra>",
        showlegend=True, name="Projection",
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=[month_start_date, month_end_date],
        y=[monthly_goal, monthly_goal],
        mode="lines", line=dict(color="#eab308", width=1.5),
        hovertemplate="<b>Monthly goal</b>: $%{y:,.0f}<extra></extra>",
        showlegend=True, name="Monthly Goal",
    ))
    smart_spark_fig.add_trace(go.Scatter(
        x=[_goal_mid_date], y=[monthly_goal],
        mode="text",
        text=[f"Goal: ${monthly_goal:,.0f}"],
        textposition="top center",
        textfont=dict(size=11, color="#eab308", weight="bold"),
        cliponaxis=False, hoverinfo="skip",
        showlegend=False, name="Goal Label",
    ))
    smart_spark_fig.update_layout(
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ["margin", "xaxis", "yaxis"]},
        height=280,
        showlegend=True,
        # Horizontal legend sits below the plot area, entries spread evenly
        # across the full width of the card so they don't feel crammed.
        # Left/right margins match so the plot area is centered in the card,
        # which in turn centers the legend under it.
        legend=dict(
            orientation="h",
            x=0.5, y=-0.22,
            xanchor="center", yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_MONO, size=11, color=TITLE_COLOR),
            itemsizing="constant",
            itemwidth=50,
            entrywidth=0.24,
            entrywidthmode="fraction",
            traceorder="normal",
        ),
        margin=dict(l=60, r=60, t=20, b=50),
        xaxis=dict(
            showgrid=False, zeroline=False,
            tickfont=dict(size=11, color=MUTED),
            range=[str(month_start_date), str(month_end_date)],
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#e2e2dc", zeroline=False,
            tickfont=dict(size=11, color=MUTED),
            tickprefix="$",
            tickformat=",.0f",
        ),
    )

    # ── Raised MTD ring donut ────────────────────────────────────────────
    # "Expected" here means: the fraction of the goal we'd expect to have raised
    # by today if this month follows the same within-month shape as the last 6
    # complete months. _smart_share_today (computed above) is that historical
    # share at t = day_of_month / days_in_month; multiplying by the goal gives
    # the dollar target. The pill, ring color, and ring tick all compare against
    # this same reference so they stay internally consistent. For campaigns with
    # no history avg_curve degrades to a straight line and this collapses to
    # plain goal × day/days.
    expected_pace_dollars = _smart_share_today * monthly_goal
    expected_pace_pct     = _smart_share_today * 100  # tick placement on the ring
    on_pace               = raised_mtd >= expected_pace_dollars
    ring_color            = ACCENT if on_pace else RED
    pace_delta_pct        = ((raised_mtd / expected_pace_dollars - 1) * 100) if expected_pace_dollars > 0 else 0.0
    pace_delta_dollars    = abs(raised_mtd - expected_pace_dollars)
    pace_direction        = "ABOVE" if on_pace else "BELOW"
    # Sentence subtitle carries the dollar delta only; the ring donut center
    # carries the percent delta (with sign) so the two aren't redundant.
    pace_delta_phrase     = f"${pace_delta_dollars:,.0f} {pace_direction}"
    pace_delta_pct_signed = f"{'+' if on_pace else '−'}{abs(pace_delta_pct):.1f}%"

    projected_delta_dollars = projected_total - monthly_goal
    projected_above_goal    = projected_delta_dollars >= 0
    projected_pill_color    = ACCENT if projected_above_goal else RED
    projected_pill_bg       = "rgba(22,163,74,0.12)" if projected_above_goal else "rgba(220,38,38,0.12)"
    projected_pill_text     = f"{'▲' if projected_above_goal else '▼'} ${abs(projected_delta_dollars):,.0f} {'ABOVE' if projected_above_goal else 'BELOW'} GOAL"

    # Tick geometry — pie fills a 140×140 square so its outer radius is 0.5 in
    # paper coordinates. Angle is measured clockwise from 12 o'clock. Placed at
    # the smart-pace share so the tick and the pill/color agree.
    _tick_angle    = math.radians(expected_pace_pct / 100 * 360)
    _sin_a, _cos_a = math.sin(_tick_angle), math.cos(_tick_angle)
    _outer_r       = 0.5
    _inner_r       = _outer_r * 0.78
    _tick_x0 = 0.5 + (_inner_r - 0.03) * _sin_a
    _tick_y0 = 0.5 + (_inner_r - 0.03) * _cos_a
    _tick_x1 = 0.5 + (_outer_r + 0.03) * _sin_a
    _tick_y1 = 0.5 + (_outer_r + 0.03) * _cos_a

    _remaining_pct = max(0, 100 - pct_to_goal)
    mtd_donut = go.Figure(go.Pie(
        values=[pct_to_goal, _remaining_pct],
        hole=0.78,
        marker=dict(colors=[ring_color, BORDER], line=dict(width=0)),
        textinfo="none",
        hoverinfo="skip",
        sort=False,
        direction="clockwise",
        rotation=0,
    ))
    mtd_donut.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=140,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        shapes=[
            dict(
                type="line",
                xref="paper", yref="paper",
                x0=_tick_x0, y0=_tick_y0,
                x1=_tick_x1, y1=_tick_y1,
                line=dict(color=TITLE_COLOR, width=2),
            ),
        ],
        annotations=[
            dict(
                text=pace_delta_pct_signed,
                x=0.5, y=0.56,
                font=dict(size=26, color=ring_color, family=FONT_TITLE),
                showarrow=False,
                align="center",
            ),
            dict(
                text="VS EXPECTED PACE",
                x=0.5, y=0.38,
                font=dict(size=9, color=LABEL_COLOR, family=FONT_MONO),
                showarrow=False,
                align="center",
            ),
        ],
    )

    # ══════════════════════════════════════════════════════════════════════
    # CONTENT (returned as the callback's children)
    # ══════════════════════════════════════════════════════════════════════
    return html.Div([

        # ── Top KPI row ──────────────────────────────────────────────────
        html.Div([

            # Raised + Projected — paired stats card
            html.Div([
                html.P("FUNDRAISING — MTD", style={
                    "fontFamily": FONT_MONO, "fontSize": "16px", "letterSpacing": "2px",
                    "textTransform": "uppercase", "color": TITLE_COLOR, "margin": "0 0 16px 0",
                    "borderBottom": f"1px solid {BORDER}", "paddingBottom": "12px",
                }),
                html.Div([
                    # Left half: single hero number (Raised MTD) with a
                    # compact colored subtitle below combining the delta
                    # and the reference amount (Expected Pace).
                    html.Div([
                        html.Div([
                            html.P("RAISED MONTH TO DATE", style={
                                "fontFamily": FONT_MONO, "fontSize": "9px", "letterSpacing": "2px",
                                "color": LABEL_COLOR, "margin": "0 0 4px 0",
                                "whiteSpace": "nowrap",
                                "borderBottom": f"1px dotted {LABEL_COLOR}",
                                "paddingBottom": "2px",
                                "display": "inline-block",
                            }),
                            html.P(f"${raised_mtd:,.0f}", style={
                                "fontFamily": FONT_TITLE, "fontSize": "48px", "color": ACCENT2,
                                "margin": "0", "lineHeight": "1",
                                "minHeight": "54px", "display": "flex", "alignItems": "flex-end",
                            }),
                        ],
                            className="tt",
                            title="Total dollars raised so far this month across all ActBlue contributions for this campaign.",
                            style={"width": "fit-content"},
                        ),
                        html.P([
                            html.Span("Current total is ", style={"color": LABEL_COLOR}),
                            html.Span(pace_delta_phrase, style={"color": ring_color, "fontWeight": "600"}),
                            html.Span(f" expected pace of ${expected_pace_dollars:,.0f}", style={"color": LABEL_COLOR}),
                        ],
                            className="tt",
                            title=(
                                "Expected pace is where you'd typically be by today based on the "
                                "last 6 months of monthly giving patterns, scaled to your monthly goal. "
                                "The dollar delta and direction (above/below) compare your actual "
                                "raised MTD to that expected amount."
                            ),
                            style={
                                "fontFamily": FONT_MONO,
                                "fontSize": "11px",
                                "letterSpacing": "0.8px",
                                "textTransform": "uppercase",
                                "margin": "14px 0 0 0",
                                "lineHeight": "1.5",
                                "width": "fit-content",
                            },
                        ),
                    ], style={"flex": "1", "minWidth": "0"}),
                    html.Div(style={"width": "1px", "background": BORDER, "margin": "0 24px", "alignSelf": "stretch"}),
                    # Ring donut sits alongside the two numbers — visually
                    # restates the same raised-vs-expected comparison the
                    # pill above already conveys, with the tick pinned to
                    # the smart-pace share.
                    html.Div(
                        dcc.Graph(
                            figure=mtd_donut,
                            config={"displayModeBar": False},
                            style={"height": "140px", "width": "140px"},
                        ),
                        className="tt",
                        title=(
                            "Ring fills to your % of goal raised. The black tick marks where "
                            "the historical curve says you should be by today. Center number "
                            "shows the % you're ahead (+) or behind (−) that expected pace."
                        ),
                        style={"display": "flex", "alignItems": "center", "justifyContent": "center", "flex": "0 0 auto"},
                    ),
                ], style={"display": "flex", "alignItems": "center", "height": "140px"}),
            ], style={
                "background": SURFACE, "border": f"1px solid {BORDER}",
                "borderRadius": "12px", "padding": "24px", "flex": "2", "minWidth": "560px",
            }),

            # Projected total — its own card
            html.Div([
                html.P("PROJECTED TOTAL", style={
                    "fontFamily": FONT_MONO, "fontSize": "16px", "letterSpacing": "2px",
                    "textTransform": "uppercase", "color": TITLE_COLOR, "margin": "0 0 16px 0",
                    "borderBottom": f"1px solid {BORDER}", "paddingBottom": "12px",
                }),
                html.Div([
                    html.Div(
                        html.P(f"${projected_total:,.0f}", style={
                            "fontFamily": FONT_TITLE, "fontSize": "44px", "color": TITLE_COLOR,
                            "margin": "0", "lineHeight": "1",
                            "minHeight": "50px", "display": "flex", "alignItems": "flex-end",
                        }),
                        className="tt",
                        title=(
                            "Full EOM forecast estimates how much will be raised this month, "
                            "taking into account the historical fundraising curve and the current "
                            "fundraising total."
                        ),
                        style={"width": "fit-content"},
                    ),
                    html.Div(projected_pill_text,
                        className="tt",
                        title=(
                            "Dollar difference between the projected EOM total and your monthly "
                            "goal. Green means you're projecting above the goal, red means below."
                        ),
                        style={
                            "display": "inline-block",
                            "width": "fit-content",
                            "fontFamily": FONT_MONO,
                            "fontSize": "11px",
                            "fontWeight": "400",
                            "letterSpacing": "1.5px",
                            "textTransform": "uppercase",
                            "color": projected_pill_color,
                            "background": projected_pill_bg,
                            "padding": "6px 12px",
                            "borderRadius": "999px",
                            "marginTop": "12px",
                            "border": f"1px solid {projected_pill_color}33",
                        },
                    ),
                ], style={
                    "display": "flex", "flexDirection": "column",
                    "justifyContent": "center", "height": "140px",
                }),
            ], style={
                "background": SURFACE, "border": f"1px solid {BORDER}",
                "borderRadius": "12px", "padding": "24px", "flex": "1", "minWidth": "280px",
            }),

        ], style={
            "display": "flex",
            "gap": "12px",
            "marginBottom": "16px",
            "flexWrap": "wrap",
        }),

        # ── Chart ────────────────────────────────────────────────────────
        html.Div([
            section_card("Fundraising Pacing", [
                dcc.Graph(figure=smart_spark_fig, config={"displayModeBar": False}),
            ],
                tooltip=(
                    "Raised MTD (solid blue) is what has actually been raised each day. "
                    "Goal Pace (black dashed) is what we'd expect the pace to look like based on the historical " 
                    "curve, assuming we met the goal exactly. Projection (dotted blue) applies "
                    "that same shape to where you actually are, projecting through "
                    "EOM. Monthly Goal (yellow) is the target line. Fills between "
                    "actuals and goal pace turn green when you're above, red when "
                    "below."
                ),
            ),
        ], style={"marginBottom": "16px"}),

    ])


# ══════════════════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════════════════
app = dash.Dash(__name__, external_stylesheets=[GOOGLE_FONTS])
app.title = "Fundraising Tracker"

# Flask server (exposed for gunicorn: `gunicorn advo_tracker_test:server`).
# init_auth() registers /login + /logout and a before_request hook that gates
# every other route behind a valid session cookie — but only if APP_USERNAME
# and APP_PASSWORD are set. Locally without those set, auth is a no-op.
server = app.server
init_auth(server)

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{ background: {BG}; color: {TEXT}; }}
            ::-webkit-scrollbar {{ width: 6px; }}
            ::-webkit-scrollbar-track {{ background: {BG}; }}
            ::-webkit-scrollbar-thumb {{ background: #c8c8c2; border-radius: 3px; }}
            ::-webkit-scrollbar-thumb:hover {{ background: #a0a09a; }}
            /* Dropdown overrides — match the warm off-white aesthetic */
            .Select-control, .Select-menu-outer {{
                background-color: {SURFACE} !important;
                border: 1px solid {BORDER} !important;
                border-radius: 8px !important;
                font-family: {FONT_MONO} !important;
                color: {TEXT} !important;
            }}
            .Select-value-label, .Select-option {{
                color: {TEXT} !important;
                font-family: {FONT_MONO} !important;
                letter-spacing: 1px !important;
            }}
            /* ── Tooltips ────────────────────────────────────────────────
             * Any element with class="tt" and a `title` attribute shows a
             * styled tooltip on hover. The JS below copies title -> data-tt
             * (and aria-label for screen readers) then strips title so the
             * browser's own delayed system tooltip doesn't stack on top of
             * ours. CSS reads from data-tt. Wrapper must NOT have
             * overflow:hidden or the tooltip will be clipped. */
            .tt {{ position: relative; cursor: help; }}
            .tt::after {{
                content: attr(data-tt);
                position: absolute;
                bottom: calc(100% + 8px);
                left: 50%;
                transform: translateX(-50%);
                background: #1c1917;
                color: #fafafa;
                padding: 8px 12px;
                border-radius: 6px;
                font-family: {FONT_BODY};
                font-size: 12px;
                font-weight: 400;
                line-height: 1.4;
                letter-spacing: 0.2px;
                text-transform: none;
                text-align: left;
                min-width: 200px;
                max-width: 320px;
                white-space: normal;
                box-shadow: 0 4px 16px rgba(0,0,0,0.18);
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.15s ease-out;
                z-index: 1000;
            }}
            .tt::before {{
                content: "";
                position: absolute;
                bottom: calc(100% + 2px);
                left: 50%;
                transform: translateX(-50%);
                border: 6px solid transparent;
                border-top-color: #1c1917;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.15s ease-out;
                z-index: 1000;
            }}
            .tt:hover::after, .tt:hover::before {{ opacity: 1; }}
        </style>
        <script>
            /* Copy `title` -> `data-tt` (+ aria-label) then remove `title`
             * so browsers don't render their delayed system tooltip on top
             * of our styled CSS tooltip. Runs on load and re-runs whenever
             * Dash re-renders (dashboard-content changes on dropdown/goal
             * updates). */
            document.addEventListener("DOMContentLoaded", function() {{
                const applyTooltips = () => {{
                    document.querySelectorAll(".tt[title]").forEach(el => {{
                        const t = el.getAttribute("title");
                        el.setAttribute("data-tt", t);
                        el.setAttribute("aria-label", t);
                        el.removeAttribute("title");
                    }});
                }};
                applyTooltips();
                new MutationObserver(applyTooltips).observe(
                    document.body, {{ childList: true, subtree: true }}
                );
            }});
        </script>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>
"""

_HEADER_DATE = pd.Timestamp.now().strftime("%B %Y").upper()

CAMPAIGN_OPTIONS = [
    {"label": CAMPAIGN_LABELS[c].upper(), "value": c}
    for c in CAMPAIGN_LABELS
]

app.layout = html.Div([
    html.Div([

        # ── Row 1: dynamic campaign title on the left, date + logo right ──
        html.Div([
            html.Div(id="campaign-title-banner"),
            html.Div([
                html.Div(
                    html.Img(
                        src=app.get_asset_url("logo-right.png"),
                        alt="Assemble",
                        style={"height": "42px", "display": "block"},
                    ),
                    style={"display": "flex", "justifyContent": "flex-end"},
                ),
                html.P(_HEADER_DATE, style={
                    "fontFamily": FONT_TITLE,
                    "fontSize": "36px",
                    "letterSpacing": "4px",
                    "color": ACCENT2,
                    "margin": "12px 0 0 0",
                    "textAlign": "right",
                    "lineHeight": "1",
                }),
            ]),
        ], style={
            "display": "flex",
            "justifyContent": "space-between",
            "alignItems": "flex-end",
            "marginBottom": "20px",
            "paddingBottom": "20px",
            "borderBottom": f"1px solid {BORDER}",
        }),

        # ── Row 2: filter controls (campaign + goal) ─────────────────────
        html.Div([
            # Campaign selector
            html.Div([
                html.P("CAMPAIGN",
                    className="tt",
                    title=(
                        "Switch which campaign this dashboard reflects. All metrics, "
                        "charts, and projections recalculate against the selected "
                        "campaign's transactions and default goal."
                    ),
                    style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "letterSpacing": "2px",
                        "color": LABEL_COLOR,
                        "margin": "0 0 6px 0",
                        "borderBottom": f"1px dotted {LABEL_COLOR}",
                        "paddingBottom": "2px",
                        "display": "inline-block",
                        "width": "fit-content",
                    },
                ),
                dcc.Dropdown(
                    id="campaign-dropdown",
                    options=CAMPAIGN_OPTIONS,
                    value=DEFAULT_CAMPAIGN,
                    clearable=False,
                    searchable=False,
                    style={
                        "width": "280px",
                        "fontFamily": FONT_MONO,
                        "fontSize": "13px",
                    },
                ),
            ]),

            # Monthly-goal override — free-form, in dollars. Any change here
            # re-runs the callback and recomputes every metric + figure. On
            # campaign switch, the value snaps back to that campaign's stored
            # default (see the goal-reset callback below).
            html.Div([
                html.P("MONTHLY GOAL",
                    className="tt",
                    title=(
                        "Custom monthly fundraising target for this campaign. Editing "
                        "the value re-scales the goal-pace curve, expected pace amounts, "
                        "and % ahead/behind indicators — but the projected total is "
                        "goal-independent (it's a forecast of what you'll actually raise)."
                    ),
                    style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "10px",
                        "letterSpacing": "2px",
                        "color": LABEL_COLOR,
                        "margin": "0 0 6px 0",
                        "borderBottom": f"1px dotted {LABEL_COLOR}",
                        "paddingBottom": "2px",
                        "display": "inline-block",
                        "width": "fit-content",
                    },
                ),
                html.Div([
                    html.Span("$", style={
                        "fontFamily": FONT_MONO,
                        "fontSize": "14px",
                        "color": TITLE_COLOR,
                        "padding": "0 8px 0 12px",
                        "background": SURFACE,
                        "border": f"1px solid {BORDER}",
                        "borderRight": "none",
                        "borderRadius": "8px 0 0 8px",
                        "display": "inline-flex",
                        "alignItems": "center",
                        "height": "36px",
                    }),
                    dcc.Input(
                        id="goal-input",
                        type="number",
                        min=0,
                        step=10000,
                        value=CAMPAIGN_GOALS[DEFAULT_CAMPAIGN],
                        style={
                            "width": "180px",
                            "height": "36px",
                            "fontFamily": FONT_MONO,
                            "fontSize": "14px",
                            "color": TEXT,
                            "background": SURFACE,
                            "border": f"1px solid {BORDER}",
                            "borderLeft": "none",
                            "borderRadius": "0 8px 8px 0",
                            "padding": "0 10px",
                            "outline": "none",
                        },
                    ),
                ], style={"display": "flex", "alignItems": "center"}),
            ], style={"marginLeft": "24px"}),
        ], style={
            "display": "flex",
            "alignItems": "flex-end",
            "marginBottom": "28px",
        }),

        # ── Dynamic content, rebuilt on every dropdown / goal change ─────
        html.Div(id="dashboard-content"),

    ], style={
        "maxWidth": "1400px",
        "margin": "0 auto",
        "padding": "36px 28px",
        "fontFamily": FONT_BODY,
        "color": TEXT,
    }),
], style={"minHeight": "100vh", "background": BG})


# Callback chain:
#   1. Campaign dropdown changes → reset goal input to that campaign's default.
#   2. Goal input changes (either from the reset above, or from user typing) →
#      rebuild the entire dashboard against the new goal.
# Campaign is read as State in step 2, so it doesn't independently trigger a
# render — the goal update from step 1 does that instead. Net effect: exactly
# one render per user action, whether that action is a dropdown pick or a goal
# edit.
@app.callback(
    Output("campaign-title-banner", "children"),
    Input("campaign-dropdown", "value"),
)
def _set_title(campaign):
    return [
        html.P(CAMPAIGN_LABELS[campaign].upper(), style={
            "fontFamily": FONT_TITLE,
            "fontSize": "45px",
            "letterSpacing": "6px",
            "color": TITLE_COLOR,
            "lineHeight": "1",
            "margin": "0",
        }),
        html.P("Fundraising Tracker", style={
            "fontFamily": FONT_MONO,
            "fontSize": "16px",
            "letterSpacing": "1.5px",
            "color": LABEL_COLOR,
            "margin": "8px 0 0 0",
        }),
    ]


@app.callback(
    Output("goal-input", "value"),
    Input("campaign-dropdown", "value"),
)
def _reset_goal(campaign):
    return CAMPAIGN_GOALS[campaign]


@app.callback(
    Output("dashboard-content", "children"),
    Input("goal-input", "value"),
    State("campaign-dropdown", "value"),
)
def _render(goal, campaign):
    return build_dashboard(campaign, monthly_goal=goal)


# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local dev: listen on all interfaces so it works in a Notion iframe on
    # a LAN device. Render provides PORT; use it if set (gunicorn handles
    # this in production, but this branch is used only for `python file.py`).
    _port = int(os.environ.get("PORT", "8050"))
    app.run(host="0.0.0.0", port=_port, debug=True)
