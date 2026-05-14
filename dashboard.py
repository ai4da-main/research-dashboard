import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Funding Opportunities Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSV_PATH = Path(__file__).parent / "research.csv"


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Keep only meaningful columns and rows.
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df = df.dropna(how="all")
    df = df[df["Grant Name"].notna()].copy()

    for col in df.columns:
        df[col] = df[col].astype(str).replace({"nan": ""}).str.strip()

    df["Funding Amount Parsed"] = df["Maximum Funding"].apply(parse_funding_amount)
    df["Funding Rank"] = df["Funding Amount Parsed"].rank(
        method="dense", ascending=False, na_option="bottom"
    ).astype(int)
    df["Deadline Group"] = df["Proposal Deadline"].apply(classify_deadline)
    df["Deadline Date Parsed"] = df["Proposal Deadline"].apply(parse_first_deadline_date)
    df["Deadline Sort"] = df["Deadline Date Parsed"].fillna(pd.Timestamp.max)
    df["Partner Required"] = df["Industry Partner Needed?"].apply(classify_partner)

    return df.sort_values(
        by=["Funding Amount Parsed", "Deadline Sort", "Grant Name"],
        ascending=[False, True, True],
        na_position="last",
    )


def parse_funding_amount(value: str) -> float:
    """Extract the largest stated funding amount from text.

    Examples handled:
    - "$1,000,000/year"
    - "Up to $9M CAD"
    - "$100,000-$650,000 per year"
    - "Small grants $5,000-$15,000; large grants up to $100,000"

    The dashboard ranks by the largest explicit dollar amount mentioned.
    """
    if not isinstance(value, str) or not value.strip():
        return np.nan

    text = value.replace("–", "-").replace("—", "-")
    if "not specified" in text.lower():
        return np.nan

    matches = re.findall(
        r"\$\s*([0-9][0-9,]*(?:\.\d+)?)\s*(M|million|K|thousand)?",
        text,
        flags=re.IGNORECASE,
    )

    amounts = []
    for number, suffix in matches:
        amount = float(number.replace(",", ""))
        suffix = suffix.lower()
        if suffix in {"m", "million"}:
            amount *= 1_000_000
        elif suffix in {"k", "thousand"}:
            amount *= 1_000
        amounts.append(amount)

    return max(amounts) if amounts else np.nan


def classify_deadline(value: str) -> str:
    text = str(value).lower().strip()
    no_fixed_deadline_terms = [
        "no deadline",
        "continuous",
        "rolling",
        "year round",
        "year-round",
        "open intake",
        "open in",
        "not specified",
        "tbd",
    ]
    if not text:
        return "No fixed deadline / rolling"
    if any(term in text for term in no_fixed_deadline_terms):
        return "No fixed deadline / rolling"
    return "Has deadline"


def classify_partner(value: str) -> str:
    text = str(value).lower()
    if any(word in text for word in ["required", "yes"]):
        return "Required"
    if any(word in text for word in ["not required", "no"]):
        return "Not required"
    return "Check details"


def parse_first_deadline_date(value: str):
    """Best-effort parser for visible 2026/2027-style deadline dates."""
    text = str(value).replace("\n", "; ")
    current_year = datetime.now().year
    months = (
        "January|February|March|April|May|June|July|August|"
        "September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )

    patterns = [
        rf"\b({months})\.?\s+([0-9]{{1,2}}),?\s+([0-9]{{4}})\b",
        rf"\b([0-9]{{1,2}})\s+({months})\.?\s+([0-9]{{4}})\b",
        rf"\b({months})\.?\s+([0-9]{{1,2}})\b",
    ]

    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                parts = match.groups()
                if len(parts) == 3 and parts[0].isalpha():
                    date_text = f"{parts[0]} {parts[1]} {parts[2]}"
                elif len(parts) == 3:
                    date_text = f"{parts[1]} {parts[0]} {parts[2]}"
                else:
                    date_text = f"{parts[0]} {parts[1]} {current_year}"
                candidates.append(pd.to_datetime(date_text, errors="coerce"))
            except Exception:
                continue

    candidates = [d for d in candidates if pd.notna(d)]
    return min(candidates) if candidates else pd.NaT


def format_money(value):
    if pd.isna(value):
        return "Not specified"
    if value >= 1_000_000:
        return f"${value/1_000_000:,.1f}M"
    return f"${value:,.0f}"


def linkify(url: str) -> str:
    if isinstance(url, str) and url.startswith("http"):
        return f"[Open source]({url})"
    return ""


# ----------------------------- UI -----------------------------

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 1.25rem;
        background: linear-gradient(135deg, rgba(49, 130, 206, 0.12), rgba(56, 178, 172, 0.10));
        border: 1px solid rgba(49, 130, 206, 0.18);
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin-bottom: 0.25rem;}
    .small-note {color: #667085; font-size: 0.9rem;}
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #EAECF0;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Funding Opportunities Dashboard</h1>
        <p class="small-note">
            Ranked by the largest stated funding amount, with opportunities separated by deadline status.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not CSV_PATH.exists():
    st.error("Could not find research.csv. Place research.csv in the same folder as app.py.")
    st.stop()

raw_df = load_data(CSV_PATH)

with st.sidebar:
    st.header("Filters")

    deadline_options = sorted(raw_df["Deadline Group"].dropna().unique())
    selected_deadline_groups = st.multiselect(
        "Deadline group",
        deadline_options,
        default=deadline_options,
    )

    partner_options = sorted(raw_df["Partner Required"].dropna().unique())
    selected_partner = st.multiselect(
        "Industry partner",
        partner_options,
        default=partner_options,
    )

    subjects = sorted([s for s in raw_df["Subject"].dropna().unique() if s])
    selected_subjects = st.multiselect("Subject", subjects)

    search = st.text_input("Search grant name or subject")

    min_funding = st.slider(
        "Minimum parsed funding amount",
        min_value=0,
        max_value=int(np.nanmax(raw_df["Funding Amount Parsed"].fillna(0))) if len(raw_df) else 0,
        value=0,
        step=5000,
        format="$%d",
    )

    top_n = st.slider("Number of ranked opportunities to show", 5, 50, 20)

filtered = raw_df[
    raw_df["Deadline Group"].isin(selected_deadline_groups)
    & raw_df["Partner Required"].isin(selected_partner)
    & (raw_df["Funding Amount Parsed"].fillna(0) >= min_funding)
].copy()

if selected_subjects:
    filtered = filtered[filtered["Subject"].isin(selected_subjects)]

if search:
    mask = (
        filtered["Grant Name"].str.contains(search, case=False, na=False)
        | filtered["Subject"].str.contains(search, case=False, na=False)
    )
    filtered = filtered[mask]

ranked = filtered.sort_values(
    by=["Funding Amount Parsed", "Deadline Sort", "Grant Name"],
    ascending=[False, True, True],
    na_position="last",
).head(top_n)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Opportunities", f"{len(filtered):,}")
col2.metric("With deadline", f"{(filtered['Deadline Group'] == 'Has deadline').sum():,}")
col3.metric("No fixed deadline", f"{(filtered['Deadline Group'] != 'Has deadline').sum():,}")
col4.metric(
    "Largest parsed funding",
    format_money(filtered["Funding Amount Parsed"].max()) if len(filtered) else "N/A",
)

st.divider()

chart_df = ranked.dropna(subset=["Funding Amount Parsed"]).copy()
if not chart_df.empty:
    fig = px.bar(
        chart_df.sort_values("Funding Amount Parsed", ascending=True),
        x="Funding Amount Parsed",
        y="Grant Name",
        color="Deadline Group",
        orientation="h",
        hover_data={
            "Maximum Funding": True,
            "Proposal Deadline": True,
            "Industry Partner Needed?": True,
            "Funding Amount Parsed": ":,.0f",
        },
        title="Top Opportunities Ranked by Available Funding",
    )
    fig.update_layout(
        height=max(450, 32 * len(chart_df)),
        xaxis_title="Parsed funding amount",
        yaxis_title="",
        legend_title="Deadline status",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No ranked funding amounts are available for the current filters.")

st.subheader("Ranked Opportunities")
display = ranked.copy()
display["Parsed Funding"] = display["Funding Amount Parsed"].apply(format_money)
display["Source"] = display["Source URL"].apply(linkify)
display["Parsed Deadline Date"] = display["Deadline Date Parsed"].dt.strftime("%Y-%m-%d")
display["Parsed Deadline Date"] = display["Parsed Deadline Date"].fillna("")

cols = [
    "Funding Rank",
    "Grant Name",
    "Subject",
    "Deadline Group",
    "Proposal Deadline",
    "Parsed Deadline Date",
    "Maximum Funding",
    "Parsed Funding",
    "Partner Required",
    "Industry Partner Needed?",
    "Source",
]

st.dataframe(
    display[cols],
    use_container_width=True,
    hide_index=True,
    column_config={
        "Source": st.column_config.LinkColumn("Source"),
        "Maximum Funding": st.column_config.TextColumn(width="medium"),
        "Proposal Deadline": st.column_config.TextColumn(width="medium"),
        "Subject": st.column_config.TextColumn(width="large"),
    },
)

st.divider()

with_deadline, no_deadline = st.tabs(["Has deadline", "No fixed deadline / rolling"])

with with_deadline:
    st.subheader("Funding Opportunities With Deadlines")
    has_deadline_df = filtered[filtered["Deadline Group"] == "Has deadline"].sort_values(
        by=["Deadline Sort", "Funding Amount Parsed"], ascending=[True, False]
    )
    st.dataframe(
        has_deadline_df.assign(
            **{
                "Parsed Funding": has_deadline_df["Funding Amount Parsed"].apply(format_money),
                "Source": has_deadline_df["Source URL"].apply(linkify),
            }
        )[
            [
                "Grant Name",
                "Proposal Deadline",
                "Maximum Funding",
                "Parsed Funding",
                "Subject",
                "Partner Required",
                "Source",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )

with no_deadline:
    st.subheader("Funding Opportunities With No Fixed Deadline or Rolling Intake")
    no_deadline_df = filtered[filtered["Deadline Group"] != "Has deadline"].sort_values(
        by=["Funding Amount Parsed", "Grant Name"], ascending=[False, True]
    )
    st.dataframe(
        no_deadline_df.assign(
            **{
                "Parsed Funding": no_deadline_df["Funding Amount Parsed"].apply(format_money),
                "Source": no_deadline_df["Source URL"].apply(linkify),
            }
        )[
            [
                "Grant Name",
                "Proposal Deadline",
                "Maximum Funding",
                "Parsed Funding",
                "Subject",
                "Partner Required",
                "Source",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={"Source": st.column_config.LinkColumn("Source")},
    )

st.caption(
    "Note: Parsed funding is a best-effort extraction of the largest dollar amount mentioned in the Maximum Funding field. "
    "Always confirm final eligibility, deadlines, and funding rules from the source link."
)
