import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import time
import numpy as np
from sklearn.cluster import KMeans

# --- Configuration & Data Loading ---
st.set_page_config(
    layout="wide",
    page_title="Matiks Mini-Dashboard 🎮",
    initial_sidebar_state="expanded" # Keep sidebar expanded by default
)

@st.cache_data
def load_and_process_data():
    """Loads the data and performs initial date conversions."""
    df = pd.read_csv('matiks_data.csv')
    df['Signup_Date'] = pd.to_datetime(df['Signup_Date'])
    df['Last_Login'] = pd.to_datetime(df['Last_Login'])
    return df

df = load_and_process_data()

# --- Caching for heavy computations ---

@st.cache_data
def compute_cohort_retention(filtered_df):
    cohort_df = filtered_df.copy()
    cohort_df['CohortMonth'] = cohort_df['Signup_Date'].dt.to_period('M')
    last_login_period = cohort_df['Last_Login'].dt.to_period('M')
    cohort_index = last_login_period - cohort_df['CohortMonth']
    cohort_df['CohortIndex'] = cohort_index.apply(lambda x: x.n if pd.notnull(x) else None)
    cohort_pivot = cohort_df.groupby(['CohortMonth', 'CohortIndex'])['User_ID'].nunique().unstack(fill_value=0)
    cohort_size = cohort_pivot.iloc[:,0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100
    return retention

@st.cache_data
def compute_kmeans_cluster(seg_df, k=3):
    cluster_data = seg_df[['User_ID', 'Total_Revenue_USD', 'Total_Play_Sessions']].dropna()
    cluster_data = cluster_data.groupby('User_ID').agg({'Total_Revenue_USD': 'sum', 'Total_Play_Sessions': 'sum'})
    if len(cluster_data) < k:
        return None
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_data['Cluster'] = kmeans.fit_predict(cluster_data[['Total_Revenue_USD', 'Total_Play_Sessions']])
    return cluster_data

# --- Sidebar Filters ---
st.sidebar.header("Dashboard Filters")
st.sidebar.markdown("<span style='color:#4F8BF9;font-weight:bold;'>Tip:</span> Use filters to explore user segments and trends.", unsafe_allow_html=True)

# Date Range Filter
st.sidebar.markdown("**Date Range**")
min_date = df['Signup_Date'].min().date()
max_date = df['Signup_Date'].max().date()
selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
elif len(selected_date_range) == 1:
    start_date, end_date = selected_date_range[0], selected_date_range[0]
else:
    start_date, end_date = min_date, max_date

# Device Type Filter
st.sidebar.markdown("**Device Type**")
all_device_types = df['Device_Type'].unique().tolist()
selected_devices = st.sidebar.multiselect(
    "Select Device Type(s)",
    options=all_device_types,
    default=all_device_types,
    help="Filter by device to compare user behavior across platforms."
)

# Subscription Tier Filter
if 'Subscription_Tier' in df:
    st.sidebar.markdown("**Subscription Tier**")
    all_tiers = df['Subscription_Tier'].dropna().unique().tolist()
    selected_tiers = st.sidebar.multiselect(
        "Select Subscription Tier(s)",
        options=all_tiers,
        default=all_tiers,
        help="Filter by user subscription segment."
    )
else:
    selected_tiers = None

# Preferred Game Mode Filter
if 'Preferred_Game_Mode' in df:
    st.sidebar.markdown("**Game Mode**")
    all_modes = df['Preferred_Game_Mode'].dropna().unique().tolist()
    selected_modes = st.sidebar.multiselect(
        "Select Game Mode(s)",
        options=all_modes,
        default=all_modes,
        help="Filter by preferred game mode."
    )
else:
    selected_modes = None

# Apply Filters
filtered_df = df[
    df['Device_Type'].isin(selected_devices) &
    (df['Signup_Date'].dt.date >= start_date) &
    (df['Signup_Date'].dt.date <= end_date)
]
if selected_tiers is not None:
    filtered_df = filtered_df[filtered_df['Subscription_Tier'].isin(selected_tiers)]
if selected_modes is not None:
    filtered_df = filtered_df[filtered_df['Preferred_Game_Mode'].isin(selected_modes)]
filtered_df = filtered_df.copy()

# --- Dashboard Header ---
st.title("Matiks Gaming Analytics 🎮")
st.markdown(f"Last updated: **{datetime.now().strftime('%Y-%m-%d %H:%M %p')}**")
st.markdown("---")

# --- Key Metrics Row ---
col1, col2, col3, col4, col5 = st.columns(5)

# DAU
total_users = filtered_df['User_ID'].nunique()
daily_users = filtered_df.groupby(filtered_df['Signup_Date'].dt.date)['User_ID'].nunique()
avg_dau = daily_users.mean()
# Revenue
total_revenue = filtered_df['Total_Revenue_USD'].sum()
# ARPU
arpu = total_revenue / total_users if total_users > 0 else 0
# Avg Session Length
avg_session_length = filtered_df['Avg_Session_Duration_Min'].mean() if 'Avg_Session_Duration_Min' in filtered_df else None
# Session Frequency
avg_session_freq = filtered_df['Total_Play_Sessions'].mean() if 'Total_Play_Sessions' in filtered_df else None

col1.metric("Total Users", f"{total_users:,}", help="Unique users in the selected period.")
col2.metric("Avg. DAU", f"{avg_dau:.0f}", help="Average daily active users.")
col3.metric("ARPU", f"${arpu:.2f}", help="Average revenue per user.")
col4.metric("Avg. Session Length (min)", f"{avg_session_length:.2f}" if avg_session_length is not None else "N/A", help="Average session duration.")
col5.metric("Avg. Session Frequency", f"{avg_session_freq:.2f}" if avg_session_freq is not None else "N/A", help="Average play sessions per user.")

# --- Quick Insights Row ---
insight_col1, insight_col2 = st.columns(2)
with insight_col1:
    st.info(f"**Peak DAU:** {int(daily_users.max()):,} on {daily_users.idxmax()}")
with insight_col2:
    st.success(f"**Top Revenue Device:** {filtered_df.groupby('Device_Type')['Total_Revenue_USD'].sum().idxmax()}\n\n**Top Segment:** {filtered_df.groupby('Subscription_Tier')['Total_Revenue_USD'].sum().idxmax() if 'Subscription_Tier' in filtered_df else 'N/A'}")

st.markdown("---")

# --- Combined Visualization Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Activity Overview", "💰 Revenue Insights", "📈 User Behavior", "🧩 Segmentation & Clustering", "🗓️ Cohort Analysis"
])

with tab1:
    st.header("User Activity & Device Distribution")
    activity_col1, activity_col2 = st.columns([2, 1])
    with activity_col1:
        # DAU Line
        fig_dau = px.line(
            daily_users.reset_index(), x='Signup_Date', y='User_ID',
            title="Daily Active Users (DAU)",
            labels={'User_ID': 'Users', 'Signup_Date': 'Date'},
            color_discrete_sequence=px.colors.qualitative.Plotly,
            height=500,
            width=1000
        )
        fig_dau.update_layout(
            hovermode="x unified",
            xaxis_tickangle=-45,
            font=dict(size=15),
            margin=dict(l=20, r=10, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        # In DAU chart, add a marker for the max DAU day
        max_dau_day = daily_users.idxmax()
        max_dau_value = daily_users.max()
        fig_dau.add_scatter(x=[max_dau_day], y=[max_dau_value], mode='markers+text',
            marker=dict(size=12, color='#FF5733'),
            text=[f"Peak: {int(max_dau_value)}"], textposition="top center",
            name="Peak DAU")
        st.plotly_chart(fig_dau, use_container_width=True)
        # WAU/MAU
        weekly_users = filtered_df.groupby(filtered_df['Signup_Date'].dt.to_period('W'))['User_ID'].nunique()
        monthly_users = filtered_df.groupby(filtered_df['Signup_Date'].dt.to_period('M'))['User_ID'].nunique()
        wau_df = weekly_users.reset_index()
        wau_df['Signup_Date'] = wau_df['Signup_Date'].astype(str)
        fig_wau = px.line(
            wau_df, x='Signup_Date', y='User_ID',
            title="Weekly Active Users (WAU)",
            labels={'User_ID': 'Users', 'Signup_Date': 'Week'},
            color_discrete_sequence=px.colors.qualitative.Set1,
            height=400,
            width=900
        )
        fig_wau.update_layout(
            font=dict(size=14),
            margin=dict(l=20, r=10, t=30, b=30),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_wau, use_container_width=True)
        mau_df = monthly_users.reset_index()
        mau_df['Signup_Date'] = mau_df['Signup_Date'].astype(str)
        fig_mau = px.line(
            mau_df, x='Signup_Date', y='User_ID',
            title="Monthly Active Users (MAU)",
            labels={'User_ID': 'Users', 'Signup_Date': 'Month'},
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=400,
            width=900
        )
        fig_mau.update_layout(
            font=dict(size=14),
            margin=dict(l=20, r=10, t=30, b=30),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_mau, use_container_width=True)
    with activity_col2:
        # Device Share
        fig_device_share = px.pie(
            filtered_df, names='Device_Type',
            title="User Device Share",
            height=300,
            width=300,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_device_share.update_layout(
            margin=dict(l=10, r=10, t=40, b=10),
            legend=dict(font=dict(size=11))
        )
        st.plotly_chart(fig_device_share, use_container_width=True)

with tab2:
    st.header("Revenue Visualizations & Breakdown")
    # Revenue over time (line)
    revenue_daily = filtered_df.groupby(filtered_df['Signup_Date'].dt.date)['Total_Revenue_USD'].sum().reset_index()
    fig_rev_line = px.line(
        revenue_daily, x='Signup_Date', y='Total_Revenue_USD',
        title="Revenue Over Time (Daily)",
        labels={'Total_Revenue_USD': 'Revenue (USD)', 'Signup_Date': 'Date'},
        color_discrete_sequence=px.colors.qualitative.Plotly,
        height=500,
        width=1000
    )
    fig_rev_line.update_layout(
        font=dict(size=15),
        margin=dict(l=20, r=10, t=40, b=40),
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_rev_line, use_container_width=True)
    # Revenue by device
    fig_rev_device = px.bar(
        filtered_df.groupby('Device_Type')['Total_Revenue_USD'].sum().reset_index(),
        x='Device_Type', y='Total_Revenue_USD',
        title="Revenue by Device",
        color='Device_Type',
        color_discrete_sequence=px.colors.qualitative.Set1,
        height=350,
        width=700
    )
    # Add clear axis labels and color for revenue by device
    fig_rev_device.update_traces(marker_color='#4F8BF9')
    fig_rev_device.update_layout(
        xaxis_title="Device Type",
        yaxis_title="Revenue (USD)",
        title_font_color="#4F8BF9",
        font=dict(size=13)
    )
    st.plotly_chart(fig_rev_device, use_container_width=True)
    # Revenue by segment
    if 'Subscription_Tier' in filtered_df:
        fig_rev_segment = px.bar(
            filtered_df.groupby('Subscription_Tier')['Total_Revenue_USD'].sum().reset_index(),
            x='Subscription_Tier', y='Total_Revenue_USD',
            title="Revenue by User Segment (Subscription Tier)",
            color='Subscription_Tier',
            color_discrete_sequence=px.colors.qualitative.Set2,
            height=300,
            width=600
        )
        fig_rev_segment.update_layout(
            font=dict(size=12),
            margin=dict(l=20, r=10, t=30, b=30)
        )
        st.plotly_chart(fig_rev_segment, use_container_width=True)
    # Revenue by game mode
    if 'Preferred_Game_Mode' in filtered_df:
        fig_rev_gamemode = px.bar(
            filtered_df.groupby('Preferred_Game_Mode')['Total_Revenue_USD'].sum().reset_index(),
            x='Preferred_Game_Mode', y='Total_Revenue_USD',
            title="Revenue by Game Mode",
            color='Preferred_Game_Mode',
            color_discrete_sequence=px.colors.qualitative.Set3,
            height=300,
            width=600
        )
        fig_rev_gamemode.update_layout(
            font=dict(size=12),
            margin=dict(l=20, r=10, t=30, b=30)
        )
        st.plotly_chart(fig_rev_gamemode, use_container_width=True)

with tab3:
    st.header("User Behavior & Engagement")
    behavior_col1, behavior_col2 = st.columns(2)
    with behavior_col1:
        # User Funnel
        funnel_data = pd.DataFrame({
            'Stage': ['Onboarded', 'First Play Session', 'Repeat Play Sessions'],
            'Users': [
                filtered_df['User_ID'].nunique(),
                filtered_df[filtered_df['Total_Play_Sessions'] > 0]['User_ID'].nunique(),
                filtered_df[filtered_df['Total_Play_Sessions'] > 1]['User_ID'].nunique()
            ]
        })
        fig_funnel = px.funnel(
            funnel_data, x='Users', y='Stage',
            title="User Engagement Funnel",
            height=250,
            width=400,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_funnel.update_layout(
            font=dict(size=12),
            margin=dict(l=20, r=10, t=30, b=30)
        )
        st.plotly_chart(fig_funnel, use_container_width=True)
        # Session Frequency
        if avg_session_freq is not None:
            st.metric("Avg. Session Frequency", f"{avg_session_freq:.2f}")
    with behavior_col2:
        # Session Length
        if avg_session_length is not None:
            st.metric("Avg. Session Length (min)", f"{avg_session_length:.2f}")
        # Add more KPIs if needed

with tab4:
    st.header("User Segmentation & Clustering")
    # Filters for segment comparison
    segment_col1, segment_col2, segment_col3 = st.columns(3)
    # Segment filter
    if 'Subscription_Tier' in filtered_df:
        selected_tiers = segment_col1.multiselect(
            "Filter by Subscription Tier", options=filtered_df['Subscription_Tier'].unique(),
            default=filtered_df['Subscription_Tier'].unique()
        )
    else:
        selected_tiers = None
    # Game mode filter
    if 'Preferred_Game_Mode' in filtered_df:
        selected_modes = segment_col2.multiselect(
            "Filter by Game Mode", options=filtered_df['Preferred_Game_Mode'].unique(),
            default=filtered_df['Preferred_Game_Mode'].unique()
        )
    else:
        selected_modes = None
    # Device filter (redundant, but for comparison)
    selected_devices_seg = segment_col3.multiselect(
        "Filter by Device (Segmentation)", options=all_device_types,
        default=all_device_types
    )
    seg_mask = (
        (filtered_df['Device_Type'].isin(selected_devices_seg))
    )
    if selected_tiers is not None:
        seg_mask &= filtered_df['Subscription_Tier'].isin(selected_tiers)
    if selected_modes is not None:
        seg_mask &= filtered_df['Preferred_Game_Mode'].isin(selected_modes)
    seg_df = filtered_df[seg_mask]
    # KPIs for selected segment
    st.metric("Segment Users", seg_df['User_ID'].nunique())
    st.metric("Segment Revenue", f"${seg_df['Total_Revenue_USD'].sum():,.0f}")
    # Clustering: Revenue vs Frequency
    if len(seg_df) > 10:
        with st.spinner("Clustering users (K-means)..."):
            cluster_data = compute_kmeans_cluster(seg_df, k=3)
            if cluster_data is not None:
                fig_cluster = px.scatter(
                    cluster_data, x='Total_Play_Sessions', y='Total_Revenue_USD',
                    color='Cluster',
                    title="User Clusters: Revenue vs Frequency (K-means)",
                    labels={'Total_Play_Sessions': 'Session Frequency', 'Total_Revenue_USD': 'Revenue'},
                    color_continuous_scale=px.colors.sequential.Viridis,
                    height=300,
                    width=500
                )
                fig_cluster.update_layout(
                    font=dict(size=12),
                    margin=dict(l=20, r=10, t=30, b=30)
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
            else:
                st.info("Not enough data for clustering.")
    else:
        st.info("Not enough data for clustering.")

with tab5:
    st.header("Cohort Analysis by Signup Date")
    # Cohort analysis: Retention over time (line chart)
    if len(filtered_df) > 30:
        with st.spinner("Calculating cohort retention trends..."):
            retention = compute_cohort_retention(filtered_df)
            # Prepare data for line chart: each cohort as a line, x=months since signup, y=retention %
            retention_long = retention.reset_index().melt(id_vars='CohortMonth', var_name='Months_Since_Signup', value_name='Retention')
            retention_long = retention_long.dropna(subset=['Retention'])
            retention_long['Months_Since_Signup'] = retention_long['Months_Since_Signup'].astype(int)
            fig_ret_line = px.line(
                retention_long,
                x='Months_Since_Signup', y='Retention', color='CohortMonth',
                title="Cohort Retention Trends (Monthly)",
                labels={'Months_Since_Signup': 'Months Since Signup', 'Retention': 'Retention (%)', 'CohortMonth': 'Signup Cohort'},
                height=350, width=700,
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            fig_ret_line.update_layout(
                font=dict(size=12),
                margin=dict(l=20, r=10, t=40, b=40),
                legend_title_text='Cohort',
                xaxis=dict(dtick=1)
            )
            st.plotly_chart(fig_ret_line, use_container_width=True)
    else:
        st.info("Not enough data for cohort analysis.")

st.markdown("---")

# --- Expandable Raw Data Table ---
with st.expander("🔍 **View Raw Data Preview**"):
    st.dataframe(filtered_df.head(100), height=300, use_container_width=True)
    st.markdown(f"Displaying top 100 rows of filtered data. Total rows: **{len(filtered_df):,}**")