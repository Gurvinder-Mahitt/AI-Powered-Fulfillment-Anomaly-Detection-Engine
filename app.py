import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from ai_assisted_eda_agent import (
    run_anomaly_pipeline,
    run_generic_pipeline,
    extract_anomaly_context,
    generate_openrouter_narrative,
    generate_generic_narrative,
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="AI Assisted EDA — Anomaly Intelligence Engine",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM STYLING
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .welcome-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    .welcome-card h2 {
        color: #E2E8F0;
        font-family: 'Inter', sans-serif;
    }
    .welcome-card p {
        color: #94A3B8;
        font-size: 1.1rem;
    }
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3B82F6, #6366F1);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    .mode-indicator {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
        margin: 0.5rem 0;
    }
    .stAlert {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📦 AI Assisted EDA — Anomaly Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated detection of operational anomalies with AI-powered root-cause diagnostics. Upload your dataset to begin.</div>', unsafe_allow_html=True)

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Engine Configuration")

# --- API Key (auto-loaded from secrets or environment) ---
openrouter_key = ""
try:
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")
except Exception:
    pass
if not openrouter_key:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

if openrouter_key:
    st.sidebar.markdown('<div class="mode-indicator">🤖 AI Narratives Active</div>', unsafe_allow_html=True)
else:
    st.sidebar.info("🔧 Rule Engine Active (No API key configured)")

# --- Dataset Mode ---
st.sidebar.subheader("📂 Dataset")

# Auto-detect bundled sample data
SAMPLE_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OLIST Dataset")
has_sample_data = os.path.exists(os.path.join(SAMPLE_DATA_DIR, "olist_orders_dataset.csv"))

mode_options = []
if has_sample_data:
    mode_options.append("📊 Sample Analysis (E-Commerce Demo)")
mode_options.append("📄 Upload Your Own Dataset")

dataset_mode = st.sidebar.radio(
    "Choose Mode",
    options=mode_options,
    help="View the sample e-commerce analysis to understand the project, or upload your own dataset."
)

uploaded_generic = None
if dataset_mode == "📄 Upload Your Own Dataset":
    uploaded_generic = st.sidebar.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        accept_multiple_files=False,
        help="Upload a CSV with numeric columns. The engine will auto-detect features for anomaly detection."
    )

# --- Anomaly Thresholds ---
st.sidebar.subheader("🎚️ Anomaly Thresholds")
contamination = st.sidebar.slider(
    "Isolation Forest Contamination Rate",
    min_value=0.005, max_value=0.05, value=0.01, step=0.005, format="%.3f",
    help="Expected proportion of anomalies in the dataset. Lower = fewer flags."
)
iqr_multiplier = st.sidebar.slider(
    "IQR Outlier Multiplier",
    min_value=1.0, max_value=3.0, value=1.5, step=0.1,
    help="Higher values = only extreme outliers flagged."
)

# --- Cache Control ---
if st.sidebar.button("🔄 Clear Cache & Re-run"):
    st.cache_data.clear()
    st.rerun()

# ==========================================
# WELCOME SCREEN (UPLOAD MODE - NO DATA YET)
# ==========================================
def show_welcome_screen():
    """Renders the onboarding screen when upload mode is selected but no file uploaded yet."""
    st.markdown("""
    <div class="welcome-card">
        <h2>📄 Upload Your Own Dataset</h2>
        <p>Upload a CSV file using the sidebar to run anomaly detection on your data.</p>
        <br>
        <span class="feature-badge">Isolation Forest ML</span>
        <span class="feature-badge">IQR Statistical Detection</span>
        <span class="feature-badge">AI Root-Cause Narratives</span>
        <span class="feature-badge">CSV Export</span>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Expected CSV Format")
        st.markdown("""
        Your CSV should contain **numeric columns** for anomaly detection. Example structure:
        
        | Column | Type | Example |
        |--------|------|---------|
        | `id` | String | `ORD-001` |
        | `value` | Numeric | `249.90` |
        | `processing_time` | Numeric | `12.5` |
        | `shipping_cost` | Numeric | `35.00` |
        | `region` | Categorical | `North` |
        
        The engine auto-detects numeric columns and lets you pick which ones to analyze.
        """)
    
    with col2:
        st.markdown("### 🤖 What You Get")
        st.markdown("""
        Upload your data and the engine automatically:
        
        1. **🔍 Detects outliers** using Isolation Forest across your numeric columns
        2. **📐 Flags extremes** with IQR analysis on a target column of your choice
        3. **🧠 Generates AI narratives** explaining each anomaly with root-cause hypotheses
        4. **📥 Exports results** as a downloadable CSV with anomaly flags
        
        > 💡 *Switch to **Sample Analysis** in the sidebar to see a full demo with real e-commerce data first!*
        """)

# ==========================================
# OLIST MODE — FULL PIPELINE (SAMPLE OR UPLOADED)
# ==========================================
def run_olist_mode(data_dir=None, uploaded_files_dict=None):
    """Runs the full OLIST pipeline from disk (sample) or uploaded files."""
    
    @st.cache_data(show_spinner=False)
    def get_olist_data(_data_dir, _files_dict, contam, iqr_mult):
        return run_anomaly_pipeline(data_dir=_data_dir, uploaded_files=_files_dict, contamination=contam, iqr_multiplier=iqr_mult)
    
    with st.spinner("⚡ Running Anomaly Engines (Isolation Forest + IQR + Time-Series)..."):
        try:
            pipeline_data = get_olist_data(data_dir, uploaded_files_dict, contamination, iqr_multiplier)
        except Exception as e:
            st.error(f"❌ Pipeline Error: {e}")
            st.info("Make sure the dataset files have the expected schema.")
            st.stop()

    df_clean = pipeline_data['df_clean']
    df_anomalies = pipeline_data['df_anomalies']
    ts_volume = pipeline_data['ts_volume']
    alerts_df = pipeline_data['alerts_df']

    # State Filter
    all_states = sorted(df_clean['customer_state'].dropna().unique().tolist())
    selected_state = st.sidebar.selectbox("Filter by Customer State", options=["All States"] + all_states)

    if selected_state != "All States":
        filtered_df = df_anomalies[df_anomalies['customer_state'] == selected_state]
    else:
        filtered_df = df_anomalies

    # KPI Metrics
    num_iqr_delays = filtered_df['Anomaly_IQR_Delivery'].sum()
    num_iso_anomalies = filtered_df['Anomaly_IsoForest'].sum()
    num_volume_drops = ts_volume['Anomaly_Volume_Drop'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Orders Analyzed", f"{len(filtered_df):,}")
    with col2:
        st.metric("Delivery Delay Outliers (IQR)", f"{num_iqr_delays:,}", delta="Needs Carrier Review", delta_color="inverse")
    with col3:
        st.metric("Multivariate Glitches (ML)", f"{num_iso_anomalies:,}", delta="Seller Audit", delta_color="inverse")
    with col4:
        st.metric("Volume Drop Dips", f"{num_volume_drops}", delta="Platform Alert", delta_color="inverse")

    st.divider()

    # Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "🚨 Priority Alerts & AI Root-Cause",
        "📊 Interactive Anomaly Analytics",
        "🔍 Data Explorer & Export"
    ])

    # TAB 1: AI ROOT CAUSE ANALYSIS
    with tab1:
        st.subheader("Priority Flagged Anomalies with AI Root-Cause Analysis")
        st.caption("AI narratives automatically generate operational hypotheses and actionable remediation steps.")

        top_iso_anomalies = filtered_df[filtered_df['Anomaly_IsoForest']].sort_values('Anomaly_Score').head(10)

        if top_iso_anomalies.empty:
            st.info("No multivariate anomalies detected under the current filters.")
        else:
            for idx, row in top_iso_anomalies.iterrows():
                order_id = row['order_id']
                val = row['Total_Order_Value']
                state = row['customer_state']
                prep_hrs = row['Prep_Time_Hours']
                deliv_hrs = row['Delivery_Time_Hours']
                
                card_title = f"Order #{order_id} | Value: R${val:.2f} | Location: {state} | Prep: {prep_hrs:.1f}h | Delivery: {deliv_hrs:.1f}h"
                
                with st.expander(card_title, expanded=(idx == top_iso_anomalies.index[0])):
                    ctx = extract_anomaly_context(row, anomaly_type="Multivariate Glitch")
                    narrative = generate_openrouter_narrative(ctx, row=row, api_key=openrouter_key)
                    
                    st.markdown("**AI Root-Cause & Action Plan:**")
                    st.info(narrative)
                    
                    det_col1, det_col2, det_col3, det_col4 = st.columns(4)
                    det_col1.write(f"**Order Value:** R${val:.2f}")
                    det_col2.write(f"**Freight Value:** R${row.get('Total_Freight_Value', 0):.2f}")
                    det_col3.write(f"**Carrier Prep Time:** {prep_hrs:.1f} hrs")
                    det_col4.write(f"**Total Delivery Time:** {deliv_hrs:.1f} hrs")

    # TAB 2: INTERACTIVE ANALYTICS
    with tab2:
        st.subheader("Time-Series Order Volume & Control Limits (3σ)")
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_volume.index, y=ts_volume['Order_Count'], mode='lines', name='Daily Orders', line=dict(color='#3B82F6', width=1.5)))
        fig_ts.add_trace(go.Scatter(x=ts_volume.index, y=ts_volume['Rolling_Mean'], mode='lines', name='7-Day Rolling Mean', line=dict(color='#10B981', width=2)))
        fig_ts.add_trace(go.Scatter(x=ts_volume.index, y=ts_volume['LCL'], mode='lines', name='Lower Control Limit (LCL)', line=dict(color='#EF4444', dash='dash')))
        
        dips = ts_volume[ts_volume['Anomaly_Volume_Drop']]
        if not dips.empty:
            fig_ts.add_trace(go.Scatter(
                x=dips.index, y=dips['Order_Count'],
                mode='markers', name='Volume Drop Anomaly',
                marker=dict(color='#DC2626', size=10, symbol='x')
            ))

        fig_ts.update_layout(
            title="Daily Order Volume Monitoring with Dynamic Control Limits",
            xaxis_title="Purchase Date",
            yaxis_title="Order Volume",
            template="plotly_dark",
            height=400,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        col_plot1, col_plot2 = st.columns(2)

        with col_plot1:
            st.subheader("Regional Delivery Delays by State (IQR)")
            delay_by_state = filtered_df[filtered_df['Anomaly_IQR_Delivery']].groupby('customer_state').size().reset_index(name='Delay_Count').sort_values('Delay_Count', ascending=False)
            
            fig_bar = px.bar(
                delay_by_state.head(10),
                x='customer_state', y='Delay_Count',
                title="Top 10 States by Flagged Delivery Delay Count",
                color='Delay_Count',
                color_continuous_scale='Reds',
                labels={'customer_state': 'Customer State', 'Delay_Count': 'Delayed Orders Count'}
            )
            fig_bar.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_plot2:
            st.subheader("Multivariate Isolation Forest Anomalies")
            fig_scatter = px.scatter(
                filtered_df,
                x='Prep_Time_Hours',
                y='Total_Order_Value',
                color='Anomaly_IsoForest',
                color_discrete_map={True: '#EF4444', False: '#94A3B8'},
                hover_data=['order_id', 'customer_state', 'Delivery_Time_Hours'],
                title="Order Value vs Prep Time (Red = Isolation Forest Flag)",
                labels={'Prep_Time_Hours': 'Prep Time (Hours)', 'Total_Order_Value': 'Order Value (R$)'}
            )
            fig_scatter.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # TAB 3: DATA EXPLORER & EXPORT
    with tab3:
        st.subheader("Operational Anomaly Dataset Explorer")
        
        anomaly_type_filter = st.radio(
            "Select Anomaly Category",
            options=["All Flagged Anomalies", "Multivariate Outliers (Isolation Forest)", "Delivery Delays (IQR)"],
            horizontal=True
        )
        
        if anomaly_type_filter == "Multivariate Outliers (Isolation Forest)":
            display_df = filtered_df[filtered_df['Anomaly_IsoForest']]
        elif anomaly_type_filter == "Delivery Delays (IQR)":
            display_df = filtered_df[filtered_df['Anomaly_IQR_Delivery']]
        else:
            display_df = filtered_df[filtered_df['Anomaly_IsoForest'] | filtered_df['Anomaly_IQR_Delivery']]

        search_query = st.text_input("Search by Order ID or Customer State", placeholder="e.g. SP or order_id...")
        if search_query:
            display_df = display_df[
                display_df['order_id'].astype(str).str.contains(search_query, case=False, na=False) |
                display_df['customer_state'].astype(str).str.contains(search_query, case=False, na=False)
            ]

        st.write(f"Showing **{len(display_df):,}** matching anomaly records:")
        
        cols_to_show = [
            'order_id', 'customer_state', 'Total_Order_Value', 'Prep_Time_Hours',
            'Delivery_Time_Hours', 'Anomaly_IQR_Delivery', 'Anomaly_IsoForest'
        ]
        if 'Anomaly_Score' in display_df.columns:
            cols_to_show.append('Anomaly_Score')

        st.dataframe(display_df[cols_to_show], use_container_width=True, height=400)

        csv_data = display_df[cols_to_show].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Flagged Anomalies CSV",
            data=csv_data,
            file_name="flagged_operational_anomalies.csv",
            mime="text/csv"
        )


# ==========================================
# GENERIC MODE — ANY CSV
# ==========================================
def run_generic_mode(uploaded_file):
    """Runs the generic anomaly detection pipeline on any uploaded CSV."""
    
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read CSV: {e}")
        st.stop()
    
    st.success(f"✅ Loaded dataset: **{len(df):,}** rows × **{len(df.columns)}** columns")
    
    # Show data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)
    
    # Detect numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if len(numeric_cols) < 1:
        st.error("❌ No numeric columns found in the uploaded CSV. Anomaly detection requires at least one numeric column.")
        st.stop()
    
    # Feature selection
    st.sidebar.subheader("🎯 Feature Selection")
    
    selected_features = st.sidebar.multiselect(
        "Numeric columns for Isolation Forest",
        options=numeric_cols,
        default=numeric_cols[:min(5, len(numeric_cols))],
        help="Select 2+ numeric columns for multivariate anomaly detection."
    )
    
    target_col = st.sidebar.selectbox(
        "Target column for IQR analysis (optional)",
        options=["None"] + numeric_cols,
        help="Select a single numeric column for IQR outlier detection."
    )
    target_col = None if target_col == "None" else target_col
    
    group_col = st.sidebar.selectbox(
        "Group-by column for IQR (optional)",
        options=["None"] + categorical_cols,
        help="Group IQR analysis by a categorical column (e.g., region, category)."
    )
    group_col = None if group_col == "None" else group_col
    
    if len(selected_features) < 2:
        st.warning("⚠️ Select at least 2 numeric columns in the sidebar for Isolation Forest detection.")
        st.stop()
    
    # Run pipeline
    with st.spinner("⚡ Running anomaly detection engines..."):
        try:
            results = run_generic_pipeline(
                df, selected_features, target_col, group_col,
                contamination=contamination, iqr_multiplier=iqr_multiplier
            )
        except Exception as e:
            st.error(f"❌ Pipeline Error: {e}")
            st.stop()
    
    df_anomalies = results['df_anomalies']
    alerts_df = results['alerts_df']
    
    # KPI Metrics
    iso_count = df_anomalies['Anomaly_IsoForest'].sum()
    iqr_count = df_anomalies['Anomaly_IQR'].sum() if 'Anomaly_IQR' in df_anomalies.columns else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df_anomalies):,}")
    with col2:
        st.metric("Isolation Forest Outliers", f"{iso_count:,}", delta=f"{iso_count/len(df_anomalies)*100:.1f}%", delta_color="inverse")
    with col3:
        if target_col:
            st.metric(f"IQR Outliers ({target_col})", f"{iqr_count:,}", delta=f"{iqr_count/len(df_anomalies)*100:.1f}%", delta_color="inverse")
        else:
            st.metric("IQR Analysis", "Not configured", delta="Select target column", delta_color="off")
    
    st.divider()
    
    # Dashboard Tabs
    tab1, tab2, tab3 = st.tabs([
        "🚨 Anomaly Alerts & AI Analysis",
        "📊 Visual Analytics",
        "🔍 Data Explorer & Export"
    ])
    
    # TAB 1: ALERTS
    with tab1:
        st.subheader("Detected Anomaly Summary")
        if not alerts_df.empty:
            st.dataframe(alerts_df, use_container_width=True)
        
        st.subheader("Top Flagged Records with AI Analysis")
        top_anomalies = df_anomalies[df_anomalies['Anomaly_IsoForest']].sort_values('Anomaly_Score').head(10)
        
        if top_anomalies.empty:
            st.info("No anomalies detected under the current configuration.")
        else:
            for idx, row in top_anomalies.iterrows():
                score = row['Anomaly_Score']
                preview_cols = selected_features[:3]
                preview = " | ".join([f"{c}: {row[c]:.2f}" if isinstance(row[c], float) else f"{c}: {row[c]}" for c in preview_cols])
                
                with st.expander(f"Row #{idx} | Score: {score:.4f} | {preview}", expanded=(idx == top_anomalies.index[0])):
                    if openrouter_key:
                        ctx_parts = [f"{col}: {row[col]}" for col in selected_features]
                        ctx = "Anomaly detected in record:\n" + "\n".join(ctx_parts)
                        narrative = generate_openrouter_narrative(ctx, row=None, api_key=openrouter_key)
                    else:
                        narrative = generate_generic_narrative(row, selected_features)
                    
                    st.markdown("**AI Analysis:**")
                    st.info(narrative)
                    
                    # Show all feature values
                    feat_cols = st.columns(min(4, len(selected_features)))
                    for i, col in enumerate(selected_features):
                        feat_cols[i % len(feat_cols)].write(f"**{col}:** {row[col]:.4f}" if isinstance(row[col], float) else f"**{col}:** {row[col]}")
    
    # TAB 2: VISUAL ANALYTICS
    with tab2:
        if len(selected_features) >= 2:
            st.subheader("Anomaly Scatter Plot")
            x_col = st.selectbox("X-axis", options=selected_features, index=0)
            y_col = st.selectbox("Y-axis", options=selected_features, index=min(1, len(selected_features)-1))
            
            fig_scatter = px.scatter(
                df_anomalies,
                x=x_col, y=y_col,
                color='Anomaly_IsoForest',
                color_discrete_map={True: '#EF4444', False: '#94A3B8'},
                title=f"{y_col} vs {x_col} (Red = Anomaly)",
                labels={x_col: x_col, y_col: y_col},
                opacity=0.6
            )
            fig_scatter.update_layout(template="plotly_dark", height=450)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Anomaly score distribution
        st.subheader("Anomaly Score Distribution")
        fig_hist = px.histogram(
            df_anomalies,
            x='Anomaly_Score',
            nbins=80,
            title="Isolation Forest Anomaly Score Distribution",
            color='Anomaly_IsoForest',
            color_discrete_map={True: '#EF4444', False: '#3B82F6'},
            labels={'Anomaly_Score': 'Anomaly Score', 'count': 'Record Count'}
        )
        fig_hist.update_layout(template="plotly_dark", height=380)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Box plots for selected features
        if target_col:
            st.subheader(f"Box Plot — {target_col}")
            fig_box = px.box(
                df_anomalies, y=target_col,
                color='Anomaly_IsoForest' if 'Anomaly_IsoForest' in df_anomalies.columns else None,
                color_discrete_map={True: '#EF4444', False: '#3B82F6'},
                title=f"Distribution of {target_col} (Split by Anomaly Flag)"
            )
            fig_box.update_layout(template="plotly_dark", height=380)
            st.plotly_chart(fig_box, use_container_width=True)
    
    # TAB 3: DATA EXPLORER & EXPORT
    with tab3:
        st.subheader("Anomaly Data Explorer")
        
        show_filter = st.radio(
            "Show Records",
            options=["All Anomalies", "Isolation Forest Only", "IQR Only", "All Data"],
            horizontal=True
        )
        
        if show_filter == "Isolation Forest Only":
            display_df = df_anomalies[df_anomalies['Anomaly_IsoForest']]
        elif show_filter == "IQR Only" and 'Anomaly_IQR' in df_anomalies.columns:
            display_df = df_anomalies[df_anomalies['Anomaly_IQR']]
        elif show_filter == "All Anomalies":
            mask = df_anomalies['Anomaly_IsoForest']
            if 'Anomaly_IQR' in df_anomalies.columns:
                mask = mask | df_anomalies['Anomaly_IQR']
            display_df = df_anomalies[mask]
        else:
            display_df = df_anomalies
        
        st.write(f"Showing **{len(display_df):,}** records:")
        st.dataframe(display_df, use_container_width=True, height=400)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_data,
            file_name="anomaly_detection_results.csv",
            mime="text/csv"
        )


# ==========================================
# MAIN ROUTING LOGIC
# ==========================================
if dataset_mode == "📊 Sample Analysis (E-Commerce Demo)":
    # Auto-load bundled OLIST dataset from disk
    st.sidebar.success("📊 Running sample e-commerce analysis")
    run_olist_mode(data_dir=SAMPLE_DATA_DIR)

elif dataset_mode == "📄 Upload Your Own Dataset" and uploaded_generic:
    run_generic_mode(uploaded_generic)

else:
    show_welcome_screen()

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.caption("📦 AI Assisted EDA Agent — Anomaly Intelligence Engine | Built with Streamlit, Scikit-Learn, Plotly & OpenRouter API")