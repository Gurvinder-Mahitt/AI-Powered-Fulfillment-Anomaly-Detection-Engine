"""
AI Assisted EDA Agent
Operational Anomaly and Intelligence Pipeline for E-Commerce Logistics

Supports:
  - OLIST e-commerce dataset (multi-CSV upload with specific schema)
  - Generic CSV datasets (auto-detects numeric columns for anomaly detection)
"""

import os
import json
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def read_csv_safe(filepath):
    """Reads a CSV with automatic encoding detection."""
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            return pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
        except Exception:
            continue
    return pd.read_csv(filepath, encoding='utf-8', encoding_errors='replace', on_bad_lines='skip')


def load_and_clean_data(data_dir=None, uploaded_files=None):
    """
    Loads and merges the OLIST e-commerce dataset.

    Args:
        data_dir: Path to directory containing OLIST CSV files (disk mode).
        uploaded_files: Dict of filename to file-like object from Streamlit uploader.

    Returns:
        Cleaned and feature-engineered pandas DataFrame.
    """
    if uploaded_files:
        print("[Pipeline] Loading datasets from uploaded files...")
        orders = pd.read_csv(uploaded_files['olist_orders_dataset.csv'])
        order_items = pd.read_csv(uploaded_files['olist_order_items_dataset.csv'])
        payments = pd.read_csv(uploaded_files['olist_order_payments_dataset.csv'])
        customers = pd.read_csv(uploaded_files['olist_customers_dataset.csv'])
    elif data_dir and os.path.exists(data_dir):
        print(f"[Pipeline] Loading raw datasets from: {data_dir}")
        orders = read_csv_safe(os.path.join(data_dir, 'olist_orders_dataset.csv'))
        order_items = read_csv_safe(os.path.join(data_dir, 'olist_order_items_dataset.csv'))
        payments = read_csv_safe(os.path.join(data_dir, 'olist_order_payments_dataset.csv'))
        customers = read_csv_safe(os.path.join(data_dir, 'olist_customers_dataset.csv'))
    else:
        raise FileNotFoundError(
            "No dataset provided. Please upload CSV files or specify a valid data directory."
        )

    payments_agg = payments.groupby('order_id').agg(
        Total_Order_Value=('payment_value', 'sum'),
        Primary_Payment_Type=('payment_type', 'first'),
        Payment_Installments=('payment_installments', 'max')
    ).reset_index()

    items_agg = order_items.groupby('order_id').agg(
        Total_Freight_Value=('freight_value', 'sum'),
        Item_Count=('order_item_id', 'count')
    ).reset_index()

    df = orders.merge(customers[['customer_id', 'customer_city', 'customer_state']], on='customer_id', how='left')
    df = df.merge(payments_agg, on='order_id', how='left')
    df = df.merge(items_agg, on='order_id', how='left')

    timestamp_cols = [
        'order_purchase_timestamp',
        'order_approved_at',
        'order_delivered_carrier_date',
        'order_delivered_customer_date',
        'order_estimated_delivery_date'
    ]

    for col in timestamp_cols:
        df[col] = pd.to_datetime(df[col])

    df['Delivery_Time_Hours'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.total_seconds() / 3600
    df['Prep_Time_Hours'] = (df['order_delivered_carrier_date'] - df['order_approved_at']).dt.total_seconds() / 3600
    df['Delivery_Delay_Hours'] = (df['order_delivered_customer_date'] - df['order_estimated_delivery_date']).dt.total_seconds() / 3600

    df_clean = df.dropna(subset=['Delivery_Time_Hours', 'Prep_Time_Hours', 'Total_Order_Value']).copy()
    print(f"[Pipeline] Cleaned Dataset Shape: {df_clean.shape}")
    return df_clean


def detect_iqr_delivery_anomalies(df, iqr_multiplier=1.5):
    """Flags orders exceeding state-specific upper delivery threshold (Q3 + multiplier * IQR)."""
    df_iqr = df.copy()
    df_iqr['Anomaly_IQR_Delivery'] = False

    for state, group in df_iqr.groupby('customer_state'):
        q1 = group['Delivery_Time_Hours'].quantile(0.25)
        q3 = group['Delivery_Time_Hours'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + (iqr_multiplier * iqr)
        flagged_indices = group[group['Delivery_Time_Hours'] > upper_bound].index
        df_iqr.loc[flagged_indices, 'Anomaly_IQR_Delivery'] = True

    flagged_count = df_iqr['Anomaly_IQR_Delivery'].sum()
    print(f"[IQR Check] Flagged {flagged_count} delivery delay anomalies out of {len(df_iqr)} orders.")
    return df_iqr


def detect_timeseries_volume_anomalies(df, freq='D'):
    """Resamples order count by frequency and applies 7-period rolling stats to detect severe volume drops."""
    ts = df.set_index('order_purchase_timestamp').resample(freq)['order_id'].count().to_frame()
    ts.columns = ['Order_Count']

    ts['Rolling_Mean'] = ts['Order_Count'].rolling(window=7, min_periods=3).mean()
    ts['Rolling_Std'] = ts['Order_Count'].rolling(window=7, min_periods=3).std()

    ts['LCL'] = ts['Rolling_Mean'] - (3 * ts['Rolling_Std'])
    ts['UCL'] = ts['Rolling_Mean'] + (3 * ts['Rolling_Std'])
    ts['Anomaly_Volume_Drop'] = ts['Order_Count'] < ts['LCL']

    flagged_dates = ts[ts['Anomaly_Volume_Drop']].index.tolist()
    print(f"[Time-Series Check] Flagged {len(flagged_dates)} severe order volume drops.")
    return ts


def detect_multivariate_anomalies(df, contamination=0.01):
    """Isolation Forest for multivariate operational glitches."""
    df_ml = df.copy()
    features = ['Total_Order_Value', 'Prep_Time_Hours', 'Total_Freight_Value', 'Payment_Installments']

    X = df_ml[features].fillna(df_ml[features].median()).values

    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X)

    df_ml['Anomaly_Score'] = iso_model.decision_function(X)
    df_ml['Anomaly_IsoForest'] = iso_model.predict(X) == -1

    flagged_count = df_ml['Anomaly_IsoForest'].sum()
    print(f"[Isolation Forest] Flagged {flagged_count} multivariate anomalies.")
    return df_ml


def detect_payment_gateway_anomalies(df, state_code='SP'):
    """Analyzes payment method distribution variance for a given state."""
    state_df = df[df['customer_state'] == state_code]
    if state_df.empty:
        return pd.DataFrame()

    overall_dist = df['Primary_Payment_Type'].value_counts(normalize=True)
    state_dist = state_df['Primary_Payment_Type'].value_counts(normalize=True)

    anomalies = []
    for ptype, prob in state_dist.items():
        expected = overall_dist.get(ptype, 0.0)
        diff = abs(prob - expected)
        if diff > 0.15:
            anomalies.append({
                'Date': 'Recent Aggregated Window',
                'State': state_code,
                'Payment_Type': ptype,
                'State_Share': f"{prob*100:.1f}%",
                'National_Share': f"{expected*100:.1f}%",
                'p_value': round(diff, 4)
            })
    return pd.DataFrame(anomalies)


def consolidate_alerts(df_phase3, volume_drop_alerts, payment_anomalies):
    """Consolidates flagged rows across all engines into structured severity tiers."""
    alerts = []

    if not volume_drop_alerts.empty and 'Anomaly_Volume_Drop' in volume_drop_alerts.columns:
        p1_volume = volume_drop_alerts[volume_drop_alerts['Anomaly_Volume_Drop']].copy()
        for date, row in p1_volume.iterrows():
            alerts.append({
                'Severity': 'P1 - CRITICAL',
                'Category': 'Platform Volume Drop',
                'Entity/Date': str(date.date()) if hasattr(date, 'date') else str(date),
                'Impact_Metric': f"Order Count: {row['Order_Count']} (Expected: {row['Rolling_Mean']:.0f})",
                'Action_Required': 'Investigate payment gateway or site outage immediately.'
            })

    if not payment_anomalies.empty:
        for _, row in payment_anomalies.iterrows():
            alerts.append({
                'Severity': 'P1 - CRITICAL',
                'Category': 'Regional Payment Failure',
                'Entity/Date': f"State: {row.get('State', 'SP')}",
                'Impact_Metric': f"{row.get('Payment_Type', 'Method')} Share Variance ({row.get('State_Share')} vs {row.get('National_Share')})",
                'Action_Required': 'Check regional UPI/Credit Card integration.'
            })

    if 'Anomaly_IsoForest' in df_phase3.columns:
        p2_ml = df_phase3[df_phase3['Anomaly_IsoForest']].copy()
        for _, row in p2_ml.head(20).iterrows():
            alerts.append({
                'Severity': 'P2 - WARNING',
                'Category': 'Multivariate Operational Glitch',
                'Entity/Date': f"Order #{row['order_id']}",
                'Impact_Metric': f"Value: R${row['Total_Order_Value']:.2f} | Prep: {row['Prep_Time_Hours']:.1f} hrs",
                'Action_Required': 'Audit seller fulfillment and shipping speed.'
            })

    if 'Anomaly_IQR_Delivery' in df_phase3.columns:
        p3_delays = df_phase3[df_phase3['Anomaly_IQR_Delivery']].groupby('customer_state').size().reset_index(name='Delay_Count')
        for _, row in p3_delays.sort_values('Delay_Count', ascending=False).head(5).iterrows():
            alerts.append({
                'Severity': 'P3 - INFO',
                'Category': 'Regional Delivery Backlog',
                'Entity/Date': f"State: {row['customer_state']}",
                'Impact_Metric': f"{row['Delay_Count']} Delayed Orders exceeding IQR Upper Limit",
                'Action_Required': 'Notify regional carrier for logistics delay.'
            })

    alerts_df = pd.DataFrame(alerts)
    print(f"[Alert Engine] Consolidated {len(alerts_df)} operational alert items.")
    return alerts_df


def extract_anomaly_context(row, anomaly_type="Delivery Delay"):
    """Extracts key metrics into clean string for prompting/narrative."""
    if anomaly_type == "Delivery Delay":
        return (
            f"Order ID: {row.get('order_id', 'N/A')}\n"
            f"State/Location: {row.get('customer_state', 'N/A')}\n"
            f"Total Order Value: R${row.get('Total_Order_Value', 0):.2f}\n"
            f"Actual Delivery Time: {row.get('Delivery_Time_Hours', 0):.1f} hours\n"
            f"Carrier Prep Time: {row.get('Prep_Time_Hours', 0):.1f} hours\n"
            f"Freight Value: R${row.get('Total_Freight_Value', 0):.2f}\n"
        )
    elif anomaly_type == "Volume Drop":
        return (
            f"Date of Incident: {row.get('Date', 'N/A')}\n"
            f"Actual Order Count: {row.get('Order_Count', 0)}\n"
            f"Expected Rolling Average: {row.get('Rolling_Mean', 0):.0f}\n"
            f"Lower Control Limit (Threshold): {row.get('LCL', 0):.0f}\n"
        )
    else:
        return (
            f"Order ID: {row.get('order_id', 'N/A')}\n"
            f"State: {row.get('customer_state', 'N/A')}\n"
            f"Order Value: R${row.get('Total_Order_Value', 0):.2f}\n"
            f"Prep Time: {row.get('Prep_Time_Hours', 0):.1f} hours\n"
            f"Freight Value: R${row.get('Total_Freight_Value', 0):.2f}\n"
            f"Payment Installments: {row.get('Payment_Installments', 1)}\n"
            f"Isolation Forest Score: {row.get('Anomaly_Score', 0):.4f}\n"
        )


def generate_rule_based_narrative(row):
    """Smart fallback generator providing 3-bullet structured operational analysis."""
    prep_hours = row.get('Prep_Time_Hours', 0)
    order_val = row.get('Total_Order_Value', 0)
    freight_val = row.get('Total_Freight_Value', 0)
    deliv_hours = row.get('Delivery_Time_Hours', 0)
    state = row.get('customer_state', 'N/A')

    if prep_hours > 72:
        root_cause = f"Severe seller dispatch latency ({prep_hours:.1f} hours from purchase to carrier handoff)."
        action = f"Issue automated SLA non-compliance ticket to seller in state {state}."
    elif freight_val > order_val * 0.8 and order_val > 0:
        root_cause = f"Abnormally high freight fee ratio (Freight: R${freight_val:.2f} vs Order Value: R${order_val:.2f})."
        action = "Audit carrier shipping weight/volume calculation rule for billing discrepancy."
    elif deliv_hours > 300:
        root_cause = f"Regional logistics bottleneck in state {state} causing overall delivery time of {deliv_hours/24:.1f} days."
        action = f"Contact regional carrier hub in {state} to re-route backlog items."
    else:
        root_cause = f"Multivariate outlier flagged by Isolation Forest (Value: R${order_val:.2f}, Prep: {prep_hours:.1f} hrs)."
        action = f"Perform manual operational review of Order #{row.get('order_id', 'N/A')} and check inventory logs."

    return (
        f"* **1. The Anomaly:** Order #{row.get('order_id', 'N/A')} in state {state} flagged with irregular operational metrics (Value: R${order_val:.2f}).\n"
        f"* **2. Potential Root Cause:** {root_cause}\n"
        f"* **3. Actionable Fix:** {action}"
    )


def generate_openrouter_narrative(context_string, row=None, api_key=None):
    """Queries OpenRouter API or falls back to rule-based analysis."""
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "openrouter/free",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Supply Chain Analyst. "
                        "Respond strictly in 3 bullet points:\n"
                        "* 1. The Anomaly: Summary of what went wrong.\n"
                        "* 2. Potential Root Cause: Operational hypothesis.\n"
                        "* 3. Actionable Fix: Immediate step to investigate."
                    )
                },
                {"role": "user", "content": f"Analyze this operational anomaly:\n\n{context_string}"}
            ],
            "temperature": 0.2
        }
        try:
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=8)
            if res.status_code == 200:
                data = res.json()
                content = data['choices'][0]['message']['content']
                if content and len(content.strip()) > 10:
                    return content.strip()
        except Exception as e:
            print(f"[OpenRouter API Warning] API call unsuccessful: {e}")

    if isinstance(row, pd.Series):
        return generate_rule_based_narrative(row)
    return (
        "* **1. The Anomaly:** High variance operational outlier detected.\n"
        "* **2. Potential Root Cause:** Unusually long carrier handoff or dispatch delay.\n"
        "* **3. Actionable Fix:** Inspect order lifecycle events and notify seller support."
    )


def generate_anomaly_narrative(context_string, anomaly_type="Multivariate Operational Glitch", api_key=None, row=None):
    """Wrapper for narrative generation."""
    return generate_openrouter_narrative(context_string, row=row, api_key=api_key)


# ==========================================
# GENERIC CSV ANOMALY DETECTION
# ==========================================
def detect_generic_iqr_anomalies(df, target_column, group_column=None, iqr_multiplier=1.5):
    """
    IQR-based anomaly detection on any numeric column, optionally grouped.

    Args:
        df: Input DataFrame.
        target_column: Numeric column to check for outliers.
        group_column: Optional categorical column to group by.
        iqr_multiplier: IQR sensitivity multiplier (default 1.5).

    Returns:
        DataFrame with Anomaly_IQR boolean column added.
    """
    df_out = df.copy()
    col_name = 'Anomaly_IQR'
    df_out[col_name] = False

    if group_column and group_column in df_out.columns:
        for group_val, group in df_out.groupby(group_column):
            q1 = group[target_column].quantile(0.25)
            q3 = group[target_column].quantile(0.75)
            iqr = q3 - q1
            upper = q3 + (iqr_multiplier * iqr)
            lower = q1 - (iqr_multiplier * iqr)
            flagged = group[(group[target_column] > upper) | (group[target_column] < lower)].index
            df_out.loc[flagged, col_name] = True
    else:
        q1 = df_out[target_column].quantile(0.25)
        q3 = df_out[target_column].quantile(0.75)
        iqr = q3 - q1
        upper = q3 + (iqr_multiplier * iqr)
        lower = q1 - (iqr_multiplier * iqr)
        df_out[col_name] = (df_out[target_column] > upper) | (df_out[target_column] < lower)

    flagged_count = df_out[col_name].sum()
    print(f"[Generic IQR] Flagged {flagged_count} outliers in '{target_column}' out of {len(df_out)} rows.")
    return df_out


def detect_generic_multivariate_anomalies(df, feature_columns, contamination=0.01):
    """
    Isolation Forest anomaly detection on user-selected numeric columns.

    Args:
        df: Input DataFrame.
        feature_columns: List of numeric column names to use as features.
        contamination: Expected proportion of anomalies (default 1%).

    Returns:
        DataFrame with Anomaly_Score and Anomaly_IsoForest columns added.
    """
    df_ml = df.copy()
    X = df_ml[feature_columns].fillna(df_ml[feature_columns].median()).values

    iso_model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X)

    df_ml['Anomaly_Score'] = iso_model.decision_function(X)
    df_ml['Anomaly_IsoForest'] = iso_model.predict(X) == -1

    flagged_count = df_ml['Anomaly_IsoForest'].sum()
    print(f"[Generic Isolation Forest] Flagged {flagged_count} multivariate anomalies across {len(feature_columns)} features.")
    return df_ml


def generate_generic_narrative(row, feature_columns):
    """Rule-based narrative generator for generic dataset anomalies."""
    anomaly_details = []
    for col in feature_columns:
        val = row.get(col, 'N/A')
        if isinstance(val, (int, float)):
            anomaly_details.append(f"{col}: {val:.4f}")
        else:
            anomaly_details.append(f"{col}: {val}")

    details_str = " | ".join(anomaly_details)
    score = row.get('Anomaly_Score', 0)

    return (
        f"* **1. The Anomaly:** Row flagged as multivariate outlier with Isolation Forest score {score:.4f}.\n"
        f"* **2. Key Metrics:** {details_str}\n"
        f"* **3. Actionable Fix:** Investigate this record for data quality issues or genuine extreme behavior."
    )


def run_generic_pipeline(df, feature_columns, target_column=None, group_column=None,
                         contamination=0.01, iqr_multiplier=1.5):
    """
    Runs a simplified anomaly detection pipeline on any CSV dataset.

    Args:
        df: Input DataFrame (already loaded).
        feature_columns: List of numeric columns for Isolation Forest.
        target_column: Optional single numeric column for IQR analysis.
        group_column: Optional categorical column for grouped IQR.
        contamination: Isolation Forest contamination rate.
        iqr_multiplier: IQR sensitivity multiplier.

    Returns:
        Dict with pipeline results.
    """
    print(f"[Generic Pipeline] Starting anomaly detection on {len(df)} rows, {len(feature_columns)} features...")

    df_anomalies = detect_generic_multivariate_anomalies(df, feature_columns, contamination)

    if target_column and target_column in df.columns:
        df_anomalies = detect_generic_iqr_anomalies(df_anomalies, target_column, group_column, iqr_multiplier)

    alerts = []
    iso_count = df_anomalies['Anomaly_IsoForest'].sum()
    alerts.append({
        'Severity': 'P2 - WARNING',
        'Category': 'Multivariate Outliers (Isolation Forest)',
        'Entity/Date': f"{iso_count} records",
        'Impact_Metric': f"{iso_count}/{len(df_anomalies)} rows flagged ({iso_count/len(df_anomalies)*100:.1f}%)",
        'Action_Required': 'Review flagged records for data quality or genuine anomalies.'
    })

    if 'Anomaly_IQR' in df_anomalies.columns:
        iqr_count = df_anomalies['Anomaly_IQR'].sum()
        alerts.append({
            'Severity': 'P3 - INFO',
            'Category': f'IQR Outliers on "{target_column}"',
            'Entity/Date': f"{iqr_count} records",
            'Impact_Metric': f"{iqr_count}/{len(df_anomalies)} rows outside IQR bounds",
            'Action_Required': f'Inspect extreme values in column "{target_column}".'
        })

    alerts_df = pd.DataFrame(alerts)

    return {
        'df_clean': df,
        'df_anomalies': df_anomalies,
        'alerts_df': alerts_df,
        'feature_columns': feature_columns
    }


# ==========================================
# OLIST PIPELINE RUNNER
# ==========================================
def run_anomaly_pipeline(data_dir=None, uploaded_files=None, contamination=0.01, iqr_multiplier=1.5):
    """Executes full OLIST detection pipeline and returns clean DataFrames for Streamlit display."""
    df_clean = load_and_clean_data(data_dir=data_dir, uploaded_files=uploaded_files)
    df_iqr = detect_iqr_delivery_anomalies(df_clean, iqr_multiplier=iqr_multiplier)
    ts_volume = detect_timeseries_volume_anomalies(df_clean, freq='D')
    df_phase3 = detect_multivariate_anomalies(df_iqr, contamination=contamination)
    payment_anomalies = detect_payment_gateway_anomalies(df_phase3, state_code='SP')
    alerts_df = consolidate_alerts(df_phase3, ts_volume, payment_anomalies)

    return {
        'df_clean': df_clean,
        'df_anomalies': df_phase3,
        'ts_volume': ts_volume,
        'payment_anomalies': payment_anomalies,
        'alerts_df': alerts_df
    }


if __name__ == '__main__':
    print("=== AI Assisted EDA Agent Pipeline ===")
    print("This module is designed to be used via the Streamlit dashboard.")
    print("Run: streamlit run app.py")
