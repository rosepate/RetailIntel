import streamlit as st
import plotly.express as px
import pandas as pd
import sys
sys.path.append(r'c:\\Users\\rozyp\\OneDrive\\Desktop\\MLFPro\\Retailintel\\RetailIntel')
from Models.forecasting import get_sales_forecast
from Models.anomaly import detect_z_score_anomalies
from agent.agent import agent_respond

def dashboard_view(df):
    st.title("📊 RETAILINTEL Sales Dashboard")
    st.markdown("This dashboard shows key metrics, trends, forecasts, anomalies, and LLM summaries.")

    # --- KPIs ---
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df[date_cols[0]] = pd.to_datetime(df[date_cols[0]])

    total_revenue = df["Revenue"].sum()
    total_units = df["Units_Sold"].sum()
    top_product = df.groupby("Product")["Revenue"].sum().idxmax()
    top_location = df.groupby("Location")["Revenue"].sum().idxmax()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💵 Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("📦 Total Units Sold", f"{total_units:,}")
    col3.metric("🏆 Top Product", top_product)
    col4.metric("📍 Top Location", top_location)

    # --- Revenue Trends ---
    st.subheader("📈 Monthly Revenue Trend")
    monthly = df.groupby(pd.Grouper(key='Date', freq='M'))["Revenue"].sum().reset_index()
    st.line_chart(monthly.rename(columns={"Date": "Month"}).set_index("Month"))

    st.subheader("📊 Revenue by Product")
    product_revenue = df.groupby("Product")["Revenue"].sum().sort_values(ascending=False)
    fig = px.bar(product_revenue, x=product_revenue.index, y="Revenue",
                 labels={"x": "Product", "y": "Revenue"}, title="Revenue by Product Category")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📍 Revenue by Location")
    location_revenue = df.groupby("Location")["Revenue"].sum().sort_values(ascending=False)
    fig_location = px.bar(location_revenue, x=location_revenue.index, y="Revenue",
                          labels={"x": "Location", "y": "Revenue"}, title="Revenue by Location")
    st.plotly_chart(fig_location, use_container_width=True)

    st.subheader("💻 Revenue by Platform")
    platform_revenue = df.groupby("Platform")["Revenue"].sum().sort_values(ascending=False)
    fig_platform = px.bar(platform_revenue, x=platform_revenue.index, y="Revenue",
                          labels={"x": "Platform", "y": "Revenue"}, title="Revenue by Platform")
    st.plotly_chart(fig_platform, use_container_width=True)

    st.subheader("📦 Inventory Status")
    inventory_status = df.groupby("Product")["Inventory_After"].sum().sort_values(ascending=False)
    fig_inventory = px.bar(inventory_status, x=inventory_status.index, y="Inventory_After",
                           labels={"x": "Product", "y": "Inventory After"}, title="Inventory Status by Product")
    st.plotly_chart(fig_inventory, use_container_width=True)

    # --- Forecast, Alerts, Anomaly, LLM Summary (all on dashboard) ---
    st.header("🔮 Forecast, Alerts & Insights")

    # Forecast section
    product_list = sorted(df["Product"].dropna().unique())
    selected_product = st.selectbox("Select Product for Forecast", product_list, key="forecast_product")
    filtered_locations = sorted(df[df["Product"] == selected_product]["Location"].dropna().unique())
    selected_location = st.selectbox("Select Location for Forecast", filtered_locations, key="forecast_location")

    alert_enabled = st.toggle("Enable Inventory Threshold Alert", value=False, key="inv_alert_toggle")

    if st.button("Show Forecast", key="show_forecast_btn"):
        with st.spinner("Generating forecast..."):
            try:
                forecast_dates, forecast_units = get_sales_forecast(selected_product, selected_location)
                if forecast_dates is None or len(forecast_dates) == 0:
                    st.warning("No forecast available for this selection.")
                else:
                    forecast_df = pd.DataFrame({
                        "Date": forecast_dates,
                        "Forecast Units": forecast_units.flatten()
                    })
                    st.line_chart(forecast_df.set_index("Date"))
                    st.write(forecast_df)

                    # Inventory alert logic
                    if alert_enabled:
                        latest_inventory = df[
                            (df["Product"] == selected_product) & (df["Location"] == selected_location)
                        ].sort_values("Date")["Inventory_After"].iloc[-1]
                        if any(forecast_df["Forecast Units"] > latest_inventory):
                            st.error("⚠️ Alert: Predicted demand exceeds current inventory for at least one future date!")
                        else:
                            st.success("Inventory is sufficient for the forecasted period.")
            except Exception as e:
                st.error(f"Could not generate forecast: {e}")

    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection")
    anomaly_product_list = sorted(df["Product"].dropna().unique())
    anomaly_location_list = sorted(df["Location"].dropna().unique())
    anomaly_product = st.selectbox(
        "Select Product for Anomaly Detection",
        anomaly_product_list,
        key="anomaly_product"
    )
    anomaly_location = st.selectbox(
        "Select Location for Anomaly Detection",
        anomaly_location_list,
        key="anomaly_location"
    )
    filtered = df[(df["Product"] == anomaly_product) & (df["Location"] == anomaly_location)]

    if not filtered.empty:
        for col in ['Units_Sold', 'Inventory_After']:
            if col in filtered.columns:
                anomalies = detect_z_score_anomalies(filtered, column=col, threshold=3)
                detected = anomalies[anomalies['Anomaly']]
                st.markdown(f"**Anomalies in {col}:**")
                if detected.empty:
                    st.success(f"No anomalies detected in {col}.")
                else:
                    st.dataframe(detected[["Date", col, "z_score"]])
    else:
        st.info("No data for this product/location.")


    
        sample_data = df.describe().to_dict()
        user_query = f"{user_query}\n\nRecent data sample: {sample_data}"
        answer = agent_respond(user_query)
        print(answer)

    

    # LLM Summary Panel
    st.subheader("📝 Auto-generated Summary (LLM)")
    if st.button("Generate Summary"):
        with st.spinner("Generating summary with LLM..."):
            try:
                summary_prompt = (
                    "Provide a concise summary of recent sales trends, "
                    "noting any significant deviations or outliers in the data. "
                    "Highlight any inventory risks or unusual patterns."
                )
                sample_data = df.tail(5).to_dict()
                user_query = f"{summary_prompt}\n\nRecent data sample: {sample_data}"
                summary = agent_respond(user_query)
                st.success(summary)
                # Show the recent data sample as a table
                st.write("Recent Data Sample:")
                st.dataframe(df.tail(5))
                st.download_button("Download Summary", str(summary), file_name="sales_summary.txt")
            except Exception as e:
                st.error(f"Could not generate summary: {e}")