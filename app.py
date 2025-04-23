import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt


# Define template headers
company_template_cols = [
    "SKU", "Pack Size", "Price", "Number of Washes",
    "Classification", "Price Tier", "Parent Brand",
    "Previous Volume", "Present Volume", "Previous Net Sales", "Present Net Sales"
]

competitor_template_cols = [
    "SKU", "Pack Size", "Price", "Number of Washes",
    "Classification", "Price Tier", "Parent Brand"
]

def generate_excel_download(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Template")
    buffer.seek(0)
    return buffer

# --- UI Starts ---
st.title("📦 Price Pack Architecture Tool")

st.markdown("Before uploading, please use the templates below to prepare your data:")

col1, col2 = st.columns(2)

with col1:
    company_buffer = generate_excel_download(pd.DataFrame(columns=company_template_cols))
    st.download_button(
        label="📥 Download Company Template",
        data=company_buffer,
        file_name="company_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    competitor_buffer = generate_excel_download(pd.DataFrame(columns=competitor_template_cols))
    st.download_button(
        label="📥 Download Competitor Template",
        data=competitor_buffer,
        file_name="competitor_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )




st.header("Upload Your Data")
company_file = st.file_uploader("Upload Your Company Data (CSV)", type="csv")
competitor_file = st.file_uploader("Upload Competitor Data (CSV)", type="csv")

company_cols = ["SKU", "Pack Size", "Price", "Number of Washes", 
                "Classification", "Price Tier", "Parent Brand", 
                "Previous Volume", "Present Volume", 
                "Previous Net Sales", "Present Net Sales"]

competitor_cols = ["SKU", "Pack Size", "Price", "Number of Washes", 
                   "Classification", "Price Tier", "Parent Brand"]

def assign_tier(ppw, thresholds):
    if ppw <= thresholds['Value'][1]:
        return 'Value'
    elif ppw <= thresholds['Mainstream'][1]:
        return 'Mainstream'
    elif ppw <= thresholds['Premium'][1]:
        return 'Premium'
    else:
        return 'Others'

# ✅ Moved outside assign_tier()
def generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers):
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #ccc;
            padding: 10px 12px; /* Increased padding */
            text-align: center;
            vertical-align: middle;
        }
        th {
            font-weight: bold;
        }
        td[colspan="3"] {
            min-width: 180px; /* Adjust to fit 20 characters easily */
        }
    </style>


    <table>
        <tr>
            <th>Classification</th>
    """
    for cls in classifications:
        html += f'<th colspan="3">{cls}</th>'
    html += '<th rowspan="3">Avg PP CPW</th>'
    html += '<th rowspan="3">Value Weight</th>'
    html += '<th rowspan="3">Growth</th></tr>'

    html += "<tr><td>Net Sales Growth %</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Growth"]}</td>'
    html += '</tr>'

    html += "<tr><td>Value Share %</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Value"]}</td>'
    html += '</tr>'

    html += "<tr><td>PPW Range</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["PPW"]}</td>'
    html += '<td></td><td></td><td></td></tr>'

    for tier in tiers:
        html += f'<tr><td>{tier}</td>'
        for cls in classifications:
            skus = sku_matrix[tier][cls]
            html += f'<td colspan="3">{"<br>".join(skus) if skus else "-"}</td>'
        html += f'<td>{tier_metrics[tier]["PPW"]}</td>'
        html += f'<td>{tier_metrics[tier]["Share"]}</td>'
        html += f'<td>{tier_metrics[tier]["Growth"]}</td></tr>'
    html += "</table>"
    return html


def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)  # Remove {currency_symbol}, commas, $, spaces
        .replace("", pd.NA)
        .astype(float)
    )


if 'classified' not in st.session_state:
    st.session_state.classified = False


if company_file and competitor_file:
    st.subheader("Currency Settings")
    currency_symbol = st.text_input("Enter your currency symbol (e.g. ₹, $, €, etc.):", value="₹")
    company_df = pd.read_csv(company_file)
    competitor_df = pd.read_csv(competitor_file)

    if company_df["Classification"].nunique() > 4:
        st.error("You have more than 4 classifications in your company data.")
    else:
        # Clean numeric fields
        for col in ["Price", "Number of Washes", "Previous Volume", "Present Volume", "Previous Net Sales", "Present Net Sales"]:
            company_df[col] = clean_numeric(company_df[col])
        for col in ["Price", "Number of Washes"]:
            competitor_df[col] = clean_numeric(competitor_df[col])
        
        # Calculate Price per Wash
        company_df["Price per Wash"] = company_df["Price"] / company_df["Number of Washes"]
        competitor_df["Price per Wash"] = competitor_df["Price"] / competitor_df["Number of Washes"]


        st.subheader("Price per Wash Range")
        st.write(f"Company: {currency_symbol}{company_df['Price per Wash'].min():.2f} – {currency_symbol}{company_df['Price per Wash'].max():.2f}")
        st.write(f"Competitor: {currency_symbol}{competitor_df['Price per Wash'].min():.2f} – {currency_symbol}{competitor_df['Price per Wash'].max():.2f}")
        
        st.subheader(f"Set Price Tier Thresholds ({currency_symbol})")
        with st.form("thresholds"):
            col1, col2, col3 = st.columns(3)
            with col1:
                value_max = st.number_input(f"Value: Max {currency_symbol}", value=.13)
            with col2:
                mainstream_max = st.number_input(f"Mainstream: Max {currency_symbol}", value=.17)
            with col3:
                premium_max = st.number_input(f"Premium: Max {currency_symbol}", value=1)
            submit_btn = st.form_submit_button("Classify SKUs")
            if submit_btn:
                st.session_state['classified'] = True



        if submit_btn:
            st.session_state['classified'] = True  # 🔒 Locks the view to analysis mode
            st.rerun()  # 🔁 Reruns the app to jump into analysis block
    
        if st.session_state['classified']:
            thresholds = {
                'Value': (0.0, value_max),
                'Mainstream': (value_max, mainstream_max),
                'Premium': (mainstream_max, premium_max)
            }
   


            company_df['Calculated Price Tier'] = company_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
            competitor_df['Calculated Price Tier'] = competitor_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
            company_df['Is Competitor'] = False
            competitor_df['Is Competitor'] = True

            full_df = pd.concat([company_df, competitor_df], ignore_index=True)
            tiers = ['Premium', 'Mainstream', 'Value']
            classifications = sorted(full_df['Classification'].unique())

            sku_matrix = {tier: {cls: [] for cls in classifications} for tier in tiers}
            classification_metrics = {}
            tier_metrics = {}
            total_company_net_sales = company_df['Present Net Sales'].sum()

            for cls in classifications:
                all_cls = full_df[full_df['Classification'] == cls]
                our_cls = company_df[company_df['Classification'] == cls]
                prev_rev = our_cls['Previous Net Sales'].sum()
                curr_rev = our_cls['Present Net Sales'].sum()
                growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                share = (curr_rev / total_company_net_sales * 100) if total_company_net_sales else 0
                ppw_range = f"{all_cls['Price per Wash'].min():.2f} – {all_cls['Price per Wash'].max():.2f}" if not all_cls.empty else "-"
                classification_metrics[cls] = {
                    "Growth": f"{growth:.1f}%",
                    "Value": f"{share:.1f}%",
                    "PPW": ppw_range
                }

            for tier in tiers:
                tier_full = full_df[full_df["Calculated Price Tier"] == tier]
                tier_our = company_df[company_df["Calculated Price Tier"] == tier]
                prev = tier_our["Previous Net Sales"].sum()
                curr = tier_our["Present Net Sales"].sum()
                
                min_ppw = tier_full["Price per Wash"].min()
                max_ppw = tier_full["Price per Wash"].max()
                ppw_range = f"{currency_symbol}{min_ppw:.2f} – {currency_symbol}{max_ppw:.2f}" if not tier_full.empty else "-"
                
                growth = ((curr - prev) / prev * 100) if prev else 0
                share = (curr / total_company_net_sales * 100) if total_company_net_sales else 0
                tier_metrics[tier] = {
                    "PPW": ppw_range,
                    "Growth": f"{growth:.1f}%",
                    "Share": f"{share:.1f}%"
                }


            for _, row in full_df.iterrows():
                tier = row["Calculated Price Tier"]
                cls = row["Classification"]
                sku = row["SKU"]
                if tier in sku_matrix and cls in sku_matrix[tier]:
                    sku_matrix[tier][cls].append(sku)

            # After HTML render
            # Store dynamic HTML once on submit
            if 'matrix_html' not in st.session_state or submit_btn:
                dynamic_html = generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers)
                st.session_state.matrix_html = dynamic_html
            
            # Always display cached HTML
            if 'matrix_html' in st.session_state:
                st.markdown(st.session_state.matrix_html, unsafe_allow_html=True)


          # ----- SKU GROWTH SUMMARY -----
            st.subheader("📈 SKU-Level Growth Summary (Our Company Only)")
            
            sku_growth_summary = []
            
            for _, row in company_df.iterrows():
                sku = row['SKU']
                prev_vol = row['Previous Volume']
                curr_vol = row['Present Volume']
                prev_rev = row['Previous Net Sales']
                curr_rev = row['Present Net Sales']
            
                volume_growth = ((curr_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                Net_Sales_growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
            
                sku_growth_summary.append({
                    "SKU": sku,
                    "Previous Volume": prev_vol,
                    "Present Volume": curr_vol,
                    "Volume Growth %": f"{volume_growth:.1f}%",
                    "Previous Net Sales": prev_rev,
                    "Present Net Sales": curr_rev,
                    "Net Sales Growth %": f"{Net_Sales_growth:.1f}%"
                })
            
            # Show as table
            st.dataframe(pd.DataFrame(sku_growth_summary))


          # SCATTER PLOT: Retail Price vs. Price Per Wash
            from adjustText import adjust_text
            import numpy as np
            
            st.subheader("📈 Scatter Plot: Retail Price vs. Price Per Wash")
            
            # Combine company and competitor for plot
            plot_df = pd.concat([company_df, competitor_df], ignore_index=True).copy()
            
            # Add small jitter to overlapping points
            plot_df['Jittered PPW'] = plot_df['Price per Wash'] + np.random.normal(0, 0.002, size=len(plot_df))
            plot_df['Jittered Price'] = plot_df['Price'] + np.random.normal(0, 0.3, size=len(plot_df))
            
            # Define colors
            plot_df['Color'] = plot_df['Is Competitor'].apply(lambda x: 'green' if x else 'navy')
            
            # Axis ranges
            x_min = plot_df['Jittered PPW'].min() - 0.03
            x_max = plot_df['Jittered PPW'].max() + 0.03
            y_min = plot_df['Jittered Price'].min() - 2
            y_max = plot_df['Jittered Price'].max() + 2
            
            # Matplotlib Plot
            fig, ax = plt.subplots(figsize=(12, 7))
            
            # Plot each point with its color
            ax.scatter(plot_df['Jittered PPW'], plot_df['Jittered Price'], c=plot_df['Color'], s=70, alpha=0.8)
            
            # Add labels with adjustText
            texts = [
                ax.text(row['Jittered PPW'], row['Jittered Price'], row['SKU'], fontsize=8)
                for _, row in plot_df.iterrows()
            ]
            
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color='gray', lw=0.5),
                expand_points=(1.2, 1.4),
                expand_text=(1.2, 1.4),
                force_text=0.5,
                force_points=0.4,
                only_move={'points': 'y', 'text': 'xy'},
            )
            
            ax.set_xlabel("Price Per Wash")
            ax.set_ylabel("Retail Price")
            ax.set_title("Scatter Plot of SKUs")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)

            

            st.subheader("📊 API Comparison: Our SKUs vs Competitors (By Classification & Tier)")

            

            api_rows = []
            
            # Loop through all classification × tier segments
            for classification in classifications:
                for tier in tiers:
                    segment_df = full_df[
                        (full_df["Classification"] == classification) &
                        (full_df["Calculated Price Tier"] == tier)
                    ]
                    our_skus = segment_df[segment_df["Is Competitor"] == False]
                    comp_skus = segment_df[segment_df["Is Competitor"] == True]
            
                    if not comp_skus.empty:
                        avg_comp_ppw = comp_skus["Price per Wash"].mean()
            
                        for _, our_row in our_skus.iterrows():
                            our_ppw = our_row["Price per Wash"]
                            api = our_ppw / avg_comp_ppw if avg_comp_ppw else float('nan')
            
                            api_rows.append({
                                "Classification": classification,
                                "Price Tier": tier,
                                "Our SKU": our_row["SKU"],
                                "Our PPW": round(our_ppw, 2),
                                "Avg Competitor PPW": round(avg_comp_ppw, 2),
                                "API (Our / Comp)": round(api, 2)
                            })
            
            if api_rows:
                api_df = pd.DataFrame(api_rows)
                st.dataframe(api_df)
            else:
                st.info("No competitor SKUs found in any classification-tier segment.")

            
            if api_rows:
                api_df = pd.DataFrame(api_rows)
                st.dataframe(api_df)
            else:
                st.info("No matching competitor SKUs found in any classification-tier combination.")

            st.subheader("🔁 Compare API Between Two SKUs")
# Combine company and competitor for dropdowns
            sku_ppw_map = full_df.set_index("SKU")["Price per Wash"].to_dict()
            sku_list = sorted(sku_ppw_map.keys())
            
            col1, col2 = st.columns(2)
            with col1:
                sku_a = st.selectbox("Select SKU A", sku_list)
            with col2:
                sku_b = st.selectbox("Select SKU B", sku_list, index=1)
            
            if sku_a and sku_b and sku_a != sku_b:
                ppw_a = sku_ppw_map[sku_a]
                ppw_b = sku_ppw_map[sku_b]
            
                api = ppw_a / ppw_b if ppw_b else float('nan')
                
                st.markdown(f"""
                **SKU A:** `{sku_a}` — PPW = {currency_symbol}{ppw_a:.2f}  
                **SKU B:** `{sku_b}` — PPW = {currency_symbol}{ppw_b:.2f}  
                
                📊 **API (A vs B)** = {ppw_a:.2f} / {ppw_b:.2f} = **{api:.2f}**
                """)
            else:
                st.info("Please select two different SKUs.")




            
