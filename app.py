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
            padding: 10px 12px;
            text-align: center;
            vertical-align: middle;
        }
        th {
            font-weight: bold;
        }
        td[colspan="3"] {
            min-width: 180px;
        }
    </style>
    
    <table>
    """
    
  # --- Row: Unilever Net Sales Growth Percentage ---
# Row: Unilever Net Sales Growth %
    html += '<tr>'
    html += '<td style="font-weight:bold;">Unilever Net Sales Growth Percentage</td>'
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Growth"]}</td>'
    html += '<td></td><td></td><td></td><td></td></tr>'
    
    # Row: Unilever Value Share %
    html += '<tr>'
    html += '<td style="font-weight:bold;">Unilever Value Share %</td>'
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Value"]}</td>'
    html += '<td></td><td></td><td></td><td></td></tr>'




    
    # ---- Now comes the actual column headers ----
    html += '<tr><th>Classification</th>'
    for cls in classifications:
        html += f'<th colspan="3">{cls}</th>'
    html += '<th rowspan="3">Avg PP CPW</th>'
    html += '<th rowspan="3">Value Weight</th>'
    html += '<th rowspan="3">Growth</th></tr>'
    
    # Continue with the rest of the HTML generation...
    # --- ADD MISSING BODY ---
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

    def generate_black_matrix_html(classifications, tiers):
        html = """
        <style>
            table.black-matrix {
                border-collapse: collapse;
                width: 100%;
                font-family: Arial, sans-serif;
                font-size: 13px;
                margin-top: 40px;
            }
            table.black-matrix th, table.black-matrix td {
                border: 1px solid #444;
                padding: 8px 10px;
                text-align: center;
            }
            table.black-matrix th {
                background-color: black;
                color: white;
            }
            table.black-matrix td.metric {
                background-color: #d9d9d9;
            }
            table.black-matrix .label {
                background-color: #bfd7ea;
                font-weight: bold;
            }
        </style>
        <table class="black-matrix">
            <!-- Value Growth Row -->
            <tr><td class="label">Unilever Value Growth</td>"""
        for _ in classifications:
            html += "<td>-</td>"
        html += "<td class='metric'>-</td><td class='metric'>-</td><td class='metric'>-</td></tr>"
    
        # Value Weight Row
        html += "<tr><td class='label'>Unilever Value Weight</td>"
        for _ in classifications:
            html += "<td>-</td>"
        html += "<td class='metric'>-</td><td class='metric'>-</td><td class='metric'>-</td></tr>"
    
        # Column Header Row
        html += "<tr><td class='label'>CVD</td>"
        for cls in classifications:
            html += f"<th>{cls}</th>"
        html += "<th class='metric'>Avg PP CPW</th><th class='metric'>Value Weight</th><th class='metric'>Growth</th></tr>"
    
        # Tier rows + shelf space
        for tier in tiers:
            html += f"<tr><td rowspan='2'>{tier}</td>"
            for _ in classifications:
                html += "<td>-</td>"
            html += "<td class='metric'>-</td><td class='metric'>-</td><td class='metric'>-</td></tr>"
    
            html += "<tr><td class='label'>Unilever Shelf Space %</td>"
            for _ in classifications:
                html += "<td>-</td>"
            html += "<td class='metric'>-</td><td class='metric'>-</td><td class='metric'>-</td></tr>"
    
        # Final row
        html += "<tr><td class='label'>CVD Avg CPW | API</td>"
        for _ in classifications:
            html += "<td>-</td>"
        html += "<td class='metric'>-</td><td class='metric'>-</td><td class='metric'>-</td></tr>"
    
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
# ----- ✅ Corrected Metrics Calculation -----

            tiers = ['Premium', 'Mainstream', 'Value']
            classifications = sorted(full_df['Classification'].unique())
            
            sku_matrix = {tier: {cls: [] for cls in classifications} for tier in tiers}
            classification_metrics = {}
            tier_metrics = {}
            
            total_present_sales = full_df['Present Net Sales'].sum()
            
            # --- Classification-level metrics
            for cls in classifications:
                our_cls_df = company_df[company_df['Classification'] == cls]
                all_cls_df = full_df[full_df['Classification'] == cls]  # still used for PPW range
            
                prev_rev = our_cls_df['Previous Net Sales'].sum()
                curr_rev = our_cls_df['Present Net Sales'].sum()
            
                growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                share = (curr_rev / company_df['Present Net Sales'].sum() * 100) if company_df['Present Net Sales'].sum() else 0
            
                ppw_range = f"{all_cls_df['Price per Wash'].min():.2f} – {all_cls_df['Price per Wash'].max():.2f}" if not all_cls_df.empty else "-"
            
                classification_metrics[cls] = {
                    "Growth": f"{growth:.1f}%",
                    "Value": f"{share:.1f}%",
                    "PPW": ppw_range
            }
            # --- Tier-level metrics
            for tier in tiers:
                tier_df = full_df[full_df["Calculated Price Tier"] == tier]
                prev_rev = tier_df["Previous Net Sales"].sum()
                curr_rev = tier_df["Present Net Sales"].sum()
                
                growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                share = (curr_rev / total_present_sales * 100) if total_present_sales else 0
                
                min_ppw = tier_df["Price per Wash"].min()
                max_ppw = tier_df["Price per Wash"].max()
                ppw_range = f"{currency_symbol}{min_ppw:.2f} – {currency_symbol}{max_ppw:.2f}" if not tier_df.empty else "-"
                
                tier_metrics[tier] = {
                    "PPW": ppw_range,
                    "Growth": f"{growth:.1f}%",
                    "Share": f"{share:.1f}%"
                }
            
            # --- Also rebuild SKU matrix (this is fine, no change needed)
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



# ------- 📊 NEW BLACK MATRIX LAYOUT (PLACEHOLDER) --------





           
