import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt


def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)  # Remove {currency_symbol}, commas, $, spaces
        .replace("", pd.NA)
        .astype(float)
    )

# Define template headers
company_template_cols = [
    "Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
    "Number of SKUs on Shelf", "Number of Washes",
    "Previous Volume", "Present Volume",
    "Previous Net Sales", "Present Net Sales"
]

competitor_template_cols = [
    "Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
    "Number of SKUs on Shelf", "Number of Washes"
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
threshold_file = st.file_uploader("Upload Category-Wise Thresholds CSV", type="csv")

if threshold_file is not None:
    thresholds_df = pd.read_csv(threshold_file)
    thresholds_df.columns = thresholds_df.columns.str.strip()  # clean whitespace

    required_cols = {"Category", "Value Max Threshold", "Mainstream Max Threshold"}
    if not required_cols.issubset(set(thresholds_df.columns)):
        st.error("Threshold CSV must contain: 'Category', 'Value Max Threshold', 'Mainstream Max Threshold'")
        st.stop()

    for col in ["Value Max Threshold", "Mainstream Max Threshold"]:
        thresholds_df[col] = clean_numeric(thresholds_df[col])
else:
    st.warning("Please upload the Threshold file.")
    st.stop()


company_cols = [ "Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
    "Number of SKUs on Shelf", "Number of Washes",
    "Previous Volume", "Present Volume",
    "Previous Net Sales", "Present Net Sales"]

competitor_cols = ["Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
    "Number of SKUs on Shelf", "Number of Washes"]

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

def generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers, shelf_space_share):
    html = """
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
            font-size: 13px;
        }
        th, td {
            border: 1px solid #000;
            padding: 8px;
            text-align: center;
        }
        .header {
            font-weight: bold;
        }
    </style>
    <table>
    """

    # Row 1: Unilever Net Sales Growth
    html += "<tr><td class='header'>Unilever Net Sales Growth Percentage</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['Growth']}</td>"
    html += "<td rowspan='4'>Avg PP CPW</td><td rowspan='4'>Value Weight</td><td rowspan='4'>Growth</td></tr>"

    # Row 2: Unilever Value Share
    html += "<tr><td class='header'>Unilever Value Share %</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['Value']}</td>"
    html += "</tr>"

    # Row 3: CVD Headers
    html += "<tr><td class='header'>CVD</td>"
    for cls in classifications:
        html += f"<td rowspan='2'><b>{cls}</b></td>"
    html += "</tr>"

    # Row 4: RSV Label
    html += "<tr><td class='header'>RSV Price Point</td></tr>"

    # Rows for each Tier
    for tier in tiers:
        # Tier SKUs row
        html += f"<tr><td>{tier}</td>"
        for cls in classifications:
            skus = sku_matrix[tier][cls]
            sku_list = "<br>".join(skus) if skus else "-"
            html += f"<td>{sku_list}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['PPW']}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['Share']}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['Growth']}</td></tr>"

        # Tier Shelf Space row
        html += f"<tr><td><i>Unilever Shelf Space Percentage</i></td>"
        html += f"<td colspan='{len(classifications)}'>{shelf_space_share[tier]}</td></tr>"

    # Final Row: Classification PPW ranges
    html += "<tr><td class='header'>CVD Avg CPW</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['PPW']}</td>"
    html += "<td></td><td></td><td></td></tr>"

    html += "</table>"
    return html





if 'classified' not in st.session_state:
    st.session_state.classified = False


if company_file and competitor_file:
    st.subheader("Currency Settings")
    currency_symbol = st.text_input("Enter your currency symbol (e.g. ₹, $, €, etc.):", value="₹")
    company_df = pd.read_csv(company_file)
    competitor_df = pd.read_csv(competitor_file)

    # Always clean numeric fields immediately
    for col in ["Price", "Number of Washes", "Previous Volume", "Present Volume", "Previous Net Sales", "Present Net Sales"]:
        company_df[col] = clean_numeric(company_df[col])
    for col in ["Price", "Number of Washes"]:
        competitor_df[col] = clean_numeric(competitor_df[col])

        
        # Calculate Price per Wash
        company_df["Price per Wash"] = company_df["Price"] / company_df["Number of Washes"]
        competitor_df["Price per Wash"] = competitor_df["Price"] / competitor_df["Number of Washes"]

        # Ensure 'Category' exists
        if 'Category' not in company_df.columns or 'Category' not in competitor_df.columns:
            st.error("Make sure 'Category' column is present in both company and competitor data.")
            st.stop()
        
        # Merge & tag
        company_df['Is Competitor'] = False
        competitor_df['Is Competitor'] = True
        full_df = pd.concat([company_df, competitor_df], ignore_index=True)
        
        # Get list of unique categories
        categories = sorted(full_df['Category'].dropna().unique())
        
      
        st.session_state['classified'] = True
    
    
        if st.session_state['classified']:
            

            all_categories = company_df["Category"].unique()
        
            for category in all_categories:
                st.header(f"📂 Category: {category}")
                row = thresholds_df[thresholds_df["Category"] == category]
                if row.empty:
                    st.warning(f"No thresholds found for category '{category}'. Skipping.")
                    continue

            value_max = row["Value Max Threshold"].values[0]
            mainstream_max = row["Mainstream Max Threshold"].values[0]
            thresholds = {
                'Value': (0.0, value_max),
                'Mainstream': (value_max, mainstream_max),
                'Premium': (mainstream_max, float('inf'))
            }
        
                company_cat = company_df[company_df["Category"] == category].copy()
                competitor_cat = competitor_df[competitor_df["Category"] == category].copy()

                if company_cat["Classification"].nunique() > 4:
                    st.error(f"Category {category} has more than 4 classifications. Please fix the input data.")
                    continue  # Skip this category
        
                company_cat["Calculated Price Tier"] = company_cat["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
                competitor_cat["Calculated Price Tier"] = competitor_cat["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
                company_cat['Is Competitor'] = False
                competitor_cat['Is Competitor'] = True
        
                full_cat = pd.concat([company_cat, competitor_cat], ignore_index=True)
                tiers = ['Premium', 'Mainstream', 'Value']
                classifications = sorted(full_cat['Classification'].unique())
        
                sku_matrix = {tier: {cls: [] for cls in classifications} for tier in tiers}
                classification_metrics = {}
                tier_metrics = {}
        
                total_present_sales = full_cat['Present Net Sales'].sum()
        
                for cls in classifications:
                    our_cls_df = company_cat[company_cat['Classification'] == cls]
                    all_cls_df = full_cat[full_cat['Classification'] == cls]
        
                    prev_rev = our_cls_df['Previous Net Sales'].sum()
                    curr_rev = our_cls_df['Present Net Sales'].sum()
        
                    growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                    share = (curr_rev / company_cat['Present Net Sales'].sum() * 100) if company_cat['Present Net Sales'].sum() else 0
        
                    ppw_range = f"{currency_symbol}{all_cls_df['Price per Wash'].min():.2f} – {currency_symbol}{all_cls_df['Price per Wash'].max():.2f}" if not all_cls_df.empty else "-"
        
                    classification_metrics[cls] = {
                        "Growth": f"{growth:.1f}%",
                        "Value": f"{share:.1f}%",
                        "PPW": ppw_range
                    }
        
                for tier in tiers:
                    tier_df = full_cat[full_cat["Calculated Price Tier"] == tier]
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
        
                shelf_space_share = {}
                for tier in tiers:
                    company_skus = company_cat[company_cat["Calculated Price Tier"] == tier]
                    all_skus = full_cat[full_cat["Calculated Price Tier"] == tier]
        
                    unilever_shelf = company_skus["Number of SKUs on Shelf"].sum()
                    total_shelf = all_skus["Number of SKUs on Shelf"].sum()
        
                    shelf_pct = (unilever_shelf / total_shelf * 100) if total_shelf else 0
                    shelf_space_share[tier] = f"{shelf_pct:.1f}%"
        
                for _, row in full_cat.iterrows():
                    tier = row["Calculated Price Tier"]
                    cls = row["Classification"]
                    sku = row["SKU"]
                    if tier in sku_matrix and cls in sku_matrix[tier]:
                        sku_matrix[tier][cls].append(sku)
        
                dynamic_html = generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers, shelf_space_share)
                st.markdown(dynamic_html, unsafe_allow_html=True)
        
                # SKU Growth Summary
                st.subheader(f"📈 SKU-Level Growth Summary ({category})")
        
                sku_growth_summary = []
                for _, row in company_cat.iterrows():
                    sku = row['SKU']
                    prev_vol = row['Previous Volume']
                    curr_vol = row['Present Volume']
                    prev_rev = row['Previous Net Sales']
                    curr_rev = row['Present Net Sales']
        
                    volume_growth = ((curr_vol - prev_vol) / prev_vol * 100) if prev_vol else 0
                    net_sales_growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
        
                    sku_growth_summary.append({
                        "SKU": sku,
                        "Previous Volume": prev_vol,
                        "Present Volume": curr_vol,
                        "Volume Growth %": f"{volume_growth:.1f}%",
                        "Previous Net Sales": prev_rev,
                        "Present Net Sales": curr_rev,
                        "Net Sales Growth %": f"{net_sales_growth:.1f}%"
                    })
        
                st.dataframe(pd.DataFrame(sku_growth_summary))
        
