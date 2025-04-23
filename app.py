import streamlit as st
import pandas as pd

st.title("Price Pack Architecture Tool")

st.header("Upload Your Data")
company_file = st.file_uploader("Upload Your Company Data (CSV)", type="csv")
competitor_file = st.file_uploader("Upload Competitor Data (CSV)", type="csv")

company_cols = ["SKU", "Pack Size", "Price", "Number of Washes", "Brand Type", 
                "Classification", "Price Tier", "Parent Brand", 
                "Previous Volume", "Present Volume", 
                "Previous Revenue", "Present Revenue"]

competitor_cols = ["SKU", "Pack Size", "Price", "Number of Washes", "Brand Type", 
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
            padding: 6px;
            text-align: center;
            vertical-align: middle;
        }
        th {
            background-color: #0B2B66;
            color: white;
        }
        .header-class {
            background-color: #3E74BA;
            color: white;
            font-weight: bold;
        }
        .metric-cell {
            background-color: #CFE2F3;
        }
        .tier-label {
            background-color: #DAE8FC;
            font-weight: bold;
        }
    </style>

    <table>
        <tr>
            <th class="header-class">Classification</th>
    """
    for cls in classifications:
        html += f'<th colspan="3" class="header-class">{cls}</th>'
    # Use rowspan=3 here for top-right metrics
    html += '<th class="metric-cell" rowspan="3">Avg PP CPW</th>'
    html += '<th class="metric-cell" rowspan="3">Value Weight</th>'
    html += '<th class="metric-cell" rowspan="3">Growth</th></tr>'

    # Row 1: Revenue Growth
    html += "<tr><td class='header-class'>Revenue Growth %</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Growth"]}</td>'
    html += "</tr>"

    # Row 2: Value Share
    html += "<tr><td class='header-class'>Value Share %</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["Value"]}</td>'
    html += "</tr>"

    # Row 3: PPW Range
    html += "<tr><td class='header-class'>PPW Range</td>"
    for cls in classifications:
        html += f'<td colspan="3">{classification_metrics[cls]["PPW"]}</td>'
    html += "</tr>"

    # Tier rows with SKUs and tier metrics
    for tier in tiers:
        html += f'<tr><td class="tier-label">{tier}</td>'
        for cls in classifications:
            skus = sku_matrix[tier][cls]
            html += f'<td colspan="3">{"<br>".join(skus) if skus else "-"}</td>'
        html += f'<td class="metric-cell">{tier_metrics[tier]["PPW"]}</td>'
        html += f'<td class="metric-cell">{tier_metrics[tier]["Share"]}</td>'
        html += f'<td class="metric-cell">{tier_metrics[tier]["Growth"]}</td></tr>'
    html += "</table>"
    return html




if company_file and competitor_file:
    company_df = pd.read_csv(company_file)
    competitor_df = pd.read_csv(competitor_file)

    if company_df["Classification"].nunique() > 4:
        st.error("You have more than 4 classifications in your company data.")
    else:
        company_df["Price per Wash"] = company_df["Price"] / company_df["Number of Washes"]
        competitor_df["Price per Wash"] = competitor_df["Price"] / competitor_df["Number of Washes"]

        st.subheader("Price per Wash Range")
        st.write(f"Company: ₹{company_df['Price per Wash'].min():.2f} – ₹{company_df['Price per Wash'].max():.2f}")
        st.write(f"Competitor: ₹{competitor_df['Price per Wash'].min():.2f} – ₹{competitor_df['Price per Wash'].max():.2f}")

        st.subheader("Set Price Tier Thresholds (₹)")
        with st.form("thresholds"):
            col1, col2, col3 = st.columns(3)
            with col1:
                value_max = st.number_input("Value: Max ₹", min_value=0.0, value=1.50)
            with col2:
                mainstream_max = st.number_input("Mainstream: Max ₹", min_value=value_max, value=2.50)
            with col3:
                premium_max = st.number_input("Premium: Max ₹", min_value=mainstream_max, value=4.00)
            submit_btn = st.form_submit_button("Classify SKUs")

        if submit_btn:
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
            total_company_revenue = company_df['Present Revenue'].sum()

            for cls in classifications:
                all_cls = full_df[full_df['Classification'] == cls]
                our_cls = company_df[company_df['Classification'] == cls]
                prev_rev = our_cls['Previous Revenue'].sum()
                curr_rev = our_cls['Present Revenue'].sum()
                growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                share = (curr_rev / total_company_revenue * 100) if total_company_revenue else 0
                ppw_range = f"{all_cls['Price per Wash'].min():.2f} – {all_cls['Price per Wash'].max():.2f}" if not all_cls.empty else "-"
                classification_metrics[cls] = {
                    "Growth": f"{growth:.1f}%",
                    "Value": f"{share:.1f}%",
                    "PPW": ppw_range
                }

            for tier in tiers:
                tier_full = full_df[full_df["Calculated Price Tier"] == tier]
                tier_our = company_df[company_df["Calculated Price Tier"] == tier]
                prev = tier_our["Previous Revenue"].sum()
                curr = tier_our["Present Revenue"].sum()
                avg_ppw = tier_full["Price per Wash"].mean()
                growth = ((curr - prev) / prev * 100) if prev else 0
                share = (curr / total_company_revenue * 100) if total_company_revenue else 0
                tier_metrics[tier] = {
                    "PPW": f"₹{avg_ppw:.2f}" if not pd.isna(avg_ppw) else "-",
                    "Growth": f"{growth:.1f}%",
                    "Share": f"{share:.1f}%"
                }

            for _, row in full_df.iterrows():
                tier = row["Calculated Price Tier"]
                cls = row["Classification"]
                sku = f"{row['SKU']} ({'Comp' if row['Is Competitor'] else 'Our'})"
                if tier in sku_matrix and cls in sku_matrix[tier]:
                    sku_matrix[tier][cls].append(sku)

            dynamic_html = generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers)
            st.markdown(dynamic_html, unsafe_allow_html=True)
