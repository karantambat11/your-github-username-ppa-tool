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

            # Classify both company and competitor data
            company_df['Calculated Price Tier'] = company_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
            competitor_df['Calculated Price Tier'] = competitor_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
            company_df['Is Competitor'] = False
            competitor_df['Is Competitor'] = True

            st.success("Price tiers assigned!")

            st.subheader("Classified SKUs (Your Company)")
            st.dataframe(company_df[["SKU", "Price per Wash", "Calculated Price Tier", "Classification"]])

            # Combine both datasets
            full_df = pd.concat([company_df, competitor_df], ignore_index=True)
            
            tiers = ['Premium', 'Mainstream', 'Value']  # Ordering from top down like your image
            classifications = sorted(full_df['Classification'].unique())
            
            # Calculate metric rows for classification columns
            ppw_range_row = []
            growth_row = []
            value_share_row = []
            
            total_company_revenue = company_df['Present Revenue'].sum()
            
            for classification in classifications:
                all_ppw = full_df[full_df['Classification'] == classification]['Price per Wash']
                company_subset = company_df[company_df['Classification'] == classification]
                
                ppw_range_row.append(
                    f"{all_ppw.min():.2f} – {all_ppw.max():.2f}" if not all_ppw.empty else "-"
                )
                prev_rev = company_subset['Previous Revenue'].sum()
                curr_rev = company_subset['Present Revenue'].sum()
                growth_pct = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                value_pct = (curr_rev / total_company_revenue * 100) if total_company_revenue else 0
                growth_row.append(f"{growth_pct:.1f}%")
                value_share_row.append(f"{value_pct:.1f}%")
            
            # Start building HTML table
            html = """
            <style>
                table { border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; font-size: 12px; }
                th, td { border: 1px solid #ccc; padding: 8px; text-align: center; }
                th { background-color: #0B2B66; color: white; }
                .light-blue { background-color: #CFE2F3; }
                .mid-blue { background-color: #3E74BA; color: white; font-weight: bold; }
            </style>
            <table>
            <tr>
                <th rowspan="3" colspan="2">Customer Value<br>Growth</th>
            """
            
            for cls in classifications:
                html += f'<th colspan="1" class="mid-blue">{cls}</th>'
            html += '<th rowspan="3" class="light-blue">Avg PP CPW</th><th rowspan="3" class="light-blue">Value Weight</th><th rowspan="3" class="light-blue">Growth</th></tr>'
            
            # Top rows for classification metrics
            html += "<tr>" + "".join(f"<td>{g}</td>" for g in growth_row) + "</tr>"
            html += "<tr>" + "".join(f"<td>{v}</td>" for v in value_share_row) + "</tr>"
            html += "<tr><td class='mid-blue'>Avg PP CPW</td>" + "".join(f"<td>{p}</td>" for p in ppw_range_row) + "<td colspan='3'></td></tr>"
            
            # Tier-based matrix with left-side metrics and SKUs
            for tier in tiers:
                tier_df = full_df[full_df["Calculated Price Tier"] == tier]
                tier_company_df = company_df[company_df["Calculated Price Tier"] == tier]
                tier_ppw = tier_df["Price per Wash"]
                avg_ppw = f"₹{tier_ppw.mean():.2f}" if not tier_ppw.empty else "-"
                price_range = f"{tier_ppw.min():.2f} – {tier_ppw.max():.2f}" if not tier_ppw.empty else "-"
                prev_rev = tier_company_df["Previous Revenue"].sum()
                curr_rev = tier_company_df["Present Revenue"].sum()
                growth = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev else 0
                share = (curr_rev / total_company_revenue * 100) if total_company_revenue else 0
            
                html += f"<tr><td class='light-blue' rowspan='1'>{tier}</td><td class='light-blue'>{price_range}</td>"
                for classification in classifications:
                    skus = tier_df[tier_df["Classification"] == classification]
                    sku_list = [f"{r['SKU']} ({'Comp' if r['Is Competitor'] else 'Our'})" for _, r in skus.iterrows()]
                    html += f"<td>{', '.join(sku_list) if sku_list else '-'}</td>"
                html += f"<td class='light-blue'>{avg_ppw}</td>"
                html += f"<td class='light-blue'>{share:.1f}%</td>"
                html += f"<td class='light-blue'>{growth:.1f}%</td></tr>"
            
            html += "</table>"
            st.markdown(html, unsafe_allow_html=True)

# ----- CLASSIFICATION METRICS -----
st.subheader("Classification Summary (Our Data Only)")

classification_summary = []

total_present_revenue = company_df['Present Revenue'].sum()

for classification in classifications:
    subset = company_df[company_df['Classification'] == classification]
    prev_rev = subset['Previous Revenue'].sum()
    curr_rev = subset['Present Revenue'].sum()

    growth_pct = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev != 0 else 0
    value_share = (curr_rev / total_present_revenue * 100) if total_present_revenue != 0 else 0

    classification_summary.append({
        "Classification": classification,
        "Revenue Growth %": f"{growth_pct:.1f}%",
        "Value Share %": f"{value_share:.1f}%"
    })

st.dataframe(pd.DataFrame(classification_summary))

# ----- PRICE TIER METRICS -----
st.subheader("Price Tier Summary (Our Data Only)")

tier_summary = []
tiers = ['Value', 'Mainstream', 'Premium', 'Others']
total_present_revenue = company_df['Present Revenue'].sum()

for tier in tiers:
    subset = company_df[company_df['Calculated Price Tier'] == tier]
    avg_ppw = subset['Price per Wash'].mean()
    curr_rev = subset['Present Revenue'].sum()
    prev_rev = subset['Previous Revenue'].sum()
    growth_pct = ((curr_rev - prev_rev) / prev_rev * 100) if prev_rev != 0 else 0
    revenue_share = (curr_rev / total_present_revenue * 100) if total_present_revenue != 0 else 0

    tier_summary.append({
        "Price Tier": tier,
        "Avg Price per Wash": f"₹{avg_ppw:.2f}" if not pd.isna(avg_ppw) else "-",
        "Revenue Share %": f"{revenue_share:.1f}%",
        "Growth %": f"{growth_pct:.1f}%"
    })

st.dataframe(pd.DataFrame(tier_summary))

