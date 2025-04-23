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
                        writing-mode: vertical-lr;
                        text-orientation: upright;
                    }
                </style>
                
                <table>
                    <tr>
                        <th rowspan="4" colspan="2">Customer Value<br>Growth</th>
                        <th colspan="3" class="header-class">Classification 1</th>
                        <th colspan="3" class="header-class">Classification 2</th>
                        <th colspan="3" class="header-class">Classification 3</th>
                        <th colspan="3" class="header-class">Classification 4</th>
                        <th rowspan="4" class="metric-cell">Avg PP CPW</th>
                        <th rowspan="4" class="metric-cell">Value Weight</th>
                        <th rowspan="4" class="metric-cell">Growth</th>
                    </tr>
                    <tr>
                        <td colspan="3">16.7%</td><td colspan="3">6.4%</td><td colspan="3">5.8%</td><td colspan="3">9.2%</td>
                    </tr>
                    <tr>
                        <td colspan="3">5.0%</td><td colspan="3">13.9%</td><td colspan="3">81.1%</td><td colspan="3">0.0%</td>
                    </tr>
                    <tr>
                        <td colspan="3">2.00 – 2.50</td><td colspan="3">1.80 – 2.50</td><td colspan="3">1.90 – 2.70</td><td colspan="3">1.50 – 2.00</td>
                    </tr>
                    <tr>
                        <td rowspan="3" class="tier-label">Premium</td>
                        <td class="metric-cell">2.20 – 2.50</td>
                        <td>SKU6 (Our)</td><td>COMP3 (Comp)</td><td>-</td>
                        <td>COMP4 (Comp)</td><td>-</td><td>-</td>
                        <td>SKU5 (Our)</td><td>-</td><td>-</td>
                        <td>-</td><td>-</td><td>-</td>
                        <td class="metric-cell" rowspan="3">₹2.42</td>
                        <td class="metric-cell" rowspan="3">72.9%</td>
                        <td class="metric-cell" rowspan="3">5.7%</td>
                    </tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                
                    <tr>
                        <td rowspan="3" class="tier-label">Mainstream</td>
                        <td class="metric-cell">1.80 – 2.00</td>
                        <td>-</td><td>-</td><td>-</td>
                        <td>SKU3 (Our)</td><td>SKU4 (Our)</td><td>COMP2 (Comp)</td>
                        <td>SKU1 (Our)</td><td>SKU2 (Our)</td><td>COMP1 (Comp)</td>
                        <td>-</td><td>-</td><td>-</td>
                        <td class="metric-cell" rowspan="3">₹1.97</td>
                        <td class="metric-cell" rowspan="3">27.1%</td>
                        <td class="metric-cell" rowspan="3">8.3%</td>
                    </tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                
                    <tr>
                        <td rowspan="3" class="tier-label">Value</td>
                        <td class="metric-cell">-</td>
                        <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
                        <td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td>
                        <td class="metric-cell" rowspan="3">-</td>
                        <td class="metric-cell" rowspan="3">0.0%</td>
                        <td class="metric-cell" rowspan="3">0.0%</td>
                    </tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                    <tr><td class="metric-cell">-</td><td colspan="12"> </td></tr>
                </table>
                """
                
                st.markdown(html, unsafe_allow_html=True)


