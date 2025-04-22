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

            # Build PPA Matrix View
            st.subheader("PPA Matrix View (SKUs per Tier × Classification)")

            # Combine company and competitor data
            full_df = pd.concat([company_df, competitor_df], ignore_index=True)
            
            tiers = ['Value', 'Mainstream', 'Premium', 'Others']
            classifications = sorted(full_df['Classification'].unique())
            
            # Initialize matrix with one extra row for PPW Range
            matrix_data = {}
            
            # Add the "PPW Range" top row first
            ppw_top_row = {}
            for classification in classifications:
                ppw_vals = full_df[full_df['Classification'] == classification]['Price per Wash']
                if not ppw_vals.empty:
                    ppw_top_row[classification] = f"{ppw_vals.min():.2f} – {ppw_vals.max():.2f}"
                else:
                    ppw_top_row[classification] = "-"
            
            matrix_data["PPW Range (₹)"] = ppw_top_row
            
            # Now build the matrix for each price tier
            for tier in tiers:
                row = {}
                tier_df = full_df[full_df['Calculated Price Tier'] == tier]
            
                for classification in classifications:
                    subset = tier_df[tier_df['Classification'] == classification]
                    sku_list = [
                        f"{row['SKU']} ({'Comp' if row['Is Competitor'] else 'Our'})"
                        for _, row in subset.iterrows()
                    ]
                    row[classification] = ", ".join(sku_list) if sku_list else "-"
            
                # Add PPW range for the tier in the first column
                ppw_vals = tier_df["Price per Wash"]
                row["PPW Range (₹)"] = f"{ppw_vals.min():.2f} – {ppw_vals.max():.2f}" if not ppw_vals.empty else "-"
                matrix_data[tier] = row
            
            # Create DataFrame and rearrange columns
            matrix_df = pd.DataFrame(matrix_data).T  # transpose to match desired layout
            cols = ["PPW Range (₹)"] + classifications
            matrix_df = matrix_df[cols]
            
            st.dataframe(matrix_df)
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

