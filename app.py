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

            company_df['Calculated Price Tier'] = company_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
            st.success("Price tiers assigned!")

            st.subheader("Classified SKUs")
            st.dataframe(company_df[["SKU", "Price per Wash", "Calculated Price Tier", "Classification"]])

        # Build PPA Matrix
        st.subheader("PPA Matrix View (SKUs per Tier × Classification)")

        # Get unique tiers and classifications
        tiers = ['Value', 'Mainstream', 'Premium', 'Others']
        classifications = sorted(company_df['Classification'].unique())

        # Create a grid layout using a DataFrame
        matrix = pd.DataFrame(index=tiers, columns=classifications)

        for tier in tiers:
            for classification in classifications:
                skus = company_df[
                    (company_df['Calculated Price Tier'] == tier) &
                    (company_df['Classification'] == classification)
                ]['SKU'].tolist()

                if skus:
                    matrix.at[tier, classification] = ", ".join(skus)
                else:
                    matrix.at[tier, classification] = "-"

        st.dataframe(matrix)

