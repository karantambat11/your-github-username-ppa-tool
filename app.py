import streamlit as st
import pandas as pd
import io
import matplotlib.pyplot as plt

# ---- Functions ----
def clean_numeric(series):
    return (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)  # Remove currency, commas, etc.
        .replace("", pd.NA)
        .astype(float)
    )

def assign_tier(ppw, thresholds):
    if ppw <= thresholds['Value'][1]:
        return 'Value'
    elif ppw <= thresholds['Mainstream'][1]:
        return 'Mainstream'
    elif ppw <= thresholds['Premium'][1]:
        return 'Premium'
    else:
        return 'Others'

def generate_excel_download(df: pd.DataFrame):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name="Template")
    buffer.seek(0)
    return buffer

def generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers, shelf_space_share):
    html = """
    <style>
        table {border-collapse: collapse;width: 100%;font-family: Arial, sans-serif;font-size: 13px;}
        th, td {border: 1px solid #000;padding: 8px;text-align: center;}
        .header {font-weight: bold;}
    </style>
    <table>
    """

    # Top Rows
    html += "<tr><td class='header'>Unilever Net Sales Growth Percentage</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['Growth']}</td>"
    html += "<td rowspan='4'>Avg PP CPW</td><td rowspan='4'>Value Weight</td><td rowspan='4'>Growth</td></tr>"

    html += "<tr><td class='header'>Unilever Value Share %</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['Value']}</td>"
    html += "</tr>"

    html += "<tr><td class='header'>CVD</td>"
    for cls in classifications:
        html += f"<td rowspan='2'><b>{cls}</b></td>"
    html += "</tr>"

    html += "<tr><td class='header'>RSV Price Point</td></tr>"

    # Tier Rows
    for tier in tiers:
        html += f"<tr><td>{tier}</td>"
        for cls in classifications:
            skus = sku_matrix[tier][cls]
            sku_list = "<br>".join(skus) if skus else "-"
            html += f"<td>{sku_list}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['PPW']}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['Share']}</td>"
        html += f"<td rowspan='2'>{tier_metrics[tier]['Growth']}</td></tr>"

        html += f"<tr><td><i>Unilever Shelf Space Percentage</i></td><td colspan='{len(classifications)}'>{shelf_space_share[tier]}</td></tr>"

    html += "<tr><td class='header'>CVD Avg CPW</td>"
    for cls in classifications:
        html += f"<td>{classification_metrics[cls]['PPW']}</td>"
    html += "<td></td><td></td><td></td></tr>"

    html += "</table>"
    return html

# ---- Streamlit App ----
st.title("📦 Price Pack Architecture Tool")

st.markdown("### Download templates to prepare your data:")

col1, col2 = st.columns(2)
with col1:
    company_buffer = generate_excel_download(pd.DataFrame(columns=[
        "Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
        "Number of SKUs on Shelf", "Number of Washes", "Previous Volume", "Present Volume",
        "Previous Net Sales", "Present Net Sales"
    ]))
    st.download_button("📥 Download Company Template", company_buffer, "company_data_template.xlsx")

with col2:
    competitor_buffer = generate_excel_download(pd.DataFrame(columns=[
        "Category", "Parent Brand", "SKU", "Pack Size", "Classification", "Price",
        "Number of SKUs on Shelf", "Number of Washes"
    ]))
    st.download_button("📥 Download Competitor Template", competitor_buffer, "competitor_data_template.xlsx")

# ---- File Uploads ----
st.header("📂 Upload Your Files")
company_file = st.file_uploader("Upload Company Data (CSV)", type="csv")
competitor_file = st.file_uploader("Upload Competitor Data (CSV)", type="csv")
threshold_file = st.file_uploader("Upload Thresholds CSV", type="csv")

if not (company_file and competitor_file and threshold_file):
    st.stop()

thresholds_df = pd.read_csv(threshold_file)
thresholds_df.columns = thresholds_df.columns.str.strip()

required_cols = {"Category", "Value Max Threshold", "Mainstream Max Threshold"}
if not required_cols.issubset(thresholds_df.columns):
    st.error("Threshold CSV must contain 'Category', 'Value Max Threshold', 'Mainstream Max Threshold'")
    st.stop()

for col in ["Value Max Threshold", "Mainstream Max Threshold"]:
    thresholds_df[col] = clean_numeric(thresholds_df[col])

# ---- Data Preparation ----
currency_symbol = st.text_input("Enter your currency symbol (e.g. ₹, $, €, etc.):", value="₹")

company_df = pd.read_csv(company_file)
competitor_df = pd.read_csv(competitor_file)

for col in ["Price", "Number of Washes", "Previous Volume", "Present Volume", "Previous Net Sales", "Present Net Sales"]:
    company_df[col] = clean_numeric(company_df[col])
for col in ["Price", "Number of Washes"]:
    competitor_df[col] = clean_numeric(competitor_df[col])

company_df["Price per Wash"] = company_df["Price"] / company_df["Number of Washes"]
competitor_df["Price per Wash"] = competitor_df["Price"] / competitor_df["Number of Washes"]

company_df['Is Competitor'] = False
competitor_df['Is Competitor'] = True
full_df = pd.concat([company_df, competitor_df], ignore_index=True)

# ✅ Assign price tier ONCE and for ALL categories
full_df["Calculated Price Tier"] = full_df.apply(
    lambda row: assign_tier(
        row["Price per Wash"],
        {
            'Value': (0.0, thresholds_df.loc[thresholds_df["Category"] == row["Category"], "Value Max Threshold"].max()),
            'Mainstream': (
                thresholds_df.loc[thresholds_df["Category"] == row["Category"], "Value Max Threshold"].max(),
                thresholds_df.loc[thresholds_df["Category"] == row["Category"], "Mainstream Max Threshold"].max(),
            ),
            'Premium': (
                thresholds_df.loc[thresholds_df["Category"] == row["Category"], "Mainstream Max Threshold"].max(),
                float("inf"),
            )
        }
    ),
    axis=1
)


# ---- Main Logic ----
categories = full_df["Category"].dropna().unique()

for category in categories:
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
        continue

    company_cat["Calculated Price Tier"] = company_cat["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))
    competitor_cat["Calculated Price Tier"] = competitor_cat["Price per Wash"].apply(lambda x: assign_tier(x, thresholds))

    full_cat = pd.concat([company_cat, competitor_cat], ignore_index=True)
    full_df["Calculated Price Tier"] = full_df["Price per Wash"].apply(lambda x: assign_tier(x, {
    'Value': (0.0, thresholds_df["Value Max Threshold"].max()),
    'Mainstream': (thresholds_df["Value Max Threshold"].max(), thresholds_df["Mainstream Max Threshold"].max()),
    'Premium': (thresholds_df["Mainstream Max Threshold"].max(), float('inf'))
    }))
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

    # ---- Show Matrix
    dynamic_html = generate_dynamic_html(sku_matrix, classification_metrics, tier_metrics, classifications, tiers, shelf_space_share)
    st.markdown(dynamic_html, unsafe_allow_html=True)

    # ---- Show SKU Growth Summary
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

        # ---- Scatter Plots per Parent Brand
    # ---- Scatter Plots per Parent Brand (Company + Competitor)
    st.subheader(f"📊 Scatter Plots by Parent Brand — {category} (All SKUs)")

    import matplotlib.pyplot as plt
    import numpy as np
    from adjustText import adjust_text

    brand_df_all = full_cat.copy()
    parent_brands = sorted(brand_df_all["Parent Brand"].dropna().unique())

    for brand in parent_brands:
        brand_df = brand_df_all[brand_df_all["Parent Brand"] == brand].copy()

        if brand_df.empty:
            continue

        # Add jitter
        brand_df['Jittered PPW'] = brand_df['Price per Wash'] + np.random.normal(0, 0.002, size=len(brand_df))
        brand_df['Jittered Price'] = brand_df['Price'] + np.random.normal(0, 0.3, size=len(brand_df))

        x_min = max(0, brand_df['Price per Wash'].min() - 0.03)
        x_max = brand_df['Price per Wash'].max() + 0.03
        y_min = max(0, brand_df['Price'].min() - 1)
        y_max = brand_df['Price'].max() + 2

        # Color by company
        brand_df['Color'] = brand_df['Is Competitor'].apply(lambda x: 'green' if x else 'navy')

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add shaded backgrounds for price tiers
        ax.axvspan(0, thresholds['Value'][1], facecolor='#5b6e9c', alpha=0.25, label='Value')
        ax.axvspan(thresholds['Value'][1], thresholds['Mainstream'][1], facecolor='#efad85', alpha=0.25, label='Mainstream')
        ax.axvspan(thresholds['Mainstream'][1], x_max, facecolor='#f7f3a0', alpha=0.25, label='Premium')

        # Create invisible plots to label shaded areas in legend
        value_patch = plt.Rectangle((0, 0), 1, 1, fc='#5b6e9c', alpha=1, label='Value')
        mainstream_patch = plt.Rectangle((0, 0), 1, 1, fc='#efad85', alpha=1, label='Mainstream')
        premium_patch = plt.Rectangle((0, 0), 1, 1, fc='#f7f3a0', alpha=1, label='Premium')
        
        ax.legend(handles=[value_patch, mainstream_patch, premium_patch], loc="upper left", bbox_to_anchor=(0, 1), title='Price Tiers')

        
        
        ax.scatter(brand_df['Jittered PPW'], brand_df['Jittered Price'], c=brand_df['Color'], s=70, alpha=0.8)

        # Add labels without overlap
        texts = [
            ax.text(row['Jittered PPW'], row['Jittered Price'], row['SKU'], fontsize=8)
            for _, row in brand_df.iterrows()
        ]

        adjust_text(
            texts,
            ax=ax,
            expand_text=(1.2, 1.4),
            expand_points=(1.2, 1.4),
            force_text=0.5,
            force_points=0.4,
            only_move={'points': 'y', 'text': 'xy'},
            arrowprops=dict(arrowstyle="-", color='gray', lw=0.5)
        )

        ax.set_xlabel("Price Per Wash")
        ax.set_ylabel("Retail Price")
        ax.set_title(f"{brand} — {category}")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle='--', alpha=0.5)

        st.pyplot(fig)
        # ---- End of Category Loop ----

        # ---- Combined Scatter Plots for Each Parent Brand Across All Categories
        # ---- Final Scatter Plots by Parent Brand (All SKUs Across Categories)
st.header("📊 Final View: All SKUs by Parent Brand (Color by Category)")

# Assign a unique color to each category
unique_categories = sorted(full_df["Category"].dropna().unique())
category_colors = plt.cm.get_cmap("tab10", len(unique_categories))
category_color_map = {cat: category_colors(i) for i, cat in enumerate(unique_categories)}

parent_brands_all = sorted(full_df["Parent Brand"].dropna().unique())

for brand in parent_brands_all:
    brand_df = full_df[full_df["Parent Brand"] == brand].copy()
    if brand_df.empty:
        continue

    # Jitter
    brand_df['Jittered PPW'] = brand_df['Price per Wash'] + np.random.normal(0, 0.002, size=len(brand_df))
    brand_df['Jittered Price'] = brand_df['Price'] + np.random.normal(0, 0.3, size=len(brand_df))

    x_min = max(0, brand_df['Price per Wash'].min() - 0.03)
    x_max = brand_df['Price per Wash'].max() + 0.03
    y_min = max(0, brand_df['Price'].min() - 1)
    y_max = brand_df['Price'].max() + 2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot points per category
    for cat in unique_categories:
        sub = brand_df[brand_df["Category"] == cat]
        if sub.empty:
            continue
        ax.scatter(sub["Jittered PPW"], sub["Jittered Price"], 
                   label=cat, alpha=0.7, s=70, 
                   color=category_color_map[cat])

    # Add labels
    texts = [
        ax.text(row['Jittered PPW'], row['Jittered Price'], row['SKU'], fontsize=8)
        for _, row in brand_df.iterrows()
    ]

    adjust_text(
        texts,
        ax=ax,
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.4),
        force_text=0.5,
        force_points=0.4,
        only_move={'points': 'y', 'text': 'xy'},
        arrowprops=dict(arrowstyle="-", color='gray', lw=0.5)
    )

    ax.set_xlabel("Price Per Wash")
    ax.set_ylabel("Retail Price")
    ax.set_title(f"{brand} — All SKUs Colored by Category")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(title="Category", loc="upper left", bbox_to_anchor=(0, 1))

    st.pyplot(fig)

    

    

# ---- 📈 Final Correct Price Movement Charts ----
# 📈 Price Movement Across Formats by Price Tier
st.header("📈 Price Tier Line Chart — Format Transition with BPS Labels")

tiers = ['Value', 'Mainstream', 'Premium']
format_order = ['Powder', 'Liquid', 'Capsules']

# Ensure Calculated Price Tier exists
if "Calculated Price Tier" not in full_df.columns:
    thresholds_values = {
        'Value': (0.0, thresholds_df["Value Max Threshold"].max()),
        'Mainstream': (thresholds_df["Value Max Threshold"].max(), thresholds_df["Mainstream Max Threshold"].max()),
        'Premium': (thresholds_df["Mainstream Max Threshold"].max(), float('inf'))
    }
    full_df["Calculated Price Tier"] = full_df["Price per Wash"].apply(lambda x: assign_tier(x, thresholds_values))

for tier in tiers:
    st.subheader(f"💠 {tier} Tier")

    tier_df = full_df[full_df["Calculated Price Tier"] == tier].copy()
    if tier_df.empty:
        st.info(f"No data available for {tier} tier.")
        continue

    fig, ax = plt.subplots(figsize=(10, 6))
    parent_brands = tier_df["Parent Brand"].dropna().unique()

    global_min_ppw = tier_df["Price per Wash"].min()
    y_min = max(0, global_min_ppw - 0.05)
    y_max = tier_df["Price per Wash"].max() + 0.05

    for brand in parent_brands:
        brand_df = tier_df[tier_df["Parent Brand"] == brand]
        avg_ppw = (
            brand_df.groupby("Category")["Price per Wash"]
            .mean()
            .reindex(format_order)
        )

        x_vals = [i for i, fmt in enumerate(format_order) if pd.notna(avg_ppw[fmt])]
        y_vals = [avg_ppw[fmt] for fmt in format_order if pd.notna(avg_ppw[fmt])]

        if len(x_vals) < 2:
            continue

        ax.plot(x_vals, y_vals, marker='o', label=brand)

        for i in range(1, len(x_vals)):
            bps_change = ((y_vals[i] - y_vals[i - 1]) / y_vals[i - 1]) * 10000
            ax.text(x_vals[i], y_vals[i] + 0.01, f"{bps_change:+.0f} BPS", ha='center', fontsize=8)

    ax.set_xticks(range(len(format_order)))
    ax.set_xticklabels(format_order)
    ax.set_xlabel("Format Category")
    ax.set_ylabel("Avg Price Per Wash")
    ax.set_title(f"{tier} Tier — Format Transition with BPS Change")
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(title="Parent Brand", fontsize=8, loc="upper left")
    st.pyplot(fig)





   
    
