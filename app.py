import streamlit as st
import pandas as pd
import io

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
st.title("\U0001F4E6 Price Pack Architecture Tool")

st.markdown("Before uploading, please use the templates below to prepare your data:")

col1, col2 = st.columns(2)

with col1:
    company_buffer = generate_excel_download(pd.DataFrame(columns=company_template_cols))
    st.download_button(
        label="\U0001F4E5 Download Company Template",
        data=company_buffer,
        file_name="company_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

with col2:
    competitor_buffer = generate_excel_download(pd.DataFrame(columns=competitor_template_cols))
    st.download_button(
        label="\U0001F4E5 Download Competitor Template",
        data=competitor_buffer,
        file_name="competitor_data_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


st.header("Upload Your Data")
company_file = st.file_uploader("Upload Your Company Data (CSV)", type="csv")
competitor_file = st.file_uploader("Upload Competitor Data (CSV)", type="csv")

def generate_clean_matrix_html(classifications):
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

    html += "<tr><td class='header'>Unilever Value Growth</td>"
    for _ in classifications:
        html += "<td></td>"
    html += "<td rowspan='4'>Avg PP CPW</td><td rowspan='4'>Value Weight</td><td rowspan='4'>Growth</td></tr>"

    html += "<tr><td class='header'>Unilever Value Weight</td>"
    for _ in classifications:
        html += "<td></td>"
    html += "</tr>"

    html += "<tr><td class='header'>CVD</td>"
    for i in range(len(classifications)):
        html += f"<td rowspan='2'>Classification {i+1}</td>"
    html += "</tr>"

    html += "<tr><td class='header'>RSV Price Point</td></tr>"

    for tier in ["Premium", "Mainstream", "Value"]:
        html += f"<tr><td class='header'>{tier}</td>"
        for _ in classifications:
            html += "<td></td>"
        html += "<td rowspan='2'></td><td rowspan='2'></td><td rowspan='2'></td></tr>"

        html += f"<tr><td class='header'>Unilever Shelf Space Percentage</td><td colspan='{len(classifications)}'></td></tr>"

    html += "<tr><td class='header'>CVD Avg CPW | API</td>"
    for _ in classifications:
        html += "<td></td>"
    html += "<td></td><td></td><td></td></tr>"

    html += "</table>"
    return html

if st.button("\U0001F50D Preview Dummy Matrix Layout"):
    dummy_classifications = ["Classification 1", "Classification 2", "Classification 3"]
    dummy_matrix = generate_clean_matrix_html(dummy_classifications)
    st.markdown(dummy_matrix, unsafe_allow_html=True)
