import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt


df = pd.read_csv("african_crises.csv")

#Preprocessing
data = df[["exch_usd", "inflation_annual_cpi", "year", "systemic_crisis","gdp_weighted_default","country"]]
data = data[(data["inflation_annual_cpi"]<100)]
# df = data.drop('systemic_crisis', axis=1)
# df_norm = (df-df.min())/(df.max()-df.min())
# df_norm = pd.concat((df_norm, data.systemic_crisis), 1)
df["banking_crisis"][df["banking_crisis"]=="crisis"] = 1
df["banking_crisis"][df["banking_crisis"]=="no_crisis"] = 0
df["banking_crisis"] = pd.to_numeric(df["banking_crisis"])
df["year"] = pd.to_datetime(df.year, format='%Y')
countries = list(df["country"].unique())
st.write("""
# Global Crises Data by Country
How different macroeconomics factor can help us predict systemic crisis in different countries. 
Shown below: Africa 
""")
# fig = px.scatter_3d(data, x = 'exch_usd', 
#                     y = 'inflation_annual_cpi', 
#                     z = 'year',
#                     color = 'country',
#                     size_max = 20, 
#                     opacity = 0.5)

# st.plotly_chart(fig, use_container_width=True)


country = st.selectbox("Select a column for distribution plot: ",countries)
c = alt.Chart(df[df["country"]==country]).mark_line().encode(
    x = 'year',
    y='inflation_annual_cpi'
    ).interactive()

st.altair_chart(c, use_container_width=True)
