import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px

data = pd.read_csv("african_crises.csv")
# data = data[["exch_usd", "inflation_annual_cpi", "year", "systemic_crisis"]]
# df = data.drop('systemic_crisis', axis=1)
# df_norm = (df-df.min())/(df.max()-df.min())
# df_norm = pd.concat((df_norm, data.systemic_crisis), 1)

st.write("""
# Global Crises Data by Country
How different macroeconomics factor can help us predict systemic crisis in different countries. 
Shown below: Africa 
""")
fig = px.scatter_3d(data, x = 'exch_usd', 
                    y = 'inflation_annual_cpi', 
                    z = 'year',
                    color = 'systemic_crisis', 
                    size_max = 20, 
                    opacity = 0.5)

st.plotly_chart(fig, use_container_width=True)