import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px

df = pd.read_csv("african_crises.csv")

# df = data.drop('systemic', axis=1)
# df_norm = (df-df.min())/(df.max()-df.min())
# df_norm = pd.concat((df_norm, data.species), 1)

st.write("""
# Global Crises Data by Country
How different macroeconomics factor can help us predict systemic crisis in different countries. 
Shown below: Africa 
""")
fig = px.scatter_3d(df, x = 'exch_usd', 
                    y = 'inflation_annual_cpi', 
                    z = 'gdp_weighted_default',
                    color = 'systemic_crisis', 
                    size_max = 20, 
                    opacity = 0.5)

st.plotly_chart(fig, use_container_width=True)