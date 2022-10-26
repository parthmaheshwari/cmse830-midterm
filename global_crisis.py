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

# c = alt.Chart(df[df["country"]==country]).mark_line().encode(
#     x = 'year',
#     y='inflation_annual_cpi'
#     ).interactive()

# st.altair_chart(c, use_container_width=True)

###
# Multi-Line chart 
###
country_df = df[df["country"]==country]
country_df.set_index("year", inplace=True)
source = country_df.drop(columns=["country","cc3","case"])
source = source.reset_index().melt('year', var_name='category', value_name='y')

# Create a selection that chooses the nearest point & selects based on x-value
nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['year'], empty='none')

# The basic line
line = alt.Chart(source).mark_line(interpolate='basis').encode(
    x='year:T',
    y='y:Q',
    color='category:N',
)

# Transparent selectors across the chart. This is what tells us
# the x-value of the cursor
selectors = alt.Chart(source).mark_point().encode(
    x='year:T',
    opacity=alt.value(0),
).add_selection(
    nearest
)

# Draw points on the line, and highlight based on selection
points = line.mark_point().encode(
    opacity=alt.condition(nearest, alt.value(1), alt.value(0))
)

# Draw text labels near the points, and highlight based on selection
text = line.mark_text(align='left', dx=5, dy=-5).encode(
    text=alt.condition(nearest, 'y:Q', alt.value(' '))
)

# Draw a rule at the location of the selection
rules = alt.Chart(source).mark_rule(color='gray').encode(
    x='year:T',
).transform_filter(
    nearest
)

# Put the five layers into a chart and bind the data
c = alt.layer(
    line, selectors, points, rules, text
).properties(
    width=600, height=300
).interactive()

st.altair_chart(c, use_container_width=True)
