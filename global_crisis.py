import streamlit as st
import seaborn as sns
import pandas as pd
import seaborn as sns
import plotly.express as px
import altair as alt
import matplotlib.pyplot as plt


df = pd.read_csv("african_crises.csv")

#Preprocessing
# data = df[["exch_usd", "inflation_annual_cpi", "year", "systemic_crisis","gdp_weighted_default","country"]]
# data = data[(data["inflation_annual_cpi"]<100)]
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

crisis_options = st.multiselect(
    'Types of crises:',
    ['banking_crisis', 'systemic_crisis', 'inflation_crises','currency_crises'])

eco_ops = st.multiselect(
    'Macroeconomic parameters:',
    ['exch_usd', 'domestic_debt_in_default', 'sovereign_external_debt_default', 'gdp_weighted_default', 'inflation_annual_cpi', 'independence'])

agree = st.checkbox('Scaling enabled')

if agree:
    df["banking_crisis"]*=10
    df["currency_crises"]*=10
    df["inflation_crises"]*=10
    df["systemic_crisis"]*=10
    df["sovereign_external_debt_default"]*=10
    df["domestic_debt_in_default"]*=10

###
# Multi-Line chart 
###
country_df = df[df["country"]==country]
country_df.set_index("year", inplace=True)
source = country_df.drop(columns=["country","cc3","case"])
column_list = [i for i in list(country_df) if i not in []]
source = source[crisis_options+eco_ops]
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

values = st.slider(
    'Select a range of years',
    1870, 2013, (1870, 2013))

selector = alt.selection_single(encodings=['x', 'color'])
df_y = df[(df["year"]>=f"01-01-{values[0]}")&(df["year"]<=f"01-01-{values[-1]}")]
c2 = alt.Chart(df_y).transform_fold(
  ['currency_crises', 'inflation_crises','systemic_crisis', 'banking_crisis'],
  as_=['column', 'value']
).mark_bar().encode(
  x='country:N',
  y='sum(value):Q',
  color=alt.condition(selector, 'column:N', alt.value('lightgray'))
).add_selection(
    selector
).interactive()

st.altair_chart(c2, use_container_width=True)

