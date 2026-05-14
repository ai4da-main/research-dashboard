import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Research Dashboard")

df = pd.read_csv("research.csv")

st.dataframe(df)

fig = px.bar(df, x="Category", y="Value")
st.plotly_chart(fig)