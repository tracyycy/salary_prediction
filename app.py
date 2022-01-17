import streamlit as st
import pandas as pd

import data
from predict import Request

st.set_page_config(layout="wide", page_title="Estimate your salary")

# sidebar

st.sidebar.header("Estimate your salary here.")

job = st.sidebar.selectbox(
    "Which of the following describes your current job?",
    options=data.all_jobs
)

country = st.sidebar.selectbox(
     'Where do you work ?',
     options=data.all_countries
)


edu = st.sidebar.selectbox(
     'Which of the following best describes the highest level of formal education that you’ve completed?',
     data.all_edu
)

years = st.sidebar.slider(
     'How many years have you coded professionally (as a part of your work)?',
     0, 30
)

st.sidebar.markdown("For the following type of tech, which have you done extensive development work in over the past year?")

language = st.sidebar.multiselect(
     'Programming, scripting, and markup languages',
     data.all_languages
)

database = st.sidebar.multiselect(
     'Database environments',
     data.all_database
)

platform = st.sidebar.multiselect(
     'Cloud platforms',
     data.all_platform
)

webframe = st.sidebar.multiselect(
     'Web frameworks and libraries',
     data.all_webframe
)


# main area


item = Request(
    job=job, 
    country=country, 
    edu=edu, 
    years=years, 
    language=language, 
    database=database, 
    platform=platform, 
    webframe=webframe
)

st.title(job + " in " + country)

col1, col2 = st.columns(2)
col1.metric("Estimated Ratio of Annual Salary to GNP", round(item.predict(),3))
salary = item.predict() * item.request['gnp']
col2.metric("Estimated Annual Salary in USD", "${:,.0f}".format(salary))

def hide_na(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'white' if pd.isna(val) else 'black'
    return 'color: %s' % color

with st.expander("See inputs"):
    if st.checkbox('Show in table'):
        tmp = item.request_df.copy()
        float_or_cat = ['Region', 'gnp', 'role_adjustment']
        int_col = [col for col in tmp.columns if col not in float_or_cat ]
        for col in int_col:
            tmp[col] = tmp[col].astype(int).astype(str)
        st.dataframe(tmp.append(pd.Series(), ignore_index=True).style.applymap(hide_na))
    else:
        st.json(item.request)
        

col1, col2 = st.columns(2)
#col1.image(item.summary_plot)
col1.subheader("Effects of all features to the model")
col1.pyplot(item.summary_plot)

col2.subheader("Explanation of the single prediction")
col2.pyplot(item.waterfall())