import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from statsmodels.formula.api import ols
from scipy import stats
from tabulate import tabulate
from io import StringIO


def main():
    selectbox = load_sidebar()

    if selectbox == "ANOVA":
        st.title('ANOVA (Analysis Of Variance)')
    elif selectbox == "Linear Regression":
        st.title('Linear Regression')

    if selectbox == "ANOVA":
        uploaded_file = st.file_uploader("Choose a CSV file.")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            if not st.checkbox("Hide dataframe"):
                st.write(dataframe)
            group_col = st.sidebar.selectbox(
                "Treatment (Group)",
                (col for col in dataframe.columns)
            )
            value_col = st.sidebar.selectbox(
                "Value",
                (col for col in dataframe.columns)
            )

            if st.sidebar.button('Run'):
                table = get_anova_table(dataframe, group_col, value_col)
                st.write(table)

                st.set_option('deprecation.showPyplotGlobalUse', False)
                fig, ax = plt.subplots()
                sns.boxplot(
                    x=group_col, y=value_col, data=dataframe, palette="Set2"
                )
                plt.grid()
                st.pyplot(fig)

    if selectbox == "Linear Regression":
        uploaded_file = st.file_uploader("Choose a CSV file.")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            if not st.checkbox("Hide dataframe"):
                st.write(dataframe)
            st.sidebar.text("Select features:")

            features_cols = []
            for col in dataframe.columns:
                option = st.sidebar.checkbox(col)
                if option:
                    features_cols.append(col)
            label_col = st.sidebar.selectbox(
                "Select label",
                (col for col in dataframe.columns)
            )

            if st.sidebar.button('Run'):
                features, labels = dataframe[features_cols], dataframe[label_col]
                scaler = MinMaxScaler()
                features = scaler.fit_transform(features)
                features = sm.add_constant(features)
                model = sm.OLS(labels, features).fit()
                st.text(model.summary())

def load_sidebar():
    selectbox = st.sidebar.selectbox(
        "Data Analysis Options:",
        ("Linear Regression", "ANOVA")
    )
    return selectbox


def get_data():
    df = pd.read_csv("treatments.csv")
    return df

def get_ssb_and_msb(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    mean_of_group = df[value_col].sum() / N
    mean_of_each_group = df.groupby(group_col).agg({value_col:"mean"})[value_col]
    SSB = n * sum([(m-mean_of_group)**2 for m in mean_of_each_group])
    MSB = SSB / df_between
    return SSB, MSB

def get_ssw_and_msw(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    SS = df[value_col] - df.groupby(group_col, axis=0).transform('mean')[value_col]
    SSW = sum([v**2 for v in SS.values])
    MSW = SSW / df_within
    return SSW, MSW

def get_f(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1
    SSB, MSB = get_ssb_and_msb(df, group_col, value_col)
    SSW, MSW = get_ssw_and_msw(df, group_col, value_col)
    F = MSB / MSW
    P = stats.f.sf(F, df_between, df_within)
    return F, P

def get_anova_table(df, group_col, value_col):
    k = len(pd.unique(df[group_col]))
    N = len(df.values)
    n = df.groupby(group_col).size()[0]
    df_between = k - 1
    df_within = N - k
    df_total = N - 1

    SSB, MSB = get_ssb_and_msb(df, group_col, value_col)
    SSW, MSW = get_ssw_and_msw(df, group_col, value_col)
    F, P = get_f(df, group_col, value_col)

    if P <= 0.001:
        significance_stars = "(***)"
    elif P > 0.001 and P <= 0.01:
        significance_stars = "(**)"
    elif P > 0.01 and P <= 0.05:
        significance_stars = "(*)"
    elif P > 0.05 and P <= 0.1:
        significance_stars = "(+)"
    elif P > 0.1 and P <= 1.0:
        significance_stars = ""

    table = pd.DataFrame({
        "Source of Variation": ["Between Treatments", "Within Treatments", "Total"], 
        "Sum of Squares (SS)": [SSB, SSW, SSB+SSW], 
        "Degrees of Freedom (df)": [df_between, df_within, df_total], 
        "Mean Squares (MS)": [str(MSB), str(MSW), "-"], 
        "F Score": [str(F), "-", "-"], 
        "PR(>F)": [f"{P}{significance_stars}", "-", "-"]
    })
    return table


if __name__ == "__main__":
    main()