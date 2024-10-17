import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 데이터 불러오기
@st.cache_data
def load_data():
    train = pd.read_csv('C:/imbank_streamlit_demo2/dataset/train.csv')
    return train


def main():
    st.title("test 1")
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    st.pyplot(fig)

    st.divider()

    # kaggle 데이터로 연습
    train = load_data() # load_data() 함수 호출로 train 데이터 가져오기

    st.title("Explatory Data Analysis")
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
    ax = sns.countplot(data=train, x='person_home_ownership')
    ax.set_title("Distribution of Loan Applicants by Home Ownership")
    st.pyplot(fig)

    # .head() 가져오기
    st.write(train.head()) # or st.dataframe(train.head())

    # st.column_config.Column
    st.data_editor(
        train,
        column_config={
            width=="large",


        },
        hide_index=True,
        num_rows="dynamic"
    )    


    st.divider()
    # st.dataframe 연습
    st.title("streamlit functions")
    st.header("st.dataframe")
    st.subheader("just copying")
    df = pd.DataFrame(np.random.randn(50,20), columns=("col %d" % i for i in range(20)) )
    st.dataframe(df)

    st.caption("st.caption is methods to write letters in small caption")

    


    








   





if __name__ == "__main__":
    main()