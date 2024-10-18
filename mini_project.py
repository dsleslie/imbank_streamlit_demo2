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
     train = load_data() # load_data() 함수 호출로 train 데이터 가져오기
    
st.title(":blue-background[Explatory Data Analysis]")
st.header(":green-background[**Features**]")
st.subheader(":orange-background[Understanding Features]")
explain_features = '''- person_age:   Age of the borrower  
- person_income:   Income of the borrower  
- person_home_ownership:   Home ownership status  
- person_emp_length:   Length of employment  
- loan_intent:   Purpose of the loan  
- loan_grade:   Loan grade  
- loan_amnt:   Loan amount  
- loan_int_rate:   Loan interest rate  
- loan_percent_income:   Loan-to-income ratio  
- cb_person_default_on_file:   Borrower's past default history  
- cb_person_cred_hist_length:   Length of credit history  
- loan_status:   Loan status (target variable)'''
st.markdown(explain_features)

st.subheader(":orange-background[Type of the Features]")
train = load_data()
st.write(str(train.dtypes))

st.header(":green-background[***Descriptive Statistics***]")
st.write(train.describe())
st.write(train.value_counts())
st.divider()

st.header(":green-background[Analysis on Features]")
st.subheader(":orange-background[Loan approval based on ***Loan Intent***]")
def stacked_bar_plot(df, feature, target='loan_status'):
    crosstab = pd.crosstab(df[feature], df[target], normalize='index')

    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
    ax=crosstab.plot(kind='bar',stacked=True, cmap='coolwarm', ax=ax)
    ax.set_title(f'Stacked Bar Plot of {feature} vs {target}')
    ax.set_ylabel('Proportion')
    #plt.show()
st.pyplot(stacked_bar_plot(train, 'loan_intent'))


st.subheader(":orange-background[Loan approval based on ***homeownership type***]")
st.pyplot(stacked_bar_plot(train, 'person_home_ownership'))


st.subheader(":orange-background[Number of loan applications by ***homeownership type***]")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
ax = sns.countplot(data=train, x='person_home_ownership')
ax.set_title("Distribution of Loan Applicants by Home Ownership")
st.pyplot(fig)

st.subheader(":orange-background[Number of loan applications by ***Loan Intent***]")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
ax = sns.countplot(data=train, x='loan_intent')
ax.set_title("Loan Intent Distribution")
ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
st.pyplot(fig)
st.write("**The most common loan applications are in the following order: EDUCATION > MEDICAL > PERSONAL > VENTURE> DEBTCONSOLIDATION > HOMEIMPROVEMENT.**")

st.subheader(":orange-background[Number of applicants by loan grade and loan status]")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
ax = sns.countplot(data=train, x='loan_grade',hue = 'loan_status')
ax.set_title("Loan Default Rate by Loan Grade")
st.pyplot(fig)
st.write("**The higher the loan grade, the more people (0) are approved.**")

st.subheader(":orange-background[Whether or not to approve a loan for someone with previous bankruptcy experience]")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
ax = sns.countplot(data=train, x='cb_person_default_on_file', hue='loan_status')
ax.set_title("Loan Default by Prior Default Status")
st.pyplot(fig)
st.write("**People who have previously been bankrupt (Y) have significantly fewer loans approved than those who have paid in full (N)**")


st.subheader(":orange-background[Boxplot showing distribution of income and loan application amount]")
def plot_boxplots(df, columns):  # 여러 열에 대한 박스플롯 한 번에 생성
    fig, axes = plt.subplots(1, len(columns), figsize=(12, 6))  # fig와 axes를 생성
    
    # 하나의 열만 있을 경우 axes를 리스트로 만들기 위해 배열 처리
    if len(columns) == 1:
        axes = [axes]
    
    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], color='lightblue', ax=axes[i])  # 각 axes에 박스플롯 그리기
        axes[i].set_title(f'Box Plot of {col}')  # 각 서브플롯 제목 설정

    plt.tight_layout()  # 그래프가 겹치지 않도록 레이아웃 조정
    
st.pyplot(plot_boxplots(train, ['person_income', 'loan_amnt']))
st.markdown('''
- Missing values, outliers, etc. can be seen, scale differences and extreme values ​​can be seen visually.
- There are many people whose loan application amount (loan_amnt) is higher than the 4th quartile.''')


st.subheader(":orange-background[Distribution of loan amounts in the form of a cumulative distribution function for loan status]")
fig, ax = plt.subplots(figsize=(10,6))
sns.kdeplot(train[train['loan_status']==1]['loan_amnt'], label = 'Default' , fill = True)
sns.kdeplot(train[train['loan_status']==0]['loan_amnt'], label = 'Non-Default', fill=True)
ax.set_title('CDF of Loan Amount by Loan Status')
ax.set_xlabel('Loan Amount')
ax.set_ylabel('Density')
ax.legend()
st.pyplot(fig)
st.write("**People who are approved for a loan (loan_status = 0) have a smaller loan amount than people who are rejected for a loan (loan_stats = 1)**")


st.subheader(":orange-background[Loan application amount and approval status according to loan grade]")
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(10, 6))
grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
train['loan_grade'] = pd.Categorical(train['loan_grade'], categories=grade_order, ordered=True)
ax = sns.barplot(x='loan_grade', y='loan_amnt', hue='loan_status', order=grade_order, data=train, estimator=np.mean)
ax.set_title("Average Loan Amount by Loan Grade and Status")
ax.set_ylabel('Average Loan Amount')
ax.set_xlabel('Loan Grade')
ax.legend(title = 'Loan Status')
st.pyplot(fig)
st.markdown('''
- The riskier the loan grade, the higher the loan amount tends to be.
- Within each loan grade, up to grade C, the loan application amount for those who are approved for a loan is smaller.
- However, for grades above D, the loan application amount for loan approval is larger.
- Why? ''')

st.divider()

st.header(":green-background[Correlation Analysis]")
st.write(":grey-background[Pick numeric variables]")
numeric_df_train=train.select_dtypes(include=['int64', 'float64'])
st.write(numeric_df_train.head())

st.subheader(":orange-background[correlation cross tab]")
result_corr=numeric_df_train.corr()
st.write(result_corr)

st.subheader(":orange-background[Heatmap]")
# 히트맵 사용자 정의 함수로 해보기
def plot_correlation_heatmap(df, figsize=(15, 15), cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5):
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(data=df.corr(), annot=annot, fmt=fmt, linewidths=linewidths, cmap=cmap)
    st.pyplot(fig)

# 함수 사용
plot_correlation_heatmap(numeric_df_train)
st.markdown("""
- **person_age** and **cb_person_cred_hist_length**: These two variables show a strong positive correlation (0.87)
    - Older individuals tend to have longer credit histories.
    - How should we handle multicollinearity here?
- **loan_percent_income** and **loan_amnt**: Correlation of 0.62
    - As loan amount increases, the percentage of income used for loans also rises.
- **loan_status**, **loan_int_rate**, and **loan_percent_income**: Positive correlation
    - Higher interest rates and income-to-loan ratios might affect loan approval.
- **person_income** and **loan_percent_income**: Correlation of -0.26, negative
    - Higher income individuals tend to allocate a smaller percentage of income to loans.
""")



st.subheader(":orange-background[Scatter plot]")
def plot_custom_pairplot(df, cols_to_highlight=None, title="To check Multicollinearity build Scatter plot between features", figsize=(15, 15), title_y=1.02):
    g = sns.pairplot(df)
    g.fig.suptitle(title, y=title_y)
    
    # 다중공선성이 의심되는 변수 쌍에 배경색 칠하기 - 찾아서 칠하기 - cols_to_highlight에다가 입력하면 됨
    if cols_to_highlight:
        for i, row_var in enumerate(df.columns):
            for j, col_var in enumerate(df.columns):
                if (row_var, col_var) in cols_to_highlight or (col_var, row_var) in cols_to_highlight:
                    g.axes[i, j].set_facecolor('lightyellow')  # 배경색 설정 (예: 'lightyellow')
    st.pyplot(fig)

# 예시 사용: 다중공선성이 의심되는 변수 쌍을 지정하여 그래프 그리기
cols_to_highlight = [('person_age', 'cb_person_cred_hist_length'), ('loan_amnt', 'loan_percent_income')]
plot_custom_pairplot(numeric_df_train, cols_to_highlight=cols_to_highlight)



st.subheader(":orange-background[VIF]")
from statsmodels.stats.outliers_influence import variance_inflation_factor
from matplotlib import rc

def calculate_vif(df, title="VIF"):
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i)
                    for i in range(df.shape[1])]
    vif["Feature"] = df.columns
    vif = vif.sort_values(by="VIF Factor", ascending = False)
    vif = vif.reset_index().drop(columns='index')
    highlight_vif_features = vif[vif["VIF Factor"]>10]["Feature"]
    st.write(vif)
    print(f'"Variables suspected of multicollinearity are:"{highlight_vif_features}')

calculate_vif(numeric_df_train)
st.markdown("""
### Correlation Analysis
- **person_age** and **cb_person_cred_hist_length**: High correlation observed.
- **loan_amnt** and **loan_percent_income**: Positive correlation noted.

### VIF Results
- Potential multicollinearity detected between **person_age** and **loan_int_rate**.
""")



'''
st.subheader(":orange-background[d]")


st.pyplot(fig)
st.write("**d**")




st.subheader(":orange-background[d]")


st.pyplot(fig)
st.write("**d**")



st.subheader(":orange-background[d]")


st.pyplot(fig)
st.write("**d**")








st.header(":green-background[VIF]")

    




'''



   





if __name__ == "__main__":
    main()