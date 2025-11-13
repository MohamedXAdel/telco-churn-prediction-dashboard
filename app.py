import streamlit as st
import pandas as pd
import joblib
import pickle


# Load the data
df = pd.read_csv('telecom_customer_churn.csv')

# Clean the data (same as training)
df['Churn'] = (df['Customer Status'] == 'Churned').astype(int)
df['Offer'] = df['Offer'].fillna('None')
df.loc[df['Phone Service'] == 'No', 'Avg Monthly Long Distance Charges'] = 0
df.loc[df['Phone Service'] == 'No', 'Multiple Lines'] = 'No'

internet_cols = ['Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                 'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data']
for col in internet_cols:
    df.loc[df['Internet Service'] == 'No', col] = 'No'

df.loc[df['Internet Service'] == 'No', 'Internet Type'] = 'None'
df['Avg Monthly GB Download'] = df['Avg Monthly GB Download'].fillna(0)


# Title
st.title("Customer Churn Prediction Dashboard")

# Sidebar filters
st.sidebar.header('Filters')
cities = st.sidebar.multiselect('Select City', options=sorted(df['City'].unique()), default=df['City'].unique())
contract = st.sidebar.radio('Select Contract', options=df['Contract'].unique())
married = st.sidebar.radio('Select Married Status', options=df['Married'].unique())
age_min, age_max = st.sidebar.slider('Select Age Range', min_value=int(df['Age'].min()), max_value=int(df['Age'].max()), value=(int(df['Age'].min()), int(df['Age'].max())))

# Filter the data
filtered_df = df[
    (df['City'].isin(cities)) &
    (df['Contract'] == contract) &
    (df['Married'] == married) &
    (df['Age'].between(age_min, age_max))
]


# Display charts
st.header('Churn Analysis')

# Churn rate metric
churn_rate = (filtered_df['Customer Status'] == 'Churned').mean() * 100
st.metric('Churn Rate', f'{churn_rate:.2f}%')

# Bar chart for customer status
st.subheader('Customer Status Distribution')
st.bar_chart(filtered_df['Customer Status'].value_counts())

# Bar chart for churn by offer
st.subheader('Churn by Offer')
churn_by_offer = filtered_df.groupby('Offer')['Churn'].mean().sort_values(ascending=False)
st.bar_chart(churn_by_offer)

# Histogram for age distribution
st.subheader('Age Distribution')
st.bar_chart(filtered_df['Age'].value_counts(bins=10, sort=False))

# Prediction section
st.header('Churn Probability Prediction')


# Load model + training column names
model = joblib.load("model.pkl")
training_columns = joblib.load("training_columns.pkl")

# User inputs for prediction
st.subheader('Select Inputs')

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider('Age', 19, 80, 37)
    num_dependents = st.slider('Number of Dependents', 0, 9, 0)
    num_referrals = st.slider('Number of Referrals', 0, 11, 0)
    tenure_months = st.slider('Tenure in Months', 1, 72, 9)

with col2:
    long_distance_charges = st.number_input('Avg Monthly Long Distance Charges', 0.0, 50.0, 0.0)
    gb_download = st.number_input('Avg Monthly GB Download', 0.0, 85.0, 0.0)
    monthly_charge = st.number_input('Monthly Charge', -10.0, 120.0, 65.0)
    gender = st.selectbox('Gender', ['Female', 'Male'])

with col3:
    married_input = st.selectbox('Married', ['No', 'Yes'])
    offer = st.selectbox('Offer', ['None', 'Offer A', 'Offer B', 'Offer C', 'Offer D', 'Offer E'])
    phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
    multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])

col4, col5, col6 = st.columns(3)

with col4:
    internet_type = st.selectbox('Internet Type', ['None', 'Cable', 'DSL', 'Fiber Optic'])
    online_security = st.selectbox('Online Security', ['No', 'Yes'])
    online_backup = st.selectbox('Online Backup', ['No', 'Yes'])
    device_protection = st.selectbox('Device Protection Plan', ['No', 'Yes'])

with col5:
    premium_support = st.selectbox('Premium Tech Support', ['No', 'Yes'])
    streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes'])
    streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes'])
    streaming_music = st.selectbox('Streaming Music', ['No', 'Yes'])

with col6:
    unlimited_data = st.selectbox('Unlimited Data', ['No', 'Yes'])
    contract_input = st.selectbox('Contract', ['Month-to-Month', 'One Year', 'Two Year'])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    payment_method = st.selectbox('Payment Method', ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])

# Prepare input data
input_dict = {
    'Age': age,
    'Number of Dependents': num_dependents,
    'Number of Referrals': num_referrals,
    'Tenure in Months': tenure_months,
    'Avg Monthly Long Distance Charges': long_distance_charges,
    'Avg Monthly GB Download': gb_download,
    'Monthly Charge': monthly_charge,
    'Gender': gender,
    'Married': married_input,
    'Offer': offer,
    'Phone Service': phone_service,
    'Multiple Lines': multiple_lines,
    'Internet Type': internet_type,
    'Online Security': online_security,
    'Online Backup': online_backup,
    'Device Protection Plan': device_protection,
    'Premium Tech Support': premium_support,
    'Streaming TV': streaming_tv,
    'Streaming Movies': streaming_movies,
    'Streaming Music': streaming_music,
    'Unlimited Data': unlimited_data,
    'Contract': contract_input,
    'Paperless Billing': paperless_billing,
    'Payment Method': payment_method
}

input_df = pd.DataFrame([input_dict])

# One-hot encode the input (same as training)
features_cat = ['Gender', 'Married', 'Offer', 'Phone Service', 'Multiple Lines', 'Internet Type', 
                'Online Security', 'Online Backup', 'Device Protection Plan', 'Premium Tech Support', 
                'Streaming TV', 'Streaming Movies', 'Streaming Music', 'Unlimited Data', 
                'Contract', 'Paperless Billing', 'Payment Method']
input_df = pd.get_dummies(input_df, columns=features_cat, drop_first=True)

# Align columns with training data (add missing columns with 0)
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns
input_df = input_df[training_columns]

# Convert types
input_df = input_df.astype(float)

# Reorder columns to match model
input_df = input_df[training_columns]

# Convert to float
input_df = input_df.astype(float)

# Predict
if st.button('Predict Churn Probability'):
    prob = model.predict_proba(input_df)[0][1]   
    st.write(f'Predicted Probability of Churn: **{prob:.2%}**')