import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
from fuzzywuzzy import process

# Load the data
@st.cache_data
def load_data():
    df = pd.read_excel('output_file.xlsx')
    # Calculate Success Rate
    df['Success Rate'] = (df['Certified'] / df['Registered']) * 100
    # Calculate Total Certificates column
    df['Total Certificates'] = df['Gold'] + df['Silver'] + df['Elite']
    return df

df = load_data()

st.title("NPTEL Course Dashboard")

# Sidebar for course selection
st.sidebar.header("Course Selection")
search_term = st.sidebar.text_input("Search for a course")
if search_term:
    matches = process.extract(search_term, df['Course Name'].unique(), limit=10)
    suggested_courses = [match[0] for match in matches]
    selected_course = st.sidebar.selectbox("Select a course", suggested_courses, format_func=lambda x: x)
else:
    selected_course = st.sidebar.selectbox("Select a course", df['Course Name'].unique(), format_func=lambda x: x)

# Filter data for the selected course
course_df = df[df['Course Name'] == selected_course]


print(selected_course)
# Display basic course info
# Instead of st.header(selected_course)
st.markdown("<h2 style='font-size: 24px;'>{}</h2>".format(selected_course), unsafe_allow_html=True)
st.write(f"NPTEL URL: {course_df['NPTEL URL'].iloc[0]}")

# Top courses based on user input
st.sidebar.header("Top Courses")
top_n = st.sidebar.slider("Select number of top courses", 5, 20, 10)
top_metric = st.sidebar.selectbox("Select metric for top courses", ['Success Rate', 'Gold %', 'Silver %', 'Elite %', 'Enrollment'])

if top_metric == 'Success Rate':
    top_courses = df.groupby('Course Name')['Success Rate'].mean().sort_values(ascending=False).head(top_n)
elif top_metric == 'Gold %':
    top_courses = (df.groupby('Course Name')['Gold'].sum() / df.groupby('Course Name')['Total Certificates'].sum() * 100).sort_values(ascending=False).head(top_n)
elif top_metric == 'Silver %':
    top_courses = (df.groupby('Course Name')['Silver'].sum() / df.groupby('Course Name')['Total Certificates'].sum() * 100).sort_values(ascending=False).head(top_n)
elif top_metric == 'Elite %':
    top_courses = (df.groupby('Course Name')['Elite'].sum() / df.groupby('Course Name')['Total Certificates'].sum() * 100).sort_values(ascending=False).head(top_n)
else:
    top_courses = df.groupby('Course Name')['Enrolled'].mean().sort_values(ascending=False).head(top_n)

st.sidebar.write(f"Top {top_n} courses based on {top_metric}:")
st.sidebar.dataframe(top_courses)

# Enrollment Trends
st.subheader("Enrollment Trends")
fig_enrollment = px.line(course_df, x='Timeline', y='Enrolled', title='Enrollment Over Time')
st.plotly_chart(fig_enrollment)

# Certification Distribution
st.subheader("Certification Distribution")
cert_df = course_df.melt(id_vars=['Timeline'], value_vars=['Gold', 'Silver', 'Elite'], var_name='Certificate Type', value_name='Count')
fig_cert = px.bar(cert_df, x='Timeline', y='Count', color='Certificate Type', title='Certification Distribution Over Time')
st.plotly_chart(fig_cert)

# Certificate Percentage Graph
st.subheader("Certificate Percentage")
course_df.loc[:, 'Gold %'] = course_df['Gold'] / course_df['Total Certificates'] * 100
course_df.loc[:, 'Silver %'] = course_df['Silver'] / course_df['Total Certificates'] * 100
course_df.loc[:, 'Elite %'] = course_df['Elite'] / course_df['Total Certificates'] * 100

fig_cert_percent = go.Figure()
fig_cert_percent.add_trace(go.Scatter(x=course_df['Timeline'], y=course_df['Gold %'], mode='lines', name='Gold %'))
fig_cert_percent.add_trace(go.Scatter(x=course_df['Timeline'], y=course_df['Silver %'], mode='lines', name='Silver %'))
fig_cert_percent.add_trace(go.Scatter(x=course_df['Timeline'], y=course_df['Elite %'], mode='lines', name='Elite %'))
fig_cert_percent.update_layout(title='Certificate Percentage Over Time', xaxis_title='Timeline', yaxis_title='Percentage')
st.plotly_chart(fig_cert_percent)

# Success Rate
st.subheader("Success Rate")
fig_success = px.line(course_df, x='Timeline', y='Success Rate', title='Success Rate Over Time')
st.plotly_chart(fig_success)

# Additional Line Graphs
st.subheader("Additional Metrics")
metrics = ['Registered', 'Certified', 'Success', 'Participation', 'Toppers']
selected_metric = st.selectbox("Select a metric", metrics)
fig_metric = px.line(course_df, x='Timeline', y=selected_metric, title=f'{selected_metric} Over Time')
st.plotly_chart(fig_metric)

# Correlation Heatmap
st.subheader("Correlation Heatmap")
numeric_cols = ['Enrolled', 'Registered', 'Certified', 'Gold', 'Silver', 'Elite', 'Success', 'Participation', 'Toppers', 'Success Rate']
corr_matrix = course_df[numeric_cols].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
st.plotly_chart(fig_corr)

# ML Model for Prediction
st.header("Enrollment and Certificate Prediction Model")

def extract_year(timeline):
    parts = timeline.split()
    return int(parts[-1])  # Always take the last part as the year

def extract_month(timeline):
    month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    parts = timeline.split()
    return month_dict.get(parts[0], 1)  # Default to 1 if month not found

# Prepare data for ML model
course_df.loc[:, 'Year'] = course_df['Timeline'].apply(extract_year)
course_df.loc[:, 'Month'] = course_df['Timeline'].apply(extract_month)

X = course_df[['Year', 'Month']]
y_enrolled = course_df['Enrolled']
y_gold = course_df['Gold']
y_silver = course_df['Silver']
y_elite = course_df['Elite']

if len(X) > 1:  # Check if there's enough data
    try:
        # Split data and train models
        X_train, X_test, y_enrolled_train, y_enrolled_test = train_test_split(X, y_enrolled, test_size=0.2, random_state=42)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        selected_model = st.selectbox("Select a model", list(models.keys()))
        
        model_enrolled = models[selected_model]
        model_enrolled.fit(X_train, y_enrolled_train)
        
        model_gold = models[selected_model]
        model_gold.fit(X, y_gold)
        
        model_silver = models[selected_model]
        model_silver.fit(X, y_silver)
        
        model_elite = models[selected_model]
        model_elite.fit(X, y_elite)

        # Calculate RMSE for enrollment
        y_enrolled_pred = model_enrolled.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_enrolled_test, y_enrolled_pred))
        st.write(f"Enrollment Model RMSE ({selected_model}): {rmse:.2f}")

        # User input for prediction
        st.subheader("Predict Enrollment and Certificates")
        pred_year = st.number_input("Enter Year", min_value=2020, max_value=2030, value=2024)
        pred_month = st.selectbox("Enter Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        pred_month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}[pred_month]

        if st.button("Predict"):
            enrolled_prediction = model_enrolled.predict([[pred_year, pred_month_num]])
            gold_prediction = model_gold.predict([[pred_year, pred_month_num]])
            silver_prediction = model_silver.predict([[pred_year, pred_month_num]])
            elite_prediction = model_elite.predict([[pred_year, pred_month_num]])
            
            st.write(f"Predicted Enrollment: {enrolled_prediction[0]:.0f}")
            st.write(f"Predicted Gold Certificates: {gold_prediction[0]:.0f}")
            st.write(f"Predicted Silver Certificates: {silver_prediction[0]:.0f}")
            st.write(f"Predicted Elite Certificates: {elite_prediction[0]:.0f}")

    except Exception as e:
        st.error(f"An error occurred while training the model: {str(e)}")
        st.write("Please try selecting a different course or check your data.")
else:
    st.warning("Not enough data to train a model. Please select a course with more data points.")

# Certificate Predictor
st.header("Certificate Predictor")
user_score = st.slider("Enter your score", 0, 100, 50)

avg_gold = course_df['Gold'].mean()
avg_silver = course_df['Silver'].mean()
avg_elite = course_df['Elite'].mean()

if user_score >= 90:
    st.success(f"With a score of {user_score}, you're likely to get a Gold certificate!")
elif user_score >= 75:
    st.info(f"With a score of {user_score}, you're likely to get a Silver certificate!")
elif user_score >= 60:
    st.warning(f"With a score of {user_score}, you're likely to get an Elite certificate!")
else:
    st.error(f"With a score of {user_score}, you may not qualify for a certificate. Try to score at least 60 for an Elite certificate.")

month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
# Display summary statistics
st.header("Summary Statistics")
summary_stats = course_df.describe()
summary_stats['Month'] = summary_stats['Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                                                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})
st.dataframe(summary_stats)

# Timeline-wise data
# Timeline-wise data
st.header("Timeline-wise Data")
course_df['Month'] = course_df['Month'].map(month_map)
st.dataframe(course_df)
