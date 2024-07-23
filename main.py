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

# Set page config
st.set_page_config(page_title="NPTEL Course Dashboard", page_icon="📊", layout="wide")

# Custom CSS to enhance UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .sidebar .sidebar-content {
        background: #262730
    }
    .Widget>label {
        color: #262730;
        font-family: sans-serif;
    }
    .stTextInput>div>div>input {
        color: #262730;
    }
    .stSelectbox>div>div>select {
        color: #262730;
    }
</style>
""", unsafe_allow_html=True)

# Load the data
@st.cache_data
def load_data():
    df = pd.read_excel('output_file.xlsx')
    df['Success Rate'] = (df['Certified'] / df['Registered']) * 100
    df['Total Certificates'] = df['Gold'] + df['Silver'] + df['Elite']
    return df

df = load_data()

# Sidebar
st.sidebar.image("https://github.com/anupammaurya6767/Kaunsa_mooc_le/blob/main/image.png", width=200)
st.sidebar.title("NPTEL Course Dashboard")

# Course selection
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

# Main content
st.title("📊 NPTEL Course Analysis")

# Course Info
st.header("📘 Course Information")
col1, col2 = st.columns(2)
with col1:
    st.markdown("<h2 style='font-size: 20px;'>{}</h2>".format(selected_course), unsafe_allow_html=True)
    st.markdown(f"**NPTEL URL:** [Link]({course_df['NPTEL URL'].iloc[0]})")
with col2:
    st.markdown(f"**Latest Enrollment:** {course_df['Enrolled'].iloc[-1]:,}")
    st.markdown(f"**Latest Success Rate:** {course_df['Success Rate'].iloc[-1]:.2f}%")

# Enrollment Trends
st.header("📈 Enrollment Trends")
fig_enrollment = px.line(course_df, x='Timeline', y='Enrolled', title='Enrollment Over Time')
fig_enrollment.update_layout(template="plotly_white")
st.plotly_chart(fig_enrollment, use_container_width=True)

# Certification Distribution
st.header("🏅 Certification Distribution")
cert_df = course_df.melt(id_vars=['Timeline'], value_vars=['Gold', 'Silver', 'Elite'], var_name='Certificate Type', value_name='Count')
fig_cert = px.bar(cert_df, x='Timeline', y='Count', color='Certificate Type', title='Certification Distribution Over Time')
fig_cert.update_layout(template="plotly_white")
st.plotly_chart(fig_cert, use_container_width=True)

# Success Rate
st.header("🎯 Success Rate")
fig_success = px.line(course_df, x='Timeline', y='Success Rate', title='Success Rate Over Time')
fig_success.update_layout(template="plotly_white")
st.plotly_chart(fig_success, use_container_width=True)

# Additional Metrics
st.header("📊 Additional Metrics")
metrics = ['Registered', 'Certified', 'Success', 'Participation', 'Toppers']
selected_metric = st.selectbox("Select a metric", metrics)
fig_metric = px.line(course_df, x='Timeline', y=selected_metric, title=f'{selected_metric} Over Time')
fig_metric.update_layout(template="plotly_white")
st.plotly_chart(fig_metric, use_container_width=True)

# Correlation Heatmap
st.header("🔗 Correlation Heatmap")
numeric_cols = ['Enrolled', 'Registered', 'Certified', 'Gold', 'Silver', 'Elite', 'Success', 'Participation', 'Toppers', 'Success Rate']
corr_matrix = course_df[numeric_cols].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto")
fig_corr.update_layout(template="plotly_white")
st.plotly_chart(fig_corr, use_container_width=True)

# ML Model for Prediction
st.header("🔮 Enrollment and Certificate Prediction")

def extract_year(timeline):
    return int(timeline.split()[-1])

def extract_month(timeline):
    month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                  'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    return month_dict.get(timeline.split()[0], 1)

course_df['Year'] = course_df['Timeline'].apply(extract_year)
course_df['Month'] = course_df['Timeline'].apply(extract_month)

X = course_df[['Year', 'Month']]
y_enrolled = course_df['Enrolled']
y_gold = course_df['Gold']
y_silver = course_df['Silver']
y_elite = course_df['Elite']

if len(X) > 1:
    try:
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

        y_enrolled_pred = model_enrolled.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_enrolled_test, y_enrolled_pred))
        st.write(f"Enrollment Model RMSE ({selected_model}): {rmse:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            pred_year = st.number_input("Enter Year", min_value=2020, max_value=2030, value=2024)
        with col2:
            pred_month = st.selectbox("Enter Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        pred_month_num = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 
                          'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}[pred_month]

        if st.button("Predict"):
            enrolled_prediction = model_enrolled.predict([[pred_year, pred_month_num]])
            gold_prediction = model_gold.predict([[pred_year, pred_month_num]])
            silver_prediction = model_silver.predict([[pred_year, pred_month_num]])
            elite_prediction = model_elite.predict([[pred_year, pred_month_num]])
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Enrollment", f"{enrolled_prediction[0]:,.0f}")
            col2.metric("Predicted Gold", f"{gold_prediction[0]:,.0f}")
            col3.metric("Predicted Silver", f"{silver_prediction[0]:,.0f}")
            col4.metric("Predicted Elite", f"{elite_prediction[0]:,.0f}")

    except Exception as e:
        st.error(f"An error occurred while training the model: {str(e)}")
        st.write("Please try selecting a different course or check your data.")
else:
    st.warning("Not enough data to train a model. Please select a course with more data points.")

# Certificate Predictor
st.header("🎓 Certificate Predictor")
user_score = st.slider("Enter your score", 0, 100, 50)

avg_gold = course_df['Gold'].mean()
avg_silver = course_df['Silver'].mean()
avg_elite = course_df['Elite'].mean()

if user_score >= 90:
    st.success(f"With a score of {user_score}, you're likely to get a Gold certificate! 🥇")
elif user_score >= 75:
    st.info(f"With a score of {user_score}, you're likely to get a Silver certificate! 🥈")
elif user_score >= 60:
    st.warning(f"With a score of {user_score}, you're likely to get an Elite certificate! 🏅")
else:
    st.error(f"With a score of {user_score}, you may not qualify for a certificate. Try to score at least 60 for an Elite certificate. 📚")

# Summary Statistics
st.header("📊 Summary Statistics")
summary_stats = course_df.describe()
summary_stats['Month'] = summary_stats['Month'].map({1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                                                     7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'})

# Separate numeric and non-numeric columns
numeric_cols = summary_stats.select_dtypes(include=[np.number]).columns
non_numeric_cols = summary_stats.select_dtypes(exclude=[np.number]).columns

# Apply highlighting only to numeric columns
styled_summary = summary_stats.style.highlight_max(subset=numeric_cols, axis=0)

# Display the styled dataframe
st.dataframe(styled_summary)

# Timeline-wise data
st.header("📅 Timeline-wise Data")
# Apply highlighting only to numeric columns for timeline data as well
numeric_cols_timeline = course_df.select_dtypes(include=[np.number]).columns
styled_timeline = course_df.style.highlight_max(subset=numeric_cols_timeline, axis=0)
st.dataframe(styled_timeline)

# New Feature: Course Comparison
st.header("🔍 Course Comparison: Top Courses")
comparison_metric = st.selectbox("Select a metric for comparison", ['Enrolled', 'Success Rate', 'Gold', 'Silver', 'Elite'])
top_n = st.slider("Select number of top courses to compare", 5, 20, 10)

top_courses = df.groupby('Course Name')[comparison_metric].mean().sort_values(ascending=False).head(top_n)
fig_comparison = px.bar(top_courses, x=top_courses.index, y=comparison_metric, 
                        title=f'Top {top_n} Courses by {comparison_metric}')
fig_comparison.update_layout(template="plotly_white", xaxis_title="Course Name", yaxis_title=comparison_metric)
st.plotly_chart(fig_comparison, use_container_width=True)

# New Feature: Trend Analysis
st.header("📈 Trend Analysis")
trend_metric = st.selectbox("Select a metric for trend analysis", ['Enrolled', 'Success Rate', 'Gold', 'Silver', 'Elite'])
trend_courses = df.groupby('Course Name')[trend_metric].mean().sort_values(ascending=False).head(5).index.tolist()

fig_trend = go.Figure()
for course in trend_courses:
    course_data = df[df['Course Name'] == course]
    fig_trend.add_trace(go.Scatter(x=course_data['Timeline'], y=course_data[trend_metric], mode='lines', name=course))

fig_trend.update_layout(title=f'{trend_metric} Trend for Top 5 Courses', template="plotly_white",
                        xaxis_title="Timeline", yaxis_title=trend_metric)
st.plotly_chart(fig_trend, use_container_width=True)

# Footer
st.markdown("---")
