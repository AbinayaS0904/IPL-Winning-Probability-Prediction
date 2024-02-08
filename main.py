import base64
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("background.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/png;base64,{img}");
width: 100%;
height:100%
background-repeat: no-repeat;
background-attachment: fixed;
background-size: cover;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

teams = ['--- select ---',
         'Sunrisers Hyderabad',
         'Mumbai Indians',
         'Kolkata Knight Riders',
         'Royal Challengers Bangalore',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Bangalore', 'Hyderabad', 'Kolkata', 'Mumbai', 'Visakhapatnam',
          'Indore', 'Durban', 'Chandigarh', 'Delhi', 'Dharamsala',
          'Ahmedabad', 'Chennai', 'Ranchi', 'Nagpur', 'Mohali', 'Pune',
          'Bengaluru', 'Jaipur', 'Port Elizabeth', 'Centurion', 'Raipur',
          'Sharjah', 'Cuttack', 'Johannesburg', 'Cape Town', 'East London',
          'Abu Dhabi', 'Kimberley', 'Bloemfontein']

# Load datasets
first_dataset = pd.read_csv('dataset.csv')
second_dataset = pd.read_csv('IPL Matches 2008-2020.csv')

# Preprocess data and train the model
# ... (perform necessary data preprocessing on first_dataset and second_dataset)

# Replace 'feature_column_names' with the actual feature column names in your dataset
feature_column_names = ['feature1', 'feature2', 'feature3', ...]
# Replace 'target_column_name' with the actual target column name in your dataset
target_column_name = 'target'

# Assuming your features and target are stored in 'first_dataset'
x = first_dataset[feature_column_names]
y = first_dataset[target_column_name]

# Assume x_train, y_train are your features and target from the datasets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Train the model (replace RandomForestClassifier with your actual model)
model = RandomForestClassifier()
model.fit(x_train, y_train)

st.markdown(page_bg_img, unsafe_allow_html=True)
st.markdown("""
    # **IPL VICTORY PREDICTOR**            
""")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select Batting Team', teams)

with col2:
    if batting_team == '--- select ---':
        bowling_team = st.selectbox('Select Bowling Team', teams)
    else:
        filtered_teams = [team for team in teams if team != batting_team]
        bowling_team = st.selectbox('Select Bowling Team', filtered_teams)

selected_city = st.selectbox('Select Venue', cities)

target = st.number_input('Target')

col1, col2, col3 = st.columns(3)

with col1:
    score = st.number_input('Score')

with col2:
    overs = st.number_input("Over Completed")

with col3:
    wickets = st.number_input("Wickets Down")

if st.button('Predict Winning Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = runs_left / (balls_left / 6)

    input_data = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                               'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                               'wickets_remaining': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = model.predict_proba(input_data)

    loss = result[0][0]
    win = result[0][1]
    st.header(batting_team + " = " + str(round(win * 100)) + "%")
    st.header(bowling_team + " = " + str(round(loss * 100)) + "%")
