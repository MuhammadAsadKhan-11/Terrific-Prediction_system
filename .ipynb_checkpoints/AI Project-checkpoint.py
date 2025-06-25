import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset
df = pd.read_csv("Traffic.csv")

# 1. Convert 'Time' to minutes since midnight
def time_to_minutes(time_str):
    # Convert time from string format to datetime object and then to minutes
    time_obj = pd.to_datetime(time_str, format='%I:%M:%S %p')
    return time_obj.hour * 60 + time_obj.minute

df['Time'] = df['Time'].apply(time_to_minutes)

# 2. Convert 'Day of the week' and 'Traffic Situation' into numerical values using Label Encoding
label_encoder = LabelEncoder()
df['Day of the week'] = label_encoder.fit_transform(df['Day of the week'])
df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])

# 3. Prepare the features (X) and target (y)
X = df[['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']]
y = df['Traffic Situation']  # We're predicting the 'Traffic Situation'

# 4. Handle 'Date' column by converting it to the number of days since the start (if needed)
df['Date'] = pd.to_datetime(df['Date'])
df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
X.loc[:, 'Date'] = df['DaysSinceStart']

# 5. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the model using RandomForestClassifier
traffic_model = RandomForestClassifier(n_estimators=100, random_state=42)
traffic_model.fit(X_train, y_train)

# 7. Make predictions on the test
