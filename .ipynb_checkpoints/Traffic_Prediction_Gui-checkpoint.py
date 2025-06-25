import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import messagebox
import os
os.environ['TCL_LIBRARY'] = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python313\tcl\tcl8.6"
os.environ['TK_LIBRARY'] = r"C:\Users\ASUS\AppData\Local\Programs\Python\Python313\tcl\tk8.6"


# Load dataset
df = pd.read_csv("Traffic.csv")

# Convert 'Time' to minutes since midnight
def time_to_minutes(time_str):
    time_obj = pd.to_datetime(time_str, format='%I:%M:%S %p')
    return time_obj.hour * 60 + time_obj.minute

df['Time'] = df['Time'].apply(time_to_minutes)

# Convert 'Day of the week' and 'Traffic Situation' into numerical values
label_encoder = LabelEncoder()
df['Day of the week'] = label_encoder.fit_transform(df['Day of the week'])
df['Traffic Situation'] = label_encoder.fit_transform(df['Traffic Situation'])

# Prepare the features (X) and target (y)
X = df[['Time', 'Date', 'Day of the week', 'CarCount', 'BikeCount', 'BusCount', 'TruckCount']]
y = df['Traffic Situation']  # We're predicting the 'Traffic Situation'

# Handle 'Date' column
df['Date'] = pd.to_datetime(df['Date'])
df['DaysSinceStart'] = (df['Date'] - df['Date'].min()).dt.days
X.loc[:, 'Date'] = df['DaysSinceStart']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
traffic_model = RandomForestClassifier(n_estimators=100, random_state=42)
traffic_model.fit(X_train, y_train)


# Create Tkinter GUI
def predict_traffic():
    try:
        # Get inputs from user
        time = int(entry_time.get())
        day_of_week = int(entry_day_of_week.get())
        date = int(entry_date.get())
        car_count = int(entry_car_count.get())
        bike_count = int(entry_bike_count.get())
        bus_count = int(entry_bus_count.get())
        truck_count = int(entry_truck_count.get())

        # Prepare data for prediction
        input_data = [[time, date, day_of_week, car_count, bike_count, bus_count, truck_count]]

        # Predict traffic situation
        prediction = traffic_model.predict(input_data)

        # Decode the prediction label
        traffic_situation = label_encoder.inverse_transform(prediction)

        # Display prediction result
        messagebox.showinfo("Prediction Result", f"Predicted Traffic Situation: {traffic_situation[0]}")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")


# Create the main window
window = tk.Tk()
window.title("Traffic Prediction")

# Create a title header
header_label = tk.Label(window, text="Traffic Prediction System", font=("Helvetica", 16, "bold"))
header_label.grid(row=0, columnspan=2, pady=10)

# Create input fields and labels with descriptions
tk.Label(window, text="Time of Prediction (in hours and minutes):").grid(row=1, column=0, sticky="w", padx=10)
entry_time = tk.Entry(window)
entry_time.grid(row=1, column=1, padx=10)

tk.Label(window, text="Day of the Week (0-6, where 0=Monday, 6=Sunday):").grid(row=2, column=0, sticky="w", padx=10)
entry_day_of_week = tk.Entry(window)
entry_day_of_week.grid(row=2, column=1, padx=10)

tk.Label(window, text="Date (days since the start of the data):").grid(row=3, column=0, sticky="w", padx=10)
entry_date = tk.Entry(window)
entry_date.grid(row=3, column=1, padx=10)

tk.Label(window, text="Car Count:").grid(row=4, column=0, sticky="w", padx=10)
entry_car_count = tk.Entry(window)
entry_car_count.grid(row=4, column=1, padx=10)

tk.Label(window, text="Bike Count:").grid(row=5, column=0, sticky="w", padx=10)
entry_bike_count = tk.Entry(window)
entry_bike_count.grid(row=5, column=1, padx=10)

tk.Label(window, text="Bus Count:").grid(row=6, column=0, sticky="w", padx=10)
entry_bus_count = tk.Entry(window)
entry_bus_count.grid(row=6, column=1, padx=10)

tk.Label(window, text="Truck Count:").grid(row=7, column=0, sticky="w", padx=10)
entry_truck_count = tk.Entry(window)
entry_truck_count.grid(row=7, column=1, padx=10)

# Add some padding to the button and improve layout
btn_predict = tk.Button(window, text="Predict Traffic Situation", command=predict_traffic, width=30)
btn_predict.grid(row=8, columnspan=2, pady=15)

# Start the Tkinter event loop
window.mainloop()
