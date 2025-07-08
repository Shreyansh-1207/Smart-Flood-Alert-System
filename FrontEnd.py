import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser # To open the generated map in a browser

# Suppress warnings from scikit-learn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class FloodAlertSystemApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Flood Alert System")
        self.root.geometry("1200x800") # Increased window size
        self.root.configure(bg="#F0F2F5") # Light grey background

        # DataFrames to store simulated data
        self.current_weather_df = pd.DataFrame()
        self.historical_floods_df = pd.DataFrame()
        self.model = None # To store the trained model

        self.create_widgets()

    def create_widgets(self):
        # Styling
        self.style = ttk.Style()
        self.style.theme_use('clam') # Modern theme
        self.style.configure('TFrame', background='#F0F2F5')
        self.style.configure('TLabel', background='#F0F2F5', font=('Arial', 10))
        self.style.configure('TButton', font=('Arial', 10, 'bold'), background='#007BFF', foreground='white')
        self.style.map('TButton', background=[('active', '#0056b3')])
        self.style.configure('TEntry', font=('Arial', 10))
        self.style.configure('TScrolledText', font=('Consolas', 9))

        # Main frame
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Notebook (tabs for different sections)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # --- Data Simulation Tab ---
        simulation_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(simulation_frame, text="Data Simulation")
        self.create_simulation_tab(simulation_frame)

        # --- Data Analysis Tab ---
        analysis_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(analysis_frame, text="Data Analysis")
        self.create_analysis_tab(analysis_frame)

        # --- Flood Prediction Tab ---
        prediction_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(prediction_frame, text="Flood Prediction")
        self.create_prediction_tab(prediction_frame)

        # --- SMS Alert Tab ---
        sms_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(sms_frame, text="SMS Alert")
        self.create_sms_tab(sms_frame)

        # --- Map Visualization Tab ---
        map_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(map_frame, text="Map Visualization")
        self.create_map_tab(map_frame)

    def create_simulation_tab(self, parent_frame):
        # Current Weather Data Simulation
        ttk.Label(parent_frame, text="Current Weather Data Simulation", font=('Arial', 12, 'bold')).pack(pady=10)
        
        frame_current = ttk.Frame(parent_frame)
        frame_current.pack(fill=tk.X, pady=5)
        ttk.Label(frame_current, text="Number of entries:").pack(side=tk.LEFT, padx=5)
        self.num_current_entries = ttk.Entry(frame_current, width=10)
        self.num_current_entries.insert(0, "20")
        self.num_current_entries.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_current, text="Simulate Current Data", command=self.simulate_current_data).pack(side=tk.LEFT, padx=5)
        
        self.current_data_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=10)
        self.current_data_output.pack(fill=tk.BOTH, expand=True, pady=10)

        # Historical Flood Data Simulation
        ttk.Label(parent_frame, text="Historical Flood Data Simulation", font=('Arial', 12, 'bold')).pack(pady=10)

        frame_historical = ttk.Frame(parent_frame)
        frame_historical.pack(fill=tk.X, pady=5)
        ttk.Label(frame_historical, text="Number of entries:").pack(side=tk.LEFT, padx=5)
        self.num_historical_entries = ttk.Entry(frame_historical, width=10)
        self.num_historical_entries.insert(0, "30")
        self.num_historical_entries.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_historical, text="Simulate Historical Data", command=self.simulate_historical_data).pack(side=tk.LEFT, padx=5)

        self.historical_data_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=10)
        self.historical_data_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_analysis_tab(self, parent_frame):
        ttk.Label(parent_frame, text="Data Analysis", font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Button(parent_frame, text="Perform Analysis", command=self.perform_analysis).pack(pady=5)
        self.analysis_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=20)
        self.analysis_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_prediction_tab(self, parent_frame):
        ttk.Label(parent_frame, text="Flood Prediction", font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Button(parent_frame, text="Train Model and Predict", command=self.train_and_predict).pack(pady=5)
        self.prediction_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=20)
        self.prediction_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_sms_tab(self, parent_frame):
        ttk.Label(parent_frame, text="SMS Alert System (Conceptual)", font=('Arial', 12, 'bold')).pack(pady=10)
        
        ttk.Label(parent_frame, text="Phone Number (e.g., +1234567890):").pack(pady=5)
        self.phone_number_entry = ttk.Entry(parent_frame, width=30)
        self.phone_number_entry.pack(pady=5)
        
        ttk.Label(parent_frame, text="Message:").pack(pady=5)
        self.sms_message_text = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=5)
        self.sms_message_text.pack(fill=tk.X, pady=5)
        self.sms_message_text.insert(tk.END, "Flood warning: High risk in your area. Take necessary precautions.")

        ttk.Button(parent_frame, text="Send Simulated SMS Alert", command=self.send_simulated_sms).pack(pady=10)
        self.sms_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=5)
        self.sms_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def create_map_tab(self, parent_frame):
        ttk.Label(parent_frame, text="Map Visualization", font=('Arial', 12, 'bold')).pack(pady=10)
        ttk.Button(parent_frame, text="Generate and Open Map", command=self.generate_and_open_map).pack(pady=5)
        self.map_output = scrolledtext.ScrolledText(parent_frame, wrap=tk.WORD, height=15)
        self.map_output.pack(fill=tk.BOTH, expand=True, pady=10)
        self.map_output.insert(tk.END, "Click 'Generate and Open Map' to visualize simulated flood risk on an interactive map. This will open in your default web browser.")


    # --- Simulation Functions ---
    def generate_weather_data(self, num_entries=10):
        """Generates synthetic current weather data."""
        data = []
        locations = [
            'Haripur', 'Shantipur', 'Ramnagar', 'Krishnapur', 'Faizabad', 'Varanasi',
            'Cuttack', 'Guwahati', 'Patna', 'Surat', 'Kochi', 'Hyderabad',
            'Chennai', 'Pune', 'Bhopal', 'Jaipur'
        ]
        for i in range(num_entries):
            timestamp = datetime.now() - timedelta(minutes=i*30)
            rainfall_amount = round(random.uniform(5, 60), 2)
            water_level = round(random.uniform(5, 12), 2)
            temperature = round(random.uniform(20, 35), 1)
            humidity = round(random.uniform(60, 95), 1)
            location = random.choice(locations)
            data.append({
                'timestamp': timestamp,
                'location': location,
                'rainfall_amount_mm': rainfall_amount,
                'water_level_m': water_level,
                'temperature_c': temperature,
                'humidity_percent': humidity
            })
        return pd.DataFrame(data)

    def generate_historical_flood_data(self, num_entries=15):
        """Generates synthetic historical flood records."""
        data = []
        locations = [
            'Haripur', 'Shantipur', 'Ramnagar', 'Krishnapur', 'Faizabad', 'Varanasi',
            'Cuttack', 'Guwahati', 'Patna', 'Surat', 'Kochi', 'Hyderabad',
            'Chennai', 'Pune', 'Bhopal', 'Jaipur'
        ]
        severities = ['Low', 'Medium', 'High']
        for i in range(num_entries):
            timestamp = datetime.now() - timedelta(days=random.randint(30, 365*5))
            rainfall_amount = round(random.uniform(40, 150), 2)
            water_level = round(random.uniform(9, 15), 2)
            location = random.choice(locations)
            severity = random.choice(severities)
            deaths = 0
            if severity == 'High':
                deaths = random.randint(1, 10)
            elif severity == 'Medium':
                deaths = random.randint(0, 3)

            data.append({
                'timestamp': timestamp,
                'location': location,
                'rainfall_amount_mm': rainfall_amount,
                'water_level_m': water_level,
                'severity': severity,
                'deaths': deaths
            })
        return pd.DataFrame(data)

    def simulate_current_data(self):
        try:
            num = int(self.num_current_entries.get())
            self.current_weather_df = self.generate_weather_data(num_entries=num)
            self.current_data_output.delete(1.0, tk.END)
            self.current_data_output.insert(tk.END, "Simulated Current Weather Data:\n")
            self.current_data_output.insert(tk.END, self.current_weather_df.to_string())
            messagebox.showinfo("Simulation Complete", f"{num} entries of current weather data simulated.")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for current data entries.")

    def simulate_historical_data(self):
        try:
            num = int(self.num_historical_entries.get())
            self.historical_floods_df = self.generate_historical_flood_data(num_entries=num)
            self.historical_data_output.delete(1.0, tk.END)
            self.historical_data_output.insert(tk.END, "Simulated Historical Flood Records:\n")
            self.historical_data_output.insert(tk.END, self.historical_floods_df.to_string())
            messagebox.showinfo("Simulation Complete", f"{num} entries of historical flood data simulated.")
        except ValueError:
            messagebox.showerror("Input Error", "Please enter a valid number for historical data entries.")

    # --- Analysis Function ---
    def perform_analysis(self):
        self.analysis_output.delete(1.0, tk.END)
        if self.historical_floods_df.empty:
            self.analysis_output.insert(tk.END, "No historical flood data available. Please simulate data first.")
            return

        output_text = ""
        avg_rainfall_flood = self.historical_floods_df['rainfall_amount_mm'].mean()
        avg_water_level_flood = self.historical_floods_df['water_level_m'].mean()
        output_text += f"Average Rainfall during Historical Floods: {avg_rainfall_flood:.2f} mm\n"
        output_text += f"Average Water Level during Historical Floods: {avg_water_level_flood:.2f} m\n\n"

        severity_counts = self.historical_floods_df['severity'].value_counts()
        output_text += "Historical Flood Severity Counts:\n"
        output_text += severity_counts.to_string() + "\n\n"

        if 'deaths' in self.historical_floods_df.columns:
            total_deaths = self.historical_floods_df['deaths'].sum()
            output_text += f"Total Recorded Deaths from Historical Floods: {total_deaths}\n\n"
        
        output_text += "Descriptive Statistics for Historical Flood Data:\n"
        output_text += self.historical_floods_df[['rainfall_amount_mm', 'water_level_m', 'deaths']].describe().to_string()

        self.analysis_output.insert(tk.END, output_text)
        messagebox.showinfo("Analysis Complete", "Data analysis performed successfully.")

    # --- Prediction Function ---
    def train_and_predict(self):
        self.prediction_output.delete(1.0, tk.END)
        output_text = ""

        if self.historical_floods_df.empty or self.current_weather_df.empty:
            output_text += "Please simulate both historical and current weather data first.\n"
            self.prediction_output.insert(tk.END, output_text)
            return

        # Prepare historical data for training
        X_hist = self.historical_floods_df[['rainfall_amount_mm', 'water_level_m']]
        y_hist = self.historical_floods_df['severity'].apply(lambda x: 1 if x in ['Medium', 'High'] else 0) # Binary classification: Flood (1) or No Flood (0)

        if len(X_hist) < 2:
            output_text += "Not enough historical data to train the model. Need at least 2 entries.\n"
            self.prediction_output.insert(tk.END, output_text)
            return

        X_train, X_test, y_train, y_test = train_test_split(X_hist, y_hist, test_size=0.3, random_state=42)

        # Train Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        output_text += "Random Forest Classifier trained successfully.\n\n"

        # Evaluate model
        if not X_test.empty:
            y_pred = self.model.predict(X_test)
            output_text += "Model Evaluation on Test Data:\n"
            output_text += f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n"
            output_text += f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}\n"
            output_text += f"Recall: {recall_score(y_test, y_pred, zero_division=0):.2f}\n"
            output_text += f"F1-Score: {f1_score(y_test, y_pred, zero_division=0):.2f}\n\n"
            output_text += "Classification Report:\n"
            output_text += classification_report(y_test, y_pred, zero_division=0) + "\n\n"
        else:
            output_text += "Not enough test data to evaluate the model.\n\n"

        # Rule-based Prediction (using current weather data)
        output_text += "Rule-based Flood Risk Assessment (Current Data):\n"
        for index, row in self.current_weather_df.iterrows():
            location = row['location']
            rainfall = row['rainfall_amount_mm']
            water_level = row['water_level_m']
            risk = "Low"
            if rainfall > 50 or water_level > 10:
                risk = "High"
            elif rainfall > 30 or water_level > 8:
                risk = "Medium"
            output_text += f"  {location}: Rainfall={rainfall}mm, Water Level={water_level}m, Risk={risk}\n"
        output_text += "\n"

        # ML-based Prediction (using current weather data)
        if self.model:
            output_text += "ML-based Flood Risk Prediction (Current Data):\n"
            X_current = self.current_weather_df[['rainfall_amount_mm', 'water_level_m']]
            current_predictions = self.model.predict(X_current)
            for i, pred in enumerate(current_predictions):
                location = self.current_weather_df.iloc[i]['location']
                risk_level = "Flood Risk" if pred == 1 else "No Flood Risk"
                output_text += f"  {location}: {risk_level}\n"
        else:
            output_text += "ML model not trained. Please train the model first.\n"

        self.prediction_output.insert(tk.END, output_text)
        messagebox.showinfo("Prediction Complete", "Flood prediction performed successfully.")

    # --- SMS Alert Function ---
    def send_simulated_sms(self):
        phone_number = self.phone_number_entry.get()
        message = self.sms_message_text.get(1.0, tk.END).strip()
        
        self.sms_output.delete(1.0, tk.END)
        if not phone_number:
            self.sms_output.insert(tk.END, "Please enter a phone number.\n")
            messagebox.showerror("Input Error", "Phone number cannot be empty.")
            return

        # This is a simulated SMS sending. In a real application, you would integrate with Twilio or similar.
        self.sms_output.insert(tk.END, f"Simulating SMS to: {phone_number}\n")
        self.sms_output.insert(tk.END, f"Message: {message}\n")
        self.sms_output.insert(tk.END, "SMS simulation complete. (No actual SMS sent)\n")
        messagebox.showinfo("SMS Alert", "Simulated SMS alert sent successfully!")

    # --- Map Visualization Function ---
    def generate_and_open_map(self):
        self.map_output.delete(1.0, tk.END)
        if self.current_weather_df.empty:
            self.map_output.insert(tk.END, "No current weather data to visualize. Please simulate data first.\n")
            return

        # Create a base map centered around a general location in India
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

        # Simulate coordinates for locations (for demonstration purposes)
        # In a real system, you'd use actual geocoding for locations
        location_coords = {
            'Haripur': [22.3511, 71.8708], 'Shantipur': [22.9964, 88.6293],
            'Ramnagar': [22.3167, 87.2167], 'Krishnapur': [22.5726, 88.3639],
            'Faizabad': [26.7725, 82.1333], 'Varanasi': [25.3176, 82.9739],
            'Cuttack': [20.4625, 85.8830], 'Guwahati': [26.1445, 91.7362],
            'Patna': [25.5941, 85.1376], 'Surat': [21.1702, 72.8311],
            'Kochi': [9.9312, 76.2673], 'Hyderabad': [17.3850, 78.4867],
            'Chennai': [13.0827, 80.2707], 'Pune': [18.5204, 73.8567],
            'Bhopal': [23.2599, 77.4126], 'Jaipur': [26.9124, 75.7873]
        }

        # Add markers based on simulated current weather data and rule-based risk
        for index, row in self.current_weather_df.iterrows():
            village = row['location']
            rainfall = row['rainfall_amount_mm']
            water_level = row['water_level_m']
            coords = location_coords.get(village, [20.5937, 78.9629]) # Default if not found

            risk_level = "Low"
            if rainfall > 50 or water_level > 10:
                risk = "High"
            elif rainfall > 30 or water_level > 8:
                risk = "Medium"
            else:
                risk = "Low"
            
            marker_color = 'blue' # Default color
            icon_type = 'info-sign' # Default icon

            if risk == 'High':
                marker_color = 'red'
                icon_type = 'exclamation-sign'
            elif risk == 'Medium':
                marker_color = 'orange'
                icon_type = 'warning-sign'
            elif risk == 'Low':
                marker_color = 'green'
                icon_type = 'ok-sign'

            folium.Marker(
                location=coords,
                popup=f"<b>{village}</b><br>Lat: {coords[0]}, Lon: {coords[1]}<br>Rainfall: {rainfall}mm<br>Water Level: {water_level}m<br>Risk: {risk}",
                tooltip=village,
                icon=folium.Icon(color=marker_color, icon=icon_type)
            ).add_to(m)

        # Save the map to an HTML file
        map_file = "flood_risk_map.html"
        m.save(map_file)
        
        # Open the HTML file in the default web browser
        webbrowser.open_new_tab(map_file)
        self.map_output.insert(tk.END, f"Map generated and saved to {map_file}. Opening in browser...\n")
        messagebox.showinfo("Map Generated", f"Interactive map saved to {map_file} and opened in your browser.")


if __name__ == "__main__":
    root = tk.Tk()
    app = FloodAlertSystemApp(root)
    root.mainloop()