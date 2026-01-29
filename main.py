"""
Face Authentication Attendance System
A complete system with face registration, recognition, punch-in/out, and reporting
Designed for Google Colab with Gradio interface
"""

# Installation commands for Google Colab
# !pip install gradio opencv-python-headless face_recognition pillow pandas numpy

import gradio as gr
import cv2
import face_recognition
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from PIL import Image
import base64
from io import BytesIO

class FaceAttendanceSystem:
    def __init__(self):
        self.data_dir = "attendance_data"
        self.faces_db_path = os.path.join(self.data_dir, "faces_db.json")
        self.attendance_path = os.path.join(self.data_dir, "attendance.csv")
        
        # Create data directory
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize databases
        self.load_face_database()
        self.initialize_attendance()
        self.create_sample_images()
    
    def create_sample_images(self):
        """Create sample face images for testing"""
        sample_dir = os.path.join(self.data_dir, "sample_images")
        os.makedirs(sample_dir, exist_ok=True)
        
        # Create colored placeholder images with different patterns
        samples = [
            ("alice.jpg", (100, 150, 200)),
            ("bob.jpg", (200, 100, 150)),
            ("charlie.jpg", (150, 200, 100))
        ]
        
        for filename, color in samples:
            filepath = os.path.join(sample_dir, filename)
            if not os.path.exists(filepath):
                # Create a 400x400 image with a face-like pattern
                img = np.zeros((400, 400, 3), dtype=np.uint8)
                img[:] = color
                
                # Add simple face-like features (circles for eyes, mouth)
                cv2.circle(img, (150, 150), 20, (255, 255, 255), -1)  # Left eye
                cv2.circle(img, (250, 150), 20, (255, 255, 255), -1)  # Right eye
                cv2.ellipse(img, (200, 250), (50, 30), 0, 0, 180, (255, 255, 255), 2)  # Mouth
                
                cv2.imwrite(filepath, img)
        
        return f"Sample images created in {sample_dir}"
    
    def load_face_database(self):
        """Load or initialize face database"""
        if os.path.exists(self.faces_db_path):
            with open(self.faces_db_path, 'r') as f:
                data = json.load(f)
                self.face_encodings = {k: np.array(v) for k, v in data.items()}
        else:
            self.face_encodings = {}
    
    def save_face_database(self):
        """Save face database to file"""
        data = {k: v.tolist() for k, v in self.face_encodings.items()}
        with open(self.faces_db_path, 'w') as f:
            json.dump(data, f)
    
    def initialize_attendance(self):
        """Initialize or load attendance records"""
        if not os.path.exists(self.attendance_path):
            df = pd.DataFrame(columns=['Name', 'Date', 'Punch-In', 'Punch-Out', 'Duration'])
            df.to_csv(self.attendance_path, index=False)
    
    def register_face(self, image, name):
        """Register a new user's face"""
        if image is None:
            return "❌ No image provided", None
        
        if not name or name.strip() == "":
            return "❌ Please provide a name", None
        
        name = name.strip()
        
        # Convert image to RGB
        if isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.array(image)
        
        # Find face encodings
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return "❌ No face detected. Please ensure your face is clearly visible.", None
        
        if len(face_encodings) > 1:
            return "❌ Multiple faces detected. Please ensure only one face is in the frame.", None
        
        # Save the encoding
        self.face_encodings[name] = face_encodings[0]
        self.save_face_database()
        
        # Draw rectangle around face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return f"✅ Successfully registered {name}!", rgb_image
    
    def identify_face(self, image):
        """Identify a face from the image"""
        if image is None:
            return "❌ No image provided", None, None
        
        if len(self.face_encodings) == 0:
            return "❌ No registered users. Please register faces first.", None, None
        
        # Convert image to RGB
        if isinstance(image, np.ndarray):
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = np.array(image)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_image)
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        
        if len(face_encodings) == 0:
            return "❌ No face detected", None, None
        
        # Compare with known faces
        identified_name = "Unknown"
        min_distance = float('inf')
        
        for name, known_encoding in self.face_encodings.items():
            distances = face_recognition.face_distance([known_encoding], face_encodings[0])
            if distances[0] < min_distance and distances[0] < 0.6:  # Threshold
                min_distance = distances[0]
                identified_name = name
        
        # Draw result
        top, right, bottom, left = face_locations[0]
        color = (0, 255, 0) if identified_name != "Unknown" else (255, 0, 0)
        cv2.rectangle(rgb_image, (left, top), (right, bottom), color, 2)
        cv2.putText(rgb_image, identified_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if identified_name != "Unknown":
            return f"✅ Identified: {identified_name}", rgb_image, identified_name
        else:
            return "❌ Face not recognized", rgb_image, None
    
    def punch_in(self, image):
        """Record punch-in time"""
        message, annotated_image, name = self.identify_face(image)
        
        if name is None:
            return message, annotated_image
        
        # Load attendance
        df = pd.read_csv(self.attendance_path)
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Check if already punched in today
        existing = df[(df['Name'] == name) & (df['Date'] == today)]
        
        if not existing.empty and pd.notna(existing.iloc[0]['Punch-In']):
            return f"⚠️ {name} already punched in today at {existing.iloc[0]['Punch-In']}", annotated_image
        
        # Create new entry
        new_entry = pd.DataFrame({
            'Name': [name],
            'Date': [today],
            'Punch-In': [current_time],
            'Punch-Out': [None],
            'Duration': [None]
        })
        
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(self.attendance_path, index=False)
        
        return f"✅ {name} punched in at {current_time}", annotated_image
    
    def punch_out(self, image):
        """Record punch-out time"""
        message, annotated_image, name = self.identify_face(image)
        
        if name is None:
            return message, annotated_image
        
        # Load attendance
        df = pd.read_csv(self.attendance_path)
        today = datetime.now().strftime("%Y-%m-%d")
        current_time = datetime.now().strftime("%H:%M:%S")
        
        # Find today's entry
        mask = (df['Name'] == name) & (df['Date'] == today) & (df['Punch-Out'].isna())
        
        if not mask.any():
            return f"⚠️ {name} hasn't punched in today or already punched out", annotated_image
        
        # Update punch-out time
        idx = df[mask].index[0]
        punch_in_time = datetime.strptime(df.loc[idx, 'Punch-In'], "%H:%M:%S")
        punch_out_time = datetime.strptime(current_time, "%H:%M:%S")
        duration = punch_out_time - punch_in_time
        
        df.loc[idx, 'Punch-Out'] = current_time
        df.loc[idx, 'Duration'] = str(duration)
        df.to_csv(self.attendance_path, index=False)
        
        return f"✅ {name} punched out at {current_time}. Duration: {duration}", annotated_image
    
    def generate_report(self, start_date, end_date):
        """Generate attendance report"""
        df = pd.read_csv(self.attendance_path)
        
        if df.empty:
            return "No attendance records found"
        
        # Filter by date range if provided
        if start_date:
            df = df[df['Date'] >= start_date]
        if end_date:
            df = df[df['Date'] <= end_date]
        
        if df.empty:
            return "No records in the selected date range"
        
        return df
    
    def get_registered_users(self):
        """Get list of registered users"""
        if not self.face_encodings:
            return "No registered users"
        return "\n".join([f"• {name}" for name in self.face_encodings.keys()])

# Initialize system
system = FaceAttendanceSystem()

# Create Gradio Interface
with gr.Blocks(title="Face Authentication Attendance System", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 👤 Face Authentication Attendance System")
    gr.Markdown("### AI/ML Intern Assignment - Complete Implementation")
    
    with gr.Tabs():
        # Tab 1: Register Face
        with gr.Tab("📝 Register New User"):
            gr.Markdown("### Register a new user by uploading their photo")
            with gr.Row():
                with gr.Column():
                    register_image = gr.Image(label="Upload Face Image", type="numpy")
                    register_name = gr.Textbox(label="Name", placeholder="Enter user's name")
                    register_btn = gr.Button("Register Face", variant="primary")
                with gr.Column():
                    register_output = gr.Textbox(label="Status")
                    register_preview = gr.Image(label="Registered Face Preview")
            
            register_btn.click(
                system.register_face,
                inputs=[register_image, register_name],
                outputs=[register_output, register_preview]
            )
        
        # Tab 2: Punch In
        with gr.Tab("⏰ Punch In"):
            gr.Markdown("### Capture your face to punch in")
            with gr.Row():
                with gr.Column():
                    punchin_image = gr.Image(label="Capture/Upload Image", type="numpy")
                    punchin_btn = gr.Button("Punch In", variant="primary")
                with gr.Column():
                    punchin_output = gr.Textbox(label="Status")
                    punchin_preview = gr.Image(label="Recognized Face")
            
            punchin_btn.click(
                system.punch_in,
                inputs=[punchin_image],
                outputs=[punchin_output, punchin_preview]
            )
        
        # Tab 3: Punch Out
        with gr.Tab("🏁 Punch Out"):
            gr.Markdown("### Capture your face to punch out")
            with gr.Row():
                with gr.Column():
                    punchout_image = gr.Image(label="Capture/Upload Image", type="numpy")
                    punchout_btn = gr.Button("Punch Out", variant="primary")
                with gr.Column():
                    punchout_output = gr.Textbox(label="Status")
                    punchout_preview = gr.Image(label="Recognized Face")
            
            punchout_btn.click(
                system.punch_out,
                inputs=[punchout_image],
                outputs=[punchout_output, punchout_preview]
            )
        
        # Tab 4: Attendance Report
        with gr.Tab("📊 Attendance Report"):
            gr.Markdown("### View and download attendance records")
            with gr.Row():
                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", placeholder="2024-01-01")
                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", placeholder="2024-12-31")
            report_btn = gr.Button("Generate Report", variant="primary")
            report_output = gr.Dataframe(label="Attendance Records")
            
            report_btn.click(
                system.generate_report,
                inputs=[start_date, end_date],
                outputs=[report_output]
            )
        
        # Tab 5: System Info
        with gr.Tab("ℹ️ System Info"):
            gr.Markdown("### Registered Users and System Information")
            users_display = gr.Textbox(label="Registered Users", lines=10)
            refresh_btn = gr.Button("Refresh User List")
            
            refresh_btn.click(
                system.get_registered_users,
                outputs=[users_display]
            )
            
            gr.Markdown("""
            ### 📝 Documentation
            
            **System Features:**
            - ✅ Face registration with real-time detection
            - ✅ Face recognition for attendance
            - ✅ Punch-in/Punch-out tracking
            - ✅ Attendance report generation
            - ✅ Works with real camera or uploaded images
            
            **How to Use:**
            1. **Register Users**: Upload clear face photos in the "Register New User" tab
            2. **Punch In**: Capture/upload image when starting work
            3. **Punch Out**: Capture/upload image when finishing work
            4. **View Reports**: Check attendance records in the "Attendance Report" tab
            
            **Sample Images**: Sample test images are available in `attendance_data/sample_images/`
            
            **Known Limitations:**
            - Lighting conditions affect recognition accuracy
            - Works best with frontal face images
            - Requires clear, unobstructed face visibility
            - May fail with masks, sunglasses, or extreme angles
            
            **Model Used:** face_recognition library (based on dlib's ResNet)
            **Accuracy:** ~99.38% on LFW benchmark (under ideal conditions)
            """)

# Launch the app
if __name__ == "__main__":
    print("🚀 Starting Face Authentication Attendance System...")
    print(f"📁 Data directory: {system.data_dir}")
    print(f"👥 Registered users: {len(system.face_encodings)}")
    app.launch(share=True, debug=True)
