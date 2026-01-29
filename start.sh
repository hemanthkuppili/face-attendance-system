#!/bin/bash
# Render.com startup script

echo "Installing system dependencies..."
apt-get update
apt-get install -y cmake libopenblas-dev liblapack-dev

echo "Starting Face Attendance System..."
python face_attendance_system.py
