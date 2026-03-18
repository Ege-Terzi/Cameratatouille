# Cameratatouille - Cooking Assistant with Computer Vision
An intelligent cooking assistant powered by computer vision that guides users through recipes in real-time by detecting ingredients and tracking cooking progress from video input.

🔍 Ingredient Detection
Detects food items using YOLOv8 in real-time

🥣 Bowl State Classification
Understands what's inside the bowl (e.g. empty, beaten eggs)

📋 Step-by-Step Guidance
Displays recipe steps and tracks progress visually

🤖 Auto Mode
Automatically advances steps when conditions are met

🎯 Stable Detection System
Reduces flickering using temporal smoothing

🖥️ Custom UI Overlay
Clean real-time interface with:

Current step

Completed steps

Missing ingredients

How It Works

<img width="2901" height="130" alt="mermaid-diagram" src="https://github.com/user-attachments/assets/9213bf8c-b842-4387-8add-51f3c27c6941" />

▶️ Usage
python cooking_assistant.py

Select a recipe from terminal

Video starts

Follow instructions on screen

🎮 Controls
Key	Action
n	Next step
p	Previous step
c	Toggle complete
a	Toggle auto mode
space	Pause
q	Quit

🛠️ Tech Stack

🧠 PyTorch

🔍 YOLOv8 (Ultralytics)

🎥 OpenCV

🖼️ PIL / torchvision

📈 Future Improvements

🎤 Voice guidance

📱 Mobile / web interface

🍲 Multi-recipe tracking

🧠 More food state classifiers

🧍 Multi-user support

📷 Webcam input

📄 License

This project is licensed under the MIT License.

⭐ If you like this project

Give it a star ⭐ and feel free to contribute!
