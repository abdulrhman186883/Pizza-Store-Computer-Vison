AI Pizza Hygiene Monitor
This repository contains a real-time AI system designed to monitor food safety. It detects barehand contact in a kitchen environment while intelligently ignoring contact when tools or pizzas are being handled.


🛠️ Installation & Setup

1. Clone and Prepare the Folder
Open your terminal (or Command Prompt) and navigate to your project folder:

2. Create a Virtual Environment
python -m venv venv
venv\Scripts\activate

3. Install Requirements
pip install -r requirements.txt
---------------------------------------------------

How to Run the System ?
- Launch the Backend
  python main.py

  You should see a message saying: Uvicorn running on http://127.0.0.1:8000

  2. Open the Dashboard
Open your web browser and go to: http://127.0.0.1:8000

3. Configure the Camera
In the Step 1: Select Video Source section, click the dropdown and choose Live Camera (Index).

Enter 0 in the box (this is usually the default ID for built-in webcams).

Click Connect.

Draw the Zone: Use your mouse to click points on the live frame to create a polygon around the "Hygiene Zone."

Click Start Monitoring.

--------------------------------------------------
System Architecture

The project follows a Microservices-style logic split into three main parts:

Frontend (HTML5/JS): Captures user input (ROI coordinates) and displays the processed stream.

API (FastAPI): Coordinates the data flow between the UI and the AI.

AI Engine (YOLOv12): Runs in a background thread to perform real-time tracking and violation logic.


--------------------------------------------------
# 🔍 Monitoring Cases & Logic

The system is designed to distinguish between standard kitchen operations and actual hygiene violations. Below are the primary detection cases:

### 1. Bare Hand Detection (Warning)
![Bare Hand](Bare%20Hand.png)
*When a bare hand enters the designated Hygiene Zone, the system starts a 1.5-second grace period timer, highlighted in yellow.*

### 2. Hygiene Violation (Flagged)
![Violation](Violation%20bare%20hand%20.png)
*If a bare hand remains in the zone beyond the grace period without using a tool, a violation is recorded and the counter increases.*

### 3. Safe Operation: Handling Pizza
![Hands on Pizza](hands%20on%20pizza.png)
*The AI recognizes when a hand is interacting directly with the pizza (processing/moving), suppressing the violation alert.*

### 4. Safe Operation: Using Tools
![With Spooner](with%20spooner.png)
*When the system detects a collision between a hand and a tool (spoon, scoop, or spooner), it marks the action as SAFE.*
