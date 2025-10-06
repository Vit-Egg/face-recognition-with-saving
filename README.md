# face-recognition-with-saving
Real-time face recognition using OpenCV with sound alerts and automatic image saving.

# Face Recognition with Automatic Saving

A Python project that uses OpenCV to recognize faces from a live camera feed, play a sound when a known person is recognized, and automatically save images if the face remains visible for a certain period.

## 🧠 Features
- Real-time face recognition using OpenCV (LBPH).
- Sound alerts for selected recognized persons.
- Automatic saving of recognized face snapshots.
- Adjustable minimum time a face must stay visible before saving.
- Global delay that prevents repeated saving within short intervals.

## 📦 Requirements
- Python 3.11 or newer  
- Packages:
  - `opencv-contrib-python`
  - `numpy`
  - `pygame`

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Vit-Egg/face-recognition-with-saving.git
   cd face-recognition-with-saving
   ```
2. Install dependencies:
   ```bash
   pip install opencv-contrib-python numpy pygame
   ```
3. Create the necessary folders:
   ```bash
   mkdir known_faces captures
   ```
4. Add folders for each person you want to recognize under `known_faces`, for example:
   ```
   known_faces/
   ├── mom/
   │   ├── mom1.jpg
   │   ├── mom2.jpg
   │   └── mom3.jpg
   ├── dad/
   │   ├── dad1.jpg
   │   ├── dad2.jpg
   │   └── dad3.jpg
   └── sister/
       ├── sister1.jpg
       ├── sister2.jpg
       └── sister3.jpg
   ```

## ▶️ Running
Run the script in your terminal:
```bash
python main.py
```
- Press `q` to quit the program.
- Adjust parameters in the script:
  - `ALERT_ONLY_FOR` – list of names for which the sound should play.
  - `MIN_PRESENCE_SECONDS` – minimum duration the face must be visible before saving.
  - `GLOBAL_SAVE_DELAY` – minimum time interval between consecutive saved captures.

## 📁 Folder Structure
```
face-recognition-with-saving/
── known_faces/                     # folder with known persons (input data)
│   ├── mom/
│   │   ├── mom1.jpg
│   │   ├── mom2.jpg
│   │   └── mom3.jpg
│   │
│   ├── dad/
│   │   ├── dad1.jpg
│   │   ├── dad2.jpg
│   │   └── dad3.jpg
│   │
│   └── sister/
│       ├── sister1.jpg
│       ├── sister2.jpg
│       └── sister3.jpg
├── captures/          # Automatically saved snapshots
├── main.py
├── alert.mp3          # Sound alert
└── README.md
```

## 🧩 Tips
- For best results, use clear, front-facing photos of each person.
- If no faces are being detected, check the image formats (preferably JPEG, RGB).
- A camera resolution of at least 720p is recommended for reliable recognition.
