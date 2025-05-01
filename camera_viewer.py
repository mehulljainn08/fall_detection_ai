import torch
import cv2
import numpy as np
from time import sleep, time
import imagezmq

# --- Twilio Config (Optional Mock for Local Testing) ---
USE_TWILIO = False  # Set True to actually send SMS
ACCOUNT_SID = 'ACd801468e46b5797d4a5c7accd4e746a1'
AUTH_TOKEN = 'fe4db7dc9eb3d1c9f91aa01a5ecb59c6'
TWILIO_NUMBER = '+918949747407'
TARGET_NUMBER = '+919630413313'

def send_fall_alert():
    if USE_TWILIO:
        from twilio.rest import Client
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        message = client.messages.create(
            body="🚨 Alert: A fall has been detected!",
            from_=TWILIO_NUMBER,
            to=TARGET_NUMBER
        )
        print(f"SMS sent! SID: {message.sid}")
    else:
        print("🚨 [TEST MODE] Fall detected! (SMS not sent)")

# --- YOLOv5 Setup ---
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
model.conf = 0.5  # Confidence threshold

# --- imagezmq Receiver Setup ---
image_hub = imagezmq.ImageHub(open_port='tcp://*:5555')  # Must match port on Pi

# --- Cooldown ---
last_alert_time = 0
alert_cooldown = 60  # seconds

print("🟢 Fall detection started. Press Ctrl+C to stop.")

try:
    while True:
        rpi_name, frame = image_hub.recv_image()
        image_hub.send_reply(b'OK')

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run YOLO inference
        results = model(img_rgb)
        labels = results.pandas().xyxy[0]['name'].tolist()

        # Fall Detection Logic
        if "Fall-Detected" in labels:
            print("🚨 Fall Detected!")

            current_time = time()
            if current_time - last_alert_time > alert_cooldown:
                send_fall_alert()
                last_alert_time = current_time
            else:
                print("⏱️ Cooldown active, not sending SMS.")

        # Show annotated frame
        annotated = np.squeeze(results.render())
        cv2.imshow("📷 Fall Detection (from Pi)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("🛑 Detection stopped by user.")

finally:
    cv2.destroyAllWindows()