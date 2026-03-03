import cv2
import os

# Create folder if it doesn't exist
save_folder = "captured_images"
os.makedirs(save_folder, exist_ok=True)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

img_count = 0

print("Press 's' to save image")
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    # Save image when 's' is pressed
    if key == ord('s'):
        img_name = f"{save_folder}/image_{img_count}.png"
        cv2.imwrite(img_name, frame)
        print(f"Saved: {img_name}")
        img_count += 1

    # Quit when 'q' is pressed
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()