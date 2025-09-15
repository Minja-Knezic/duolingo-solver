import time
import cv2
import numpy as np
import pyautogui
import keyboard
import pygetwindow
import os
from datetime import datetime
import torch 
from torchvision import transforms
import requests

from ocr import extract_text_from_image

from classifier.cnn_model import SimpleCNN

from dotenv import load_dotenv


# === CONFIG ===
save_folder = "screenshots"
crop_window_hotkey = "F8"
screenshot_hotkey = "F9"
exit_hotkey = "F10"



os.makedirs(save_folder, exist_ok=True)
region = []


print("🔲 Please select a region by dragging the mouse.")

screen = pyautogui.screenshot()
screen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2BGR)

clone = screen.copy()
start_point = None
end_point = None
drawing = False

def mouse_crop(event, x, y, flags, param):
    global start_point, end_point, drawing, screen
    starting_screen = screen.copy()

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)
            temp = screen.copy()
            cv2.rectangle(temp, start_point, end_point, (0, 255, 0), 2)
            cv2.imshow("Select Region", temp)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.rectangle(screen, start_point, end_point, (0, 255, 0), 2)
        cv2.imshow("Select Region", screen)
        cv2.destroyWindow("Select Region")

    screen = starting_screen.copy()

def create_crop_window():    
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", mouse_crop)
    cv2.imshow("Select Region", screen)
    cv2.waitKey(0)

    # Convert coordinates to (left, top, width, height)
    x1, y1 = start_point
    x2, y2 = end_point
    left = min(x1, x2)
    top = min(y1, y2)
    width = abs(x2 - x1)
    height = abs(y2 - y1)
    region = (left, top, width, height)
    return region

def gen_message(content,type):
    message = "You are solving a Duolingo excercise. "
    if type == 0:
        message+= ("Write the following sentence in English, using the words given below: "
        f"{content}\n"
        "\nInstruction: Answer the question as if you are a Duolingo user. Only provide the direct answer without extra explanation."
        "There are extra words you don't need. You may only use each word once per number of appearances."
    )
    if type == 1:  
        message += (
        "Complete the following sentence by filling in the missing word(s): "
        f"{content}\n"
        "Instruction: Provide ONLY the missing word(s) as the answer, nothing else."
    )
    if type == 2:  
        message += (
        "Select the correct meaning of the given word from the provided options: "
        f"{content}\n"
        "Instruction: Answer ONLY with the correct option exactly as given (no explanations)."
    )
    if type == 3:  
        message += (
        "Translate the following sentence into English: "
        f"{content}\n"
        "Instruction: Provide ONLY the translation, nothing else."
    )
    return message

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def send_to_groq(content: str, type):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    message = gen_message(content,type)
    print(message)
    payload = {
        "model": "llama-3.3-70b-versatile",  
        "messages": [{"role": "user", "content":message}]
    }

    response = requests.post(url, headers=headers, json=payload)
    print("Response: ",response.json()["choices"][0]["message"]["content"])
    #return response.json()



def main():
    region = create_crop_window()
    print(f"✅ Region selected: {region}")
    print(f"📸 Press {screenshot_hotkey} to take screenshots.")

    # === STEP 2: Hotkey Listener Loop ===
    while True:
        event = keyboard.read_event()
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == screenshot_hotkey.lower():
                screenshot = pyautogui.screenshot(region=region)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = os.path.join(save_folder, f"screenshot_{timestamp}.png")
                
                transform = transforms.Compose([
                    transforms.Grayscale(num_output_channels=1),
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                input_tensor = transform(screenshot).unsqueeze(0)


                model = SimpleCNN()
                model.load_state_dict(torch.load("classifier/duolingo_cnn.pth"))
                model.eval()
                
                classes = ['choose words', 'fill blank', 'select meaning', 'type sentence']  
                with torch.no_grad():
                    output = model(input_tensor)
                    predicted_class = torch.argmax(output, dim=1).item()
                    print(f"🖼️ Predicted class: {classes[predicted_class]}")
                    extracted_text = extract_text_from_image(screenshot)
                    groq_response = send_to_groq(extracted_text,predicted_class)
                    print(groq_response)
                time.sleep(0.2)  
                

            elif event.name == exit_hotkey.lower():
                print("🔴 Exiting...")
                exit(0)

            elif event.name == crop_window_hotkey.lower():
                print("🔲 Please select a new region by dragging the mouse.")
                region = create_crop_window()
                print(f"✅ New region selected: {region}")



if __name__ == "__main__":
    main()