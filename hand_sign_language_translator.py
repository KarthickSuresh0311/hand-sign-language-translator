"""
Hand Sign Language Translator
Author: Karthick Suresh
Description: End-to-end hand sign language translator using CV and DL
"""

# ======================
# 1. Imports
# ======================
import cv2
import numpy as np
import tensorflow as tf

# ======================
# 2. Configuration
# ======================
IMG_SIZE = 64
MODEL_PATH = "model.h5"

# ======================
# 3. Data Preprocessing
# ======================
def preprocess_frame(frame):
    pass

# ======================
# 4. Model Definition
# ======================
def build_model():
    pass

# ======================
# 5. Training Function
# ======================
def train_model():
    pass

# ======================
# 6. Real-time Prediction
# ======================
def run_realtime():
    pass

# ======================
# 7. Main
# ======================
if __name__ == "__main__":
    # train_model()   # run once
    run_realtime()    # default

import os
import json
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 on ASL alphabet')
    parser.add_argument('--data', type=str, default=os.path.expanduser('~/ASL_dataset/ASL_images/asl_alphabet_train'), help='path to dataset root')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output-dir', type=str, default='output')
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    full_dataset = datasets.ImageFolder(args.data, transform=transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print('Found classes (count):', num_classes)

    # Save class mapping
    with open(os.path.join(args.output_dir, 'classes.json'), 'w') as f:
        json.dump(classes, f)

    # Split: 70% train, 15% val, 15% test
    n = len(full_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_acc_list, val_acc_list, loss_list = [], [], []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} - Train')
        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=running_loss/ (total / images.size(0)), acc=100.*correct/total)

        avg_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        loss_list.append(avg_loss)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100.0 * correct / total
        val_acc_list.append(val_acc)

        print(f'Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} Train Acc: {train_acc:.2f}% Val Acc: {val_acc:.2f}%')

        # Optionally save checkpoint each epoch
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc
        }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # Final save
    final_path = os.path.join(args.output_dir, 'asl_resnet18_final.pth')
    torch.save({'model_state_dict': model.state_dict(), 'classes': classes}, final_path)
    print('Model saved to', final_path)

    # Plot accuracy and loss
    plt.figure()
    plt.plot(range(1, args.epochs+1), train_acc_list, label='Train Accuracy')
    plt.plot(range(1, args.epochs+1), val_acc_list, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Train vs Val Accuracy')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'accuracy.png'))

    plt.figure()
    plt.plot(range(1, args.epochs+1), loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss.png'))

    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    print(f'Final Test Accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    main()


# ======================================
# File: inference_gui.py
# Real-time inference GUI using MediaPipe + PyTorch model
# ======================================

import json
import time
import threading
import pickle

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import pyttsx3
import tkinter as tk
from tkinter import StringVar, Label, Button, Frame
from PIL import Image, ImageTk

# -----------------------------
# Load model and class map
# -----------------------------
MODEL_PATH = 'output/asl_resnet18_final.pth'  # path where train.py saved the final model
CLASSES_JSON = 'output/classes.json'
IMG_SIZE = 64
DEVICE = torch.device('cpu')  # inference on CPU; change to cuda if available

# Build model architecture (must match train.py)
num_classes = None
with open(CLASSES_JSON, 'r') as f:
    classes = json.load(f)
    num_classes = len(classes)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE)['model_state_dict'])
model.to(DEVICE)
model.eval()

# label mapping to characters (if your classes are letters/digits leave as is)
label_map = {i: classes[i] for i in range(len(classes))}

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, max_num_hands=1)

# -----------------------------
# TTS
# -----------------------------
engine = pyttsx3.init()

def speak_text(text):
    def tts_thread():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=tts_thread, daemon=True).start()

# -----------------------------
# GUI variables
# -----------------------------
stabilization_buffer = []
stable_char = None
word_buffer = ""
sentence = ""
expected_features = 42
last_registered_time = time.time()
registration_delay = 1.5

# -----------------------------
# Tkinter GUI
# -----------------------------
root = tk.Tk()
root.title('Sign Language to Speech (PyTorch)')
root.geometry('1300x650')
root.configure(bg='#2c2f33')
root.resizable(False, False)

current_alphabet = StringVar(value='N/A')
current_word = StringVar(value='N/A')
current_sentence = StringVar(value='N/A')
is_paused = StringVar(value='False')

# Layout
video_frame = Frame(root, bg='#2c2f33', bd=5, relief='solid', width=500, height=400)
video_frame.grid(row=1, column=0, rowspan=3, padx=20, pady=20)
video_frame.grid_propagate(False)
video_label = tk.Label(video_frame)
video_label.pack(expand=True)

content_frame = Frame(root, bg='#2c2f33')
content_frame.grid(row=1, column=1, sticky='n', padx=(20, 40), pady=(60,20))

button_frame = Frame(root, bg='#2c2f33')
button_frame.grid(row=3, column=1, pady=(10,20), padx=(10,20), sticky='n')

Label(content_frame, text='Current Alphabet:', font=('Arial', 20), fg='#ffffff', bg='#2c2f33').pack(anchor='w', pady=(0,10))
Label(content_frame, textvariable=current_alphabet, font=('Arial', 24, 'bold'), fg='#1abc9c', bg='#2c2f33').pack(anchor='center')
Label(content_frame, text='Current Word:', font=('Arial', 20), fg='#ffffff', bg='#2c2f33').pack(anchor='w', pady=(20,10))
Label(content_frame, textvariable=current_word, font=('Arial', 20), fg='#f39c12', bg='#2c2f33', wraplength=500, justify='left').pack(anchor='center')
Label(content_frame, text='Current Sentence:', font=('Arial', 20), fg='#ffffff', bg='#2c2f33').pack(anchor='w', pady=(20,10))
Label(content_frame, textvariable=current_sentence, font=('Arial', 20), fg='#9b59b6', bg='#2c2f33', wraplength=500, justify='left').pack(anchor='center')

# Buttons

def reset_sentence():
    global word_buffer, sentence
    word_buffer = ""
    sentence = ""
    current_word.set('N/A')
    current_sentence.set('N/A')
    current_alphabet.set('N/A')


def toggle_pause():
    if is_paused.get() == 'False':
        is_paused.set('True')
        pause_button.config(text='Play')
    else:
        is_paused.set('False')
        pause_button.config(text='Pause')

Button(button_frame, text='Reset Sentence', font=('Arial', 16), command=reset_sentence, bg='#e74c3c', fg='#ffffff', relief='flat', height=2, width=14).grid(row=0, column=0, padx=10)
pause_button = Button(button_frame, text='Pause', font=('Arial', 16), command=toggle_pause, bg='#3498db', fg='#ffffff', relief='flat', height=2, width=12)
pause_button.grid(row=0, column=1, padx=10)
speak_button = Button(button_frame, text='Speak Sentence', font=('Arial', 16), command=lambda: speak_text(current_sentence.get()), bg='#27ae60', fg='#ffffff', relief='flat', height=2, width=14)
speak_button.grid(row=0, column=2, padx=10)

# -----------------------------
# Video capture
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 400)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)


def preprocess_landmarks(landmarks):
    # landmarks: list of 21 (x,y) normalized values
    x_list = [l[0] for l in landmarks]
    y_list = [l[1] for l in landmarks]
    data = []
    min_x = min(x_list)
    min_y = min(y_list)
    for x, y in zip(x_list, y_list):
        data.append(x - min_x)
        data.append(y - min_y)
    # ensure length
    if len(data) < expected_features:
        data.extend([0] * (expected_features - len(data)))
    elif len(data) > expected_features:
        data = data[:expected_features]
    return np.array(data, dtype=np.float32)


def predict_from_landmarks(data_aux):
    # convert to image-like tensor expected by ResNet18
    # simplest approach: tile the 42-d vector into a 3xHxW tensor
    # create a small gray image by reshaping to (6,7) => 42
    arr = data_aux.reshape(6, 7)
    # normalize to 0-1
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    # resize to IMG_SIZE x IMG_SIZE using cv2
    arr_img = (arr * 255).astype(np.uint8)
    arr_resized = cv2.resize(arr_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    # make 3 channels
    img3 = np.stack([arr_resized, arr_resized, arr_resized], axis=2)
    img3 = img3.astype(np.float32) / 255.0
    img3 = (img3 - 0.5) / 0.5
    tensor = torch.from_numpy(img3.transpose(2,0,1)).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return label_map[pred]


# -----------------------------
# Frame processing loop
# -----------------------------

def process_frame():
    global stabilization_buffer, stable_char, word_buffer, sentence, last_registered_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, process_frame)
        return

    if is_paused.get() == 'True':
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = img_tk
        video_label.configure(image=img_tk)
        root.after(10, process_frame)
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list = []
            y_list = []
            landmarks = []
            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                landmarks.append((lm.x, lm.y))

            data_aux = preprocess_landmarks(landmarks)
            predicted_character = predict_from_landmarks(data_aux)

            stabilization_buffer.append(predicted_character)
            if len(stabilization_buffer) > 30:
                stabilization_buffer.pop(0)

            if stabilization_buffer.count(predicted_character) > 25:
                current_time = time.time()
                if current_time - last_registered_time > registration_delay:
                    stable_char = predicted_character
                    last_registered_time = current_time
                    current_alphabet.set(stable_char)

                    if stable_char == ' ' or stable_char.lower() == 'space':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + ' '
                            current_sentence.set(sentence.strip())
                        word_buffer = ''
                        current_word.set('N/A')
                    elif stable_char == '.' or stable_char == 'period' or stable_char == 'dot':
                        if word_buffer.strip():
                            speak_text(word_buffer)
                            sentence += word_buffer + '.'
                            current_sentence.set(sentence.strip())
                        word_buffer = ''
                        current_word.set('N/A')
                    else:
                        word_buffer += stable_char
                        current_word.set(word_buffer)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing_styles.get_default_hand_landmarks_style(),
                                      mp_drawing_styles.get_default_hand_connections_style())

    cv2.putText(frame, f"Alphabet: {current_alphabet.get()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    root.after(10, process_frame)


if __name__ == '__main__':
    process_frame()
    root.mainloop()


# ======================================
# File: requirements.txt
# ======================================
# List of packages to include in your report
opencv-python
mediapipe
torch
torchvision
numpy
pyttsx3
pillow
matplotlib

