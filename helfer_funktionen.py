import zipfile
import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

# Funktion zum Entpacken von ZIP-Dateien
def unzip_file(zip_file, extract_to):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Funktion zum Resizing der Bilder
def resize_images(image_folder, target_size=(416, 416)):
    for root, dirs, files in os.walk(image_folder):
        for file in files:
            if file.endswith('.ppm'):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, target_size)
                cv2.imwrite(img_path, img_resized)  # Überschreibt das Bild mit der neuen Größe

# Funktion zur Konvertierung der Labels in das YOLO-Format
def convert_labels_to_yolo_format(csv_file, output_folder, img_width=416, img_height=416):
    os.makedirs(output_folder, exist_ok=True)
    df = pd.read_csv(csv_file, delimiter=';')
    for index, row in df.iterrows():
        class_id = row['ClassId']
        x_center = (row['Roi.X1'] + row['Roi.X2']) / 2.0 / img_width
        y_center = (row['Roi.Y1'] + row['Roi.Y2']) / 2.0 / img_height
        width = (row['Roi.X2'] - row['Roi.X1']) / img_width
        height = (row['Roi.Y2'] - row['Roi.Y1']) / img_height
        
        label_path = os.path.join(output_folder, f"{row['Filename'].replace('.ppm', '.txt')}")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Funktion zum Aufteilen der Trainingsdaten in Trainings- und Validierungsdaten
def split_train_val(image_folder, label_folder, val_size=0.2):
    images = []
    labels = []
    
    for file in os.listdir(image_folder):
        if file.endswith('.ppm'):
            images.append(os.path.join(image_folder, file))
            label_file = file.replace('.ppm', '.txt')
            labels.append(os.path.join(label_folder, label_file))
    
    # Aufteilen der Daten
    train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=val_size)
    
    # Ordner für die Daten erstellen
    os.makedirs('dataset/images/train', exist_ok=True)
    os.makedirs('dataset/images/val', exist_ok=True)
    os.makedirs('dataset/labels/train', exist_ok=True)
    os.makedirs('dataset/labels/val', exist_ok=True)

    # Kopieren der Trainingsdaten
    for img, lbl in zip(train_images, train_labels):
        shutil.copy(img, 'dataset/images/train')
        shutil.copy(lbl, 'dataset/labels/train')
    
    # Kopieren der Validierungsdaten
    for img, lbl in zip(val_images, val_labels):
        shutil.copy(img, 'dataset/images/val')
        shutil.copy(lbl, 'dataset/labels/val')
