from PIL import Image
import matplotlib.pyplot as plt
import kaggle
import zipfile
import os
import cv2
import shutil
import re
import numpy as np
import yaml

#Funktion zum Entpacken und Importieren des Datensatzes
def datensatz_entpacken_importieren():
    ziel_verzeichnis = './data'
    datensatz = 'valentynsichkar/yolo-v5-format-of-the-traffic-signs-dataset' 
    print(f"Datensatz {datensatz} wird heruntergeladen....")
    kaggle.api.dataset_download_files(datensatz, path='./', unzip=False)
    zip_pfad = './yolo-v5-format-of-the-traffic-signs-dataset.zip'
    with zipfile.ZipFile(zip_pfad, 'r') as zip_ref:
        print("Verzeichnis 'ts43classes' wird entpackt.... ")
        for member in zip_ref.namelist():
            if 'ts43classes/' in member:
                zip_ref.extract(member, './')
    if os.path.exists('./ts_yolo_v5_format/ts43classes'):
        shutil.move('./ts_yolo_v5_format/ts43classes', ziel_verzeichnis)
        print(f"Verzeichnis 'ts43classes' erfolgreich nach {ziel_verzeichnis} verschoben.")
    os.remove(zip_pfad)
    shutil.rmtree('./ts_yolo_v5_format', ignore_errors=True)
    print("Import abgeschlossen.")

#Funktion zum Einlesen des Pfades und den Dateinamen
def get_filenames(filepath):
    return [os.path.join(filepath, filename) for filename in os.listdir(filepath)]

#Funktion zum Einlesen der Bilder 
def read_images(filenames, height=None, width=None):
    images = [Image.open(filename) for filename in filenames]
    if (not height is None) and (not width is None):
        images = [img.resize((width, height)) for img in images]
    return images

#Funktion zum konvertieren eine Liste von Bildern in einem NumPy-Array
def images_to_array(images):
    return np.asarray([np.asarray(img) for img in images])

#Funktion zum Testen der Annotationen auf die Bilder
def plot_images_with_annotations(image_names, annotations_dir, input_shape):
    fig = plt.figure(figsize=(8, 8))
    rows, columns = 3, 4
    num_images_to_display = min(8, len(image_names))

    for i in range(num_images_to_display):
        img_path = image_names[i]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = input_shape["height"], input_shape["width"]
        
        # Skaliere das Bild auf die festgelegte Größe
        img = cv2.resize(img, (img_width, img_height))
        
        # Annotationen laden
        annotation_path = os.path.join(annotations_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    # YOLO-Format: <class_id> <x_center> <y_center> <width> <height>
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])

                    # Umrechnung der Normalisierten Werte auf Bildkoordinaten
                    x_center *= img_width
                    y_center *= img_height
                    width *= img_width
                    height *= img_height

                    # Berechne obere linke Ecke und untere rechte Ecke
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    # Zeichne die Bounding Box und die Klassen-ID
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blaue Box
                    cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Bild im Plot anzeigen
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)

    plt.tight_layout()
    plt.show()

#Funktion zum Erstellen einer Datensatz-Konfigurations-Datei im .yaml-Format in dem Projektverzeichnis
def write_yaml_to_file(py_obj, filename):
    save_path = 'G:/Meine Ablage/projekt-cnn-yolo-echtzeiterkennung-verkehszeichen-gtsrb/data'
    full_path = os.path.join(save_path, f'{filename}.yaml')
    with open(full_path, 'w') as f:
        yaml.dump(py_obj, f, sort_keys=False)
    print(f'Datei wurde erfolgreich erstellt in: {full_path}')