from PIL import Image
import kaggle
import zipfile
import os
import cv2
import shutil
import re
import numpy as np

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


#Funktion zum erstellen des dataset.yaml-Datei aus der VOC.yaml-Datei