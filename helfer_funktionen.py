from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import kaggle
import zipfile
import os
import cv2
import shutil
import numpy as np
import yaml

# Funktion zum Entpacken und Importieren des Datensatzes
def datensatz_entpacken_importieren():
    ziel_verzeichnis = './data'
    datensatz = 'valentynsichkar/yolo-v5-format-of-the-traffic-signs-dataset'
    kaggle.api.dataset_download_files(datensatz, path='./', unzip=False)
    with zipfile.ZipFile('yolo-v5-format-of-the-traffic-signs-dataset.zip', 'r') as zip_ref:
        zip_ref.extractall('./')
    if os.path.exists('./ts_yolo_v5_format/ts43classes'):
        shutil.move('./ts_yolo_v5_format/ts43classes', ziel_verzeichnis)
    os.remove('yolo-v5-format-of-the-traffic-signs-dataset.zip')
    shutil.rmtree('./ts_yolo_v5_format', ignore_errors=True)

# Funktion zum Einlesen der Dateinamen
def get_filenames(filepath):
    return [os.path.join(filepath, f) for f in os.listdir(filepath)]

# Funktion zum Einlesen der Bilder
def read_images(filenames, height=None, width=None):
    images = []
    for filename in filenames:
        img = Image.open(filename)
        if height and width:
            img = img.resize((width, height))
        images.append(img)
    return images

# Funktion zum Konvertieren einer Liste von Bildern in ein NumPy-Array
def images_to_array(images):
    return np.array([np.array(img) for img in images])

# Funktion zum Plotten von Bildern mit Annotationen
def plot_images_with_annotations(image_names, annotations_dir, input_shape):
    fig = plt.figure(figsize=(8, 8))
    for i, img_path in enumerate(image_names[:8]):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape["width"], input_shape["height"]))
        # Annotationen laden
        annotation_path = os.path.join(annotations_dir, os.path.splitext(os.path.basename(img_path))[0] + '.txt')
        if os.path.exists(annotation_path):
            with open(annotation_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:])
                    x_center *= img.shape[1]
                    y_center *= img.shape[0]
                    width *= img.shape[1]
                    height *= img.shape[0]
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, str(class_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        fig.add_subplot(3, 4, i + 1)
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

# Funktion zum Erstellen einer YAML-Datei
def write_yaml_to_file(py_obj, filename):
    with open(f'{filename}.yaml', 'w') as f:
        yaml.dump(py_obj, f)

# Funktion zum Plotten der Klassenverteilung
def plot_class_distribution(label_path, class_names, output_path):
    class_counts = Counter()
    for label_file in os.listdir(label_path):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_path, label_file), 'r') as file:
                for line in file:
                    class_id = int(line.split()[0])
                    class_counts[class_id] += 1
    classes = [class_names[i] for i in range(len(class_names))]
    counts = [class_counts.get(i, 0) for i in range(len(class_names))]
    plt.figure(figsize=(15, 20))
    bars = plt.barh(classes, counts, color='skyblue')
    plt.xlabel('Anzahl der Bilder')
    plt.ylabel('Klassen')
    plt.title('Anzahl der Bilder pro Klasse')
    # Anzahl der Bilder an den Balken anzeigen
    for bar, count in zip(bars, counts):
        plt.text(count, bar.get_y() + bar.get_height()/2, str(count), va='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'labels_.jpg'), dpi=300)
    plt.show()

