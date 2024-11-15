from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import os
import cv2
import shutil
import numpy as np
import yaml

#Funktion zum Entpacken und Importieren des Datensatzes
def copy_images_and_convert_annotations(df, split_type, base_image_dir):
    copied_files = 0  #Zähler für kopierte Dateien
    for _, row in df.iterrows():
        class_id = row['ClassId']
        img_filename = row['Path'] if 'Path' in row else row['Filename']
        #Überprüfen, ob der Pfad in der CSV-Datei bereits "Train/" oder "Test/" enthält
        if split_type == "test":
            src_img_path = os.path.join(base_image_dir, img_filename.split('/')[-1])  # nur den Dateinamen verwenden
        else:
            #Zugriff auf die Unterordner für den Trainings- und Validierungssplit
            src_img_path = os.path.join(base_image_dir, str(class_id), img_filename.split('/')[-1])
        #Überprüfen, ob die Datei existiert
        if os.path.exists(src_img_path):
            #Zielpfad für das Bild
            dest_img_path = os.path.join(f"dataset/images/{split_type}", os.path.basename(img_filename))
            #Kopiere das Bild
            shutil.copyfile(src_img_path, dest_img_path)
            copied_files += 1
            #Konvertiere Annotationen ins YOLO-Format
            label_filename = os.path.basename(img_filename).replace('.ppm', '.txt').replace('.png', '.txt')
            with open(f"dataset/labels/{split_type}/{label_filename}", 'w') as f:
                x_center = (row['Roi.X1'] + row['Roi.X2']) / 2 / row['Width'] #Im Zähler wird der Mittelpunkt der Bounding Box berechnet, im Nenner wird dann der Mittelpnkt normalisiert
                y_center = (row['Roi.Y1'] + row['Roi.Y2']) / 2 / row['Height'] #Im Zähler wird der Mittelpunkt der Bounding Box berechnet, im Nenner wird dann Normalisiert
                width = (row['Roi.X2'] - row['Roi.X1']) / row['Width'] #Im Zähler wird die Breite der Bounding Box berechnet, im Nenner wird dann Normalisiert
                height = (row['Roi.Y2'] - row['Roi.Y1']) / row['Height'] #Im Zähler wird die Höhe der Bounding Box berechnet, im Nenner wird dann Normalisiert
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        else:
            print(f"Datei nicht gefunden: {src_img_path}")
    print(f"{copied_files} Dateien wurden für den Split '{split_type}' kopiert.")


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

