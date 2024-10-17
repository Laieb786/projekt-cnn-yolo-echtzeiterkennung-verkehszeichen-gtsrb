import kaggle
import zipfile
import os
import shutil

#Funktion zum Entpacken und Importieren des Datensatzes
def datensatz_entpacken_importieren():
    ziel_verzeichnis = './ts43classes'
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

#Funktion zum erstellen des dataset.yaml-Datei