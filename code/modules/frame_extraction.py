from absl import app, flags
import os
import cv2
import sys

# Ajouter le chemin du module
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'modules'))
sys.path.append(module_path)

from pooling import max_pooling


def get_output_folder_video(input_path):
    base_name = os.path.basename(input_path)
    name, _ = os.path.splitext(base_name)
    return os.path.join(os.getcwd(), os.path.dirname(input_path), f"{name}/")

def get_output_folder_dossier(input_path, suffix):
    base_name = os.path.basename(os.path.normpath(input_path))
    return os.path.join(os.path.dirname(input_path), f"{base_name}_{suffix}")


def create_folder(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
            print(f"Dossier crée : {path}")
        except Exception as e:
            print(f"Erreur lors de la création du dossier {path} : {e}")
            return


def extract_frames_from_video(video_path, output_folder, num_frames=11):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        print("Erreur : Impossible d'extraire 11 frames de la vidéo.")
        return

    create_folder(output_folder)

    for i in range(1, num_frames + 1):
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire la frame.")
            break
        frame_filename = os.path.join(output_folder, f"frame_{i}.png")
        cv2.imwrite(frame_filename, frame)
    
    cap.release()
    print(f"{num_frames} frames extraites et sauvegardées dans le dossier {output_folder}.")

def apply_pooling(input_folder, taille):
    # Lister tous les fichiers dans le dossier d'entrée
    img_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    window = (taille, taille)
    stride = (taille, taille)

    output_folder = get_output_folder_dossier(input_folder, f"pool_{taille}")
    create_folder(output_folder)

    
    for ind, img in enumerate(img_files):
        input_path = os.path.join(input_folder, img)
        output_path = os.path.join(output_folder,f"poolx{taille}_{ind}.png")

        try :
            max_pooling(input_path,output_path,window,stride)
            
        except Exception as e:
            print(f"Erreurs lors du traitement de l'image {input_path} : {e}")
        



