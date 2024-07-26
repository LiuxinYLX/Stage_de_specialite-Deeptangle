# Deeptangle

## Installation

1. Créer un environnement virtuel avec Conda

   ```shell
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Installer les dépendances nécessaires 

   ```shell
   pip install -r requirements.txt
   ```

3. Installer le modèle et les fonctions auxiliaires

   ```shell
   pip install -e .
   ```

4. Télécharger les poids pré-entraînés

   ```shell
   wget https://sid.erda.dk/share_redirect/cEjIpG1yQl -O weights.zip
   ```

## Configurer l'environnement

1. Déplacez-vous dans le répertoire **deeptangle**
2. Saisir dans le terminal : 

   ```shell
   source venv/bin/activate
   ```



## Exécuter le code

3. Déplacez-vous dans le répertoire **deeptangle**
4. Saisir dans le terminal : 

   ```python
   python3 code/main.py --model=weights/ --input=images/Movie_2.avi --pooling=True
   ```


Voici les paramètres que vous pourrez modifier :

### Paramètres nécessaires

1. **--input** = L'adresse de la vidéo que vous voulez analyser

   Exemple : --input = images/Movie_2.avi 


2. **--output** = L'adresse du répertoire où vous voulez mettre le résultat.

   Exemple : --output = result/output.png
   

3. **--pooling** = True/False 

   > Si vous avez déjà testé une fois une vidéo et vous voulez juste modifier les paramètres pour obtenir un meilleur résultat, choisissez *False* pour réduire le temps d'exécution.
   > *True* fonctionne dans tous les cas. 

   Exemple : --pooling = False 


4. **--model** = Chemin vers les poids du modèle

   > Les poids sont essentiels pour un modèle. Dans notre cas, on utilise les poids pré-entraînés par les auteurs avec les données synthétiques.

   Exemple : --model = weights/

   


### Hyper-paramètres
> Vous pouvez également les modifier directement dans le fichier '*main.py*'

1. -**-score_threshold** = Seuil de score pour élaguer les mauvaises prédictions.

   Dans le cas où beaucoup de vers ne sont pas détectés, essayez de le baisser. Il prend des valeurs entre 0 et 1.

   Exemple : --score_threshold = 0.5
 

2. **--overlap_threshold** = Seuil de score de chevauchement pour supprimer les prédictions.

   Dans le cas où une seule cible a plusieurs prédictions, essayez de l'augmenter. Il prend des valeurs entre 0 et 1.

   Exemple : --overlap_threshold = 0.5

   


### Paramètres visant à filtrer les prédictions

> Nous les modifions rarement.

   1. hausdorff_threshold
   2. distance_threshold
   3. nb_threshold

