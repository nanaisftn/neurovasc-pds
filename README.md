# Neurovasc-PDS

## Description du projet
Ce projet vise à constituer une table de données du profil et du parcours clinique de 529 patients atteints d’hémorragie sous-arachnoïdienne (HSA).  
Il combine l’exploitation des données structurées et textuelles issues de l’entrepôt de données de Nantes.  

## Contenu du dépôt
- `script_BDD_HSA.py` : Script Python d'extraction et de traitement des données  
- `expressions_regulieres/` : Expressions régulières utilisées pour extraire les informations des comptes rendus  
- `Dictionnaire.xlsx` : Dictionnaire des variables extraites par le script  
- `README.md` : Ce fichier

## Prérequis
- Python 3.9
- Librairies Python : medkit, pandas, numpy, oracledb, os, yaml, re
- Accès à l’entrepôt de données de Nantes pour l’exécution du script

## Fonctionnalités principales du script
- Fusion et sélection des parcours pertinents pour l’étude  
- Extraction de données structurées (actes, traitements, résultats de laboratoire et mesures anthropométriques)  
- Extraction de données textuelles via :
  - Un Transformer entraîné en local (Gavroche)  
  - Une pipeline Medkit s'appuyant sur des expressions régulières
