Modélisation Project Finance & Analyse de Risque (Python)
Ce projet est un exercice personnel réalisé dans le cadre de ma préparation aux métiers des Financements Structurés.

Si Excel reste l'outil de référence pour la modélisation financière, j'ai voulu utiliser Python pour aller plus loin sur deux aspects précis : l'automatisation des calculs et l'analyse statistique du risque (Monte Carlo).

L'objectif était de modéliser le financement d'une plateforme pétrolière offshore et de tester la robustesse de la dette face à des conditions de marché difficiles.

Structure du projet
J'ai séparé le projet en deux scripts pour distinguer la modélisation comptable de l'analyse de risque.

1. Le modèle de base Le fichier 1 
model_basique_hypothèses.py contient la mécanique financière classique. Il reproduit la logique d'un fichier Excel :

Calcul des revenus, des Opex et de l'impôt.

Structuration de la dette (calcul des intérêts et du profil de remboursement).

Détermination des flux disponibles pour le service de la dette (CFADS) et pour l'actionnaire.

Export des résultats vers un fichier Excel formaté et génération d'un tableau de bord visuel.

2. La simulation de risque (Monte Carlo) Le fichier 2 model_montecarlo.py sert à stresser le modèle. Au lieu de se baser sur un prix unique du pétrole, ce script lance 1000 simulations aléatoires pour voir comment le projet réagit à la volatilité des prix.

Résultats et Analyse
Pour vérifier la viabilité du projet, j'ai calibré la simulation sur un scénario bancaire conservateur avec un prix moyen du baril à 45$, bien en dessous des cours actuels.

Les résultats montrent que même avec ce prix dégradé :

Le TRI moyen reste aux alentours de 21%, ce qui est cohérent pour rémunérer le risque actionnaire sur ce type d'actif.

La probabilité que le projet atteigne son seuil de rentabilité (Hurdle Rate de 10%) reste supérieure à 80%.

Le graphique de distribution généré par le script (disponible dans le dossier output) permet de visualiser clairement la "queue de risque", c'est-à-dire les scénarios minoritaires où l'investisseur pourrait perdre du capital.

Utilisation
Le code a été écrit en Python 3 et utilise principalement les librairies Pandas (pour les flux) et NumPy dont numpy_financial !!! (pour les calculs financiers).

Pour lancer l'analyse de risque :

Installer les dépendances (pandas, numpy,numpy_financial, matplotlib, openpyxl).

Exécuter le script modèle_montecarlo.py.
