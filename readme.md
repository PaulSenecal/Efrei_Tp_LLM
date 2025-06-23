# Assistant Culinaire Intelligent
## Développement d'un LLM avec Deep Learning et NLP

---

## Objectif du Projet

Créer un chatbot intelligent spécialisé dans le domaine culinaire utilisant :
- **Deep Learning** : Réseau de neurones LSTM
- **NLP** : Traitement du langage naturel
- **Application pratique** : Assistant conversationnel

---

## Architecture du Système

### Composants Principaux

1. **TextProcessor**
   - Nettoyage et normalisation du texte
   - Construction du vocabulaire
   - Conversion texte ↔ séquences numériques

2. **CulinaryLanguageModel**
   - Architecture LSTM bidirectionnelle
   - Génération de texte avec température
   - Prédiction du mot suivant

3. **CulinaryChatbot**
   - Détection d'intention
   - Génération de réponses contextuelles
   - Interface conversationnelle

---

## Architecture du Modèle

### Couches du Réseau de Neurones

```
Embedding (128 dimensions)
    ↓
LSTM (256 unités) + Dropout (0.3)
    ↓
LSTM (256 unités) + Dropout (0.3)
    ↓
Dense (256 unités, ReLU)
    ↓
Dense (taille_vocabulaire, Softmax)
```

### Paramètres Clés
- Longueur de séquence : 50 mots
- Dimension d'embedding : 128
- Unités LSTM : 256
- Taux de dropout : 30%

---

## Processus d'Entraînement

### Pipeline de Données

1. **Collecte** : Dataset culinaire (recettes, techniques, ingrédients)
2. **Prétraitement** : Nettoyage, tokenisation, padding
3. **Augmentation** : Division des textes longs pour plus de données
4. **Séquençage** : Création de paires (contexte, mot_suivant)

### Hyperparamètres
- Époques : 50
- Batch size : 64
- Optimiseur : Adam
- Fonction de perte : Sparse Categorical Crossentropy

---

## Génération de Texte

### Méthode de Génération

1. **Seed Text** : Phrase d'amorce selon l'intention détectée
2. **Prédiction** : Le modèle prédit le prochain mot
3. **Échantillonnage** : Sélection probabiliste avec température
4. **Itération** : Répétition jusqu'à ponctuation ou limite

### Paramètre de Température
- **Basse (0.5)** : Réponses plus prévisibles
- **Moyenne (0.8)** : Équilibre créativité/cohérence
- **Haute (1.2)** : Plus de variété, risque d'incohérence

---

## Détection d'Intention

### Catégories Identifiées

- **Recette** : Instructions de préparation
- **Conseil** : Astuces culinaires
- **Ingrédient** : Information sur les produits
- **Cuisson** : Techniques et températures
- **Dessert** : Pâtisserie et douceurs
- **Plat** : Plats principaux

### Phrases d'Amorce Contextuelles
Chaque intention a sa phrase de départ spécifique pour guider la génération

---

## Démonstration

### Exemples d'Interactions

**Utilisateur** : "Comment faire une recette?"
**Assistant** : "Pour préparer cette recette, commencez par rassembler tous vos ingrédients..."

**Utilisateur** : "Conseil pour la cuisson"
**Assistant** : "Mon conseil culinaire est de toujours préchauffer votre four..."

**Utilisateur** : "Parlez-moi du safran"
**Assistant** : "Concernant cet ingrédient, le safran est une épice précieuse..."

---

## Points Forts du Projet

### Aspects Techniques
- **Architecture modulaire** : Séparation claire des responsabilités
- **Code épuré** : Noms explicites, pas de commentaires superflus
- **Extensibilité** : Facile d'ajouter des intentions ou données

### Aspects Pédagogiques
- **Compréhension** : Architecture simple mais complète
- **Reproductibilité** : Dataset généré inclus
- **Évolutivité** : Base solide pour améliorations

---

## Améliorations Possibles

### Court Terme
- Augmentation du dataset avec vraies recettes
- Fine-tuning des hyperparamètres
- Ajout de métriques de qualité (BLEU, perplexité)

### Long Terme
- Migration vers Transformer (GPT-like)
- Intégration d'une base de connaissances
- Interface web avec Flask/FastAPI
- Multilingue (français, anglais, italien)

---

## Technologies Utilisées

### Frameworks et Bibliothèques
- **TensorFlow/Keras** : Construction du modèle
- **NumPy** : Manipulation des données
- **Python 3.8+** : Langage principal

### Concepts Appliqués
- **LSTM** : Mémoire à long terme
- **Embeddings** : Représentation vectorielle des mots
- **Dropout** : Régularisation
- **Softmax** : Probabilités de sortie

---

## Conclusion

### Réalisations
- ✓ Modèle de langage fonctionnel
- ✓ Application chatbot interactive
- ✓ Architecture clean code
- ✓ Pipeline d'entraînement complet

### Apprentissages Clés
- Importance du prétraitement des données
- Balance entre complexité et performance
- Rôle de la température dans la génération
- Architecture modulaire pour la maintenabilité

---

## Questions ?

### Ressources
- Code source complet disponible
- Documentation des classes
- Dataset extensible

### Contact
[Vos informations de contact]

**Merci de votre attention !**