import os
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Lade die Daten aus den npy-Dateien
def load_data(features_folder):
    X = []
    y = []
    for label in ['0', '1']:  # Ordnernamen, die die Labels bestimmen
        folder_path = os.path.join(features_folder, label)
        for file in pathlib.Path(folder_path).rglob("*.npy"):
            # Lese die npy-Datei und flache sie ab
            tmp1 = np.load(file)
            X.append(tmp1.flatten())
            y.append(label)  # Verwende den Ordnernamen als Label (0 oder 1)
    X = np.array(X)
    y = np.array(y)
    return X, y

# Hauptfunktion zum Trainieren und Auswerten des SVM-Modells
def train_and_evaluate_svm(features_folder):
    # Lade die Daten
    X, y = load_data(features_folder)
    
    # Teile die Daten in Trainings- und Testsets auf (80% Training, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Definiere das SVM-Modell mit den gewünschten Parametern
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    
    # 10-fache Kreuzvalidierung
    cv_scores = cross_val_score(svm_model, X_train, y_train, cv=10, scoring='accuracy')
    
    print("Durchschnittliche Genauigkeit aus 10-facher Kreuzvalidierung: ", np.mean(cv_scores))
    print("Kreuzvalidierungsgenauigkeiten pro Fold: ", cv_scores)
    
    # Trainiere das Modell auf dem Trainingsdatensatz
    svm_model.fit(X_train, y_train)
    
    # Mache Vorhersagen auf dem Testdatensatz
    y_pred = svm_model.predict(X_test)
    
    # Auswertung der Vorhersagen
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Verwirrungsmatrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Visualisierung der Verwirrungsmatrix
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(set(y)))
    plt.xticks(tick_marks, ['Non-Exoplanet', 'Exoplanet'])
    plt.yticks(tick_marks, ['Non-Exoplanet', 'Exoplanet'])
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

# Verzeichnis der Features
features_folder = './features/mfcc'

# Trainiere und bewerte das Modell
train_and_evaluate_svm(features_folder)

