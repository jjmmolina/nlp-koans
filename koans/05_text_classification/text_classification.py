"""
Koan 05: Text Classification - Clasificación de Textos

La clasificación de textos asigna categorías a documentos.

Ejemplos:
- Spam vs No Spam
- Categorías de noticias (deportes, política, tecnología)
- Temas de documentos

Usaremos scikit-learn para clasificación tradicional con TF-IDF.
"""

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple
import numpy as np


def create_tfidf_features(texts: List[str], max_features: int = 100) -> Tuple:
    """
    Crea características TF-IDF a partir de textos.

    TF-IDF (Term Frequency - Inverse Document Frequency) representa
    la importancia de palabras en documentos.

    Ejemplo:
        >>> texts = ["Python es genial", "Java es genial", "Python es mejor"]
        >>> vectorizer, features = create_tfidf_features(texts)
        >>> features.shape
        (3, 100)

    Args:
        texts: Lista de textos
        max_features: Número máximo de características

    Returns:
        Tupla (vectorizer, features_matrix)
    """
    # TODO: Implementa creación de características TF-IDF
    # Pistas:
    # 1. Necesitas TfidfVectorizer (ya está importado)
    # 2. Crea una instancia con max_features
    # 3. Usa fit_transform para ajustar y transformar los textos
    # 4. Retorna tanto el vectorizer como la matriz de características
    pass


def create_bow_features(texts: List[str], max_features: int = 100) -> Tuple:
    """
    Crea características Bag of Words (BoW).

    BoW cuenta la frecuencia de cada palabra.

    Ejemplo:
        >>> texts = ["hola mundo", "hola Python"]
        >>> vectorizer, features = create_bow_features(texts)

    Args:
        texts: Lista de textos
        max_features: Número máximo de características

    Returns:
        Tupla (vectorizer, features_matrix)
    """
    # TODO: Implementa BoW con CountVectorizer
    # Pista: Similar a TF-IDF pero con CountVectorizer (ya importado)
    pass


def train_naive_bayes_classifier(X_train, y_train):
    """
    Entrena un clasificador Naive Bayes.

    Naive Bayes es rápido y funciona bien para clasificación de texto.

    Ejemplo:
        >>> X = [[0.1, 0.2], [0.3, 0.4]]
        >>> y = [0, 1]
        >>> clf = train_naive_bayes_classifier(X, y)

    Args:
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento

    Returns:
        Clasificador entrenado
    """
    # TODO: Implementa entrenamiento con MultinomialNB
    # Pista: Crea una instancia, llama a fit(), y retorna el clasificador
    pass


def train_logistic_regression_classifier(X_train, y_train):
    """
    Entrena un clasificador de Regresión Logística.

    Ejemplo:
        >>> X = [[0.1, 0.2], [0.3, 0.4]]
        >>> y = [0, 1]
        >>> clf = train_logistic_regression_classifier(X, y)

    Args:
        X_train: Características de entrenamiento
        y_train: Etiquetas de entrenamiento

    Returns:
        Clasificador entrenado
    """
    # TODO: Implementa con LogisticRegression (ya importado)
    pass


def predict_class(classifier, vectorizer, text: str) -> int:
    """
    Predice la clase de un nuevo texto.

    Ejemplo:
        >>> # Asumiendo classifier y vectorizer ya entrenados
        >>> predict_class(classifier, vectorizer, "Python es genial")
        1

    Args:
        classifier: Clasificador entrenado
        vectorizer: Vectorizador entrenado
        text: Texto a clasificar

    Returns:
        Clase predicha
    """
    # TODO: Implementa predicción
    # Pistas:
    # 1. Transforma el texto a características
    # 2. Usa el método predict del clasificador
    pass


def predict_proba(classifier, vectorizer, text: str) -> np.ndarray:
    """
    Predice las probabilidades de cada clase.

    Ejemplo:
        >>> predict_proba(classifier, vectorizer, "Python es genial")
        array([0.2, 0.8])  # 20% clase 0, 80% clase 1

    Args:
        classifier: Clasificador entrenado
        vectorizer: Vectorizador entrenado
        text: Texto a clasificar

    Returns:
        Array de probabilidades
    """
    # TODO: Usa predict_proba() en lugar de predict()
    pass


def evaluate_classifier(classifier, X_test, y_test) -> dict:
    """
    Evalúa el rendimiento de un clasificador.

    Ejemplo:
        >>> evaluate_classifier(clf, X_test, y_test)
        {'accuracy': 0.85, 'report': '...'}

    Args:
        classifier: Clasificador entrenado
        X_test: Características de prueba
        y_test: Etiquetas verdaderas

    Returns:
        Diccionario con métricas
    """
    # TODO: Implementa evaluación
    # Pistas:
    # 1. Haz predicciones
    # 2. Calcula accuracy (ya está importado accuracy_score)
    # 3. Genera classification_report (ya está importado)
    pass


def get_top_features(
    vectorizer, classifier, class_label: int, top_n: int = 10
) -> List[str]:
    """
    Obtiene las palabras más importantes para una clase.

    Ejemplo:
        >>> get_top_features(vectorizer, clf, class_label=1, top_n=5)
        ['python', 'genial', 'mejor', 'lenguaje', 'código']

    Args:
        vectorizer: Vectorizador entrenado
        classifier: Clasificador entrenado
        class_label: Clase a analizar
        top_n: Número de características a retornar

    Returns:
        Lista de palabras más importantes
    """
    # TODO: Implementa extracción de top features
    # Pistas:
    # 1. Obtén los coeficientes del clasificador (attr coef_)
    # 2. Obtén los nombres de las características del vectorizer
    # 3. Ordena por importancia y toma top_n
    pass


def build_simple_spam_classifier(emails: List[str], labels: List[int]) -> Tuple:
    """
    Construye un clasificador de spam simple.

    Ejemplo:
        >>> emails = ["Win money now!", "Meeting at 3pm", "Free prize!"]
        >>> labels = [1, 0, 1]  # 1=spam, 0=no spam
        >>> clf, vec = build_simple_spam_classifier(emails, labels)

    Args:
        emails: Lista de emails
        labels: Lista de etiquetas (1=spam, 0=no spam)

    Returns:
        Tupla (clasificador, vectorizador)
    """
    # TODO: Implementa clasificador completo
    # Combina las funciones anteriores para crear un pipeline completo
    pass
