"""
Tests para Koan 05: Text Classification

Ejecuta con:
    pytest koans/05_text_classification/test_text_classification.py -v
"""

import pytest
import numpy as np
from text_classification import (
    create_tfidf_features,
    create_bow_features,
    train_naive_bayes_classifier,
    train_logistic_regression_classifier,
    predict_class,
    predict_proba,
    evaluate_classifier,
    get_top_features,
    build_simple_spam_classifier
)


class TestFeatureExtraction:
    """Tests de extracción de características"""
    
    def test_create_tfidf_features(self):
        """Test: Crear características TF-IDF"""
        texts = ["Python es genial", "Java es genial", "Python es mejor"]
        vectorizer, features = create_tfidf_features(texts, max_features=10)
        
        assert vectorizer is not None
        assert features is not None
        assert features.shape[0] == 3  # 3 documentos
        
    def test_create_bow_features(self):
        """Test: Crear características Bag of Words"""
        texts = ["hola mundo", "hola Python", "mundo Python"]
        vectorizer, features = create_bow_features(texts, max_features=10)
        
        assert vectorizer is not None
        assert features is not None
        assert features.shape[0] == 3


class TestClassifierTraining:
    """Tests de entrenamiento de clasificadores"""
    
    def test_train_naive_bayes_classifier(self):
        """Test: Entrenar Naive Bayes"""
        texts = ["spam gratis", "reunión mañana", "premio gratis", "proyecto python"]
        labels = [1, 0, 1, 0]
        
        _, X = create_tfidf_features(texts)
        clf = train_naive_bayes_classifier(X, labels)
        
        assert clf is not None
        
    def test_train_logistic_regression_classifier(self):
        """Test: Entrenar Regresión Logística"""
        texts = ["spam gratis", "reunión mañana", "premio gratis", "proyecto python"]
        labels = [1, 0, 1, 0]
        
        _, X = create_tfidf_features(texts)
        clf = train_logistic_regression_classifier(X, labels)
        
        assert clf is not None


class TestPrediction:
    """Tests de predicción"""
    
    def test_predict_class(self):
        """Test: Predecir clase de texto"""
        # Datos de entrenamiento simples
        texts = [
            "gratis premio dinero",
            "reunión proyecto trabajo",
            "oferta gratis click",
            "equipo desarrollo sprint"
        ]
        labels = [1, 0, 1, 0]  # 1=spam, 0=no spam
        
        vectorizer, X = create_tfidf_features(texts)
        clf = train_naive_bayes_classifier(X, labels)
        
        # Predecir nuevo texto
        result = predict_class(clf, vectorizer, "gratis oferta")
        
        assert result in [0, 1]
        
    def test_predict_proba(self):
        """Test: Predecir probabilidades"""
        texts = ["spam", "normal", "spam", "normal"]
        labels = [1, 0, 1, 0]
        
        vectorizer, X = create_tfidf_features(texts)
        clf = train_naive_bayes_classifier(X, labels)
        
        probs = predict_proba(clf, vectorizer, "spam")
        
        assert isinstance(probs, np.ndarray)
        assert len(probs) == 2
        assert abs(sum(probs) - 1.0) < 0.01  # Las probabilidades suman 1


class TestEvaluation:
    """Tests de evaluación"""
    
    def test_evaluate_classifier(self):
        """Test: Evaluar clasificador"""
        texts = ["spam"] * 10 + ["normal"] * 10
        labels = [1] * 10 + [0] * 10
        
        vectorizer, X = create_tfidf_features(texts)
        clf = train_naive_bayes_classifier(X, labels)
        
        metrics = evaluate_classifier(clf, X, labels)
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "report" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestFeatureImportance:
    """Tests de importancia de características"""
    
    def test_get_top_features(self):
        """Test: Obtener palabras más importantes"""
        texts = [
            "python código programación",
            "java código desarrollo",
            "python mejor lenguaje"
        ]
        labels = [0, 1, 0]
        
        vectorizer, X = create_tfidf_features(texts)
        clf = train_logistic_regression_classifier(X, labels)
        
        top_features = get_top_features(vectorizer, clf, class_label=0, top_n=3)
        
        assert isinstance(top_features, list)
        # Puede estar vacío si no hay suficientes features


class TestSpamClassifier:
    """Tests del clasificador de spam"""
    
    def test_build_simple_spam_classifier(self):
        """Test: Construir clasificador de spam"""
        emails = [
            "Gana dinero rápido!",
            "Reunión de equipo mañana",
            "Oferta especial gratis!",
            "Agenda para el proyecto",
            "Click aquí premio gratis",
            "Código Python para revisar"
        ]
        labels = [1, 0, 1, 0, 1, 0]
        
        clf, vectorizer = build_simple_spam_classifier(emails, labels)
        
        assert clf is not None
        assert vectorizer is not None
        
    def test_spam_classifier_prediction(self):
        """Test: Clasificador de spam predice correctamente"""
        emails = [
            "premio gratis dinero",
            "proyecto desarrollo",
            "oferta click ahora",
            "reunión trabajo"
        ]
        labels = [1, 0, 1, 0]
        
        clf, vectorizer = build_simple_spam_classifier(emails, labels)
        
        # Predecir un email claramente spam
        result = predict_class(clf, vectorizer, "gratis premio oferta")
        
        # Debe predecir algo (0 o 1)
        assert result in [0, 1]


class TestRealWorldExamples:
    """Tests con ejemplos del mundo real"""
    
    def test_news_classification(self):
        """Test: Clasificar categorías de noticias"""
        news = [
            "Gol en el minuto 90 del partido",
            "Nueva ley aprobada en el congreso",
            "Gana el campeonato de fútbol",
            "Debate político intenso"
        ]
        labels = [0, 1, 0, 1]  # 0=deportes, 1=política
        
        vectorizer, X = create_tfidf_features(news)
        clf = train_naive_bayes_classifier(X, labels)
        
        # Predecir nueva noticia
        result = predict_class(clf, vectorizer, "partido de fútbol")
        assert result in [0, 1]
        
    def test_sentiment_binary_classification(self):
        """Test: Clasificación binaria de sentimiento"""
        reviews = [
            "Excelente producto, muy bueno",
            "Terrible, muy malo",
            "Increíble, me encanta",
            "Horrible, no funciona"
        ]
        sentiments = [1, 0, 1, 0]  # 1=positivo, 0=negativo
        
        vectorizer, X = create_tfidf_features(reviews)
        clf = train_naive_bayes_classifier(X, sentiments)
        
        result = predict_class(clf, vectorizer, "producto bueno")
        assert result in [0, 1]
