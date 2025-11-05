#!/bin/bash
# Verificador de Progreso - NLP Koans
# Ejecuta: bash check_progress.sh

echo "🧠 NLP Koans - Verificador de Progreso"
echo ""

koans=(
    "01_tokenization"
    "02_stemming_lemmatization"
    "03_pos_tagging"
    "04_ner"
    "05_text_classification"
    "06_sentiment_analysis"
    "07_word_embeddings"
    "08_transformers"
    "09_language_models"
)

total=0
passed=0

for koan in "${koans[@]}"; do
    echo -n "Testing $koan... "
    
    if pytest "koans/$koan/test_*.py" -q --tb=no > /dev/null 2>&1; then
        echo "✅ PASSED"
        ((passed++))
    else
        echo "❌ FAILED"
    fi
    
    ((total++))
done

echo ""
echo "📊 Resumen:"
echo "Koans completados: $passed/$total"
percentage=$(awk "BEGIN {printf \"%.2f\", ($passed/$total)*100}")
echo "Progreso: $percentage%"

if [ $passed -eq $total ]; then
    echo ""
    echo "🎉 ¡Felicidades! Has completado todos los NLP Koans! 🎓"
fi
