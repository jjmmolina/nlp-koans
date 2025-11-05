# 📊 Verificador de Progreso - NLP Koans

# Script para verificar el progreso en los koans
# Ejecuta: python check_progress.py

Write-Host "🧠 NLP Koans - Verificador de Progreso`n" -ForegroundColor Cyan

$koans = @(
    "01_tokenization",
    "02_stemming_lemmatization",
    "03_pos_tagging",
    "04_ner",
    "05_text_classification",
    "06_sentiment_analysis",
    "07_word_embeddings",
    "08_transformers",
    "09_language_models"
)

$total = 0
$passed = 0

foreach ($koan in $koans) {
    Write-Host "Testing $koan..." -NoNewline
    
    $result = pytest "koans/$koan/test_*.py" -q --tb=no 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅ PASSED" -ForegroundColor Green
        $passed++
    } else {
        Write-Host " ❌ FAILED" -ForegroundColor Red
    }
    
    $total++
}

Write-Host "`n📊 Resumen:" -ForegroundColor Cyan
Write-Host "Koans completados: $passed/$total" -ForegroundColor Yellow
$percentage = [math]::Round(($passed / $total) * 100, 2)
Write-Host "Progreso: $percentage%" -ForegroundColor Yellow

if ($passed -eq $total) {
    Write-Host "`n🎉 ¡Felicidades! Has completado todos los NLP Koans! 🎓" -ForegroundColor Green
}
