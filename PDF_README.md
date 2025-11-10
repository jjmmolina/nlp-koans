# 📄 Generación de PDF - Teoría Completa

Este directorio contiene un script para generar una versión HTML/PDF profesional del documento `THEORY_COMPLETE.md`.

## 🚀 Uso Rápido

```bash
# Generar HTML (listo para imprimir a PDF)
python generate_html.py
```

## 📖 Pasos para crear el PDF

1. **Genera el HTML:**
   ```bash
   python generate_html.py
   ```

2. **Abre en tu navegador:**
   - Windows: Doble clic en `NLP_Koans_Teoria_Completa.html`
   - Linux/Mac: `open NLP_Koans_Teoria_Completa.html`

3. **Imprime a PDF:**
   - Presiona `Ctrl+P` (Windows/Linux) o `Cmd+P` (Mac)
   - Selecciona "Guardar como PDF" como destino
   - **Configuración recomendada:**
     - Orientación: Vertical
     - Márgenes: Predeterminados
     - Escala: 100%
     - Color de fondo: Activado (para ver código con colores)

4. **Guarda:**
   - Nombra el archivo como desees
   - ¡Listo! Ahora tienes un PDF profesional de ~150-200 páginas

## ✨ Características del HTML/PDF

- 📚 **Portada profesional** con título y fecha
- 🎨 **Código con syntax highlighting** y bordes coloridos
- 📊 **Tablas estilizadas** con headers en gradiente
- 🔗 **TOC interactivo** (en HTML, navegación suave)
- 📖 **Tipografía optimizada** para lectura prolongada
- 🖨️ **Saltos de página inteligentes** (no parte bloques de código)
- ⚡ **Botón de impresión rápida** (solo visible en pantalla)

## 🎯 Ventajas de este método

✅ **Multiplataforma**: Funciona en Windows, Mac y Linux
✅ **Sin dependencias externas**: Solo Python + markdown
✅ **Control total**: Puedes editar el CSS en `generate_html.py`
✅ **Doble uso**: El HTML se ve genial en pantalla y en PDF
✅ **Rápido**: Genera en < 5 segundos

## 🛠️ Personalización

Edita `generate_html.py` para cambiar:

- **Colores**: Busca los códigos hex (#3498db, #2c3e50, etc.)
- **Fuentes**: Cambia `font-family` en el CSS
- **Márgenes**: Ajusta `margin` en `@page`
- **Tamaño de código**: Modifica `font-size` en `pre code`

## 📝 Notas

- El HTML generado (~200 KB) no se guarda en Git (ver `.gitignore`)
- Puedes regenerarlo en cualquier momento con el script
- El PDF resultante suele pesar ~2-3 MB dependiendo de tu navegador
- Para mejor calidad, usa Chrome/Edge (mejor rendering de CSS para impresión)

## 🐛 Troubleshooting

**El código se ve sin formato:**
- Asegúrate de que "Gráficos de fondo" esté activado en las opciones de impresión

**Las tablas se parten entre páginas:**
- Esto es normal, pero el CSS intenta evitarlo con `page-break-inside: avoid`

**El archivo es muy grande:**
- Considera imprimir solo las secciones que necesitas
- O usa una herramienta de compresión de PDF online

**Faltan emojis:**
- Algunos navegadores/PDFs no renderizan emojis perfectamente
- Es cosmético, no afecta el contenido

## 💡 Alternativas

Si prefieres usar otras herramientas:

```bash
# Con pandoc (requiere instalación separada)
pandoc THEORY_COMPLETE.md -o output.pdf --toc --pdf-engine=xelatex

# Con markdown-pdf (Node.js)
npm install -g markdown-pdf
markdown-pdf THEORY_COMPLETE.md
```

## 📬 Feedback

Si encuentras problemas o tienes sugerencias para mejorar el formato del PDF, abre un issue en el repositorio.

---

**Happy Reading! 📖✨**
