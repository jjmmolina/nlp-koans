#!/usr/bin/env python3
"""
Script para generar HTML profesional del documento THEORY_COMPLETE.md
que puede ser convertido a PDF desde el navegador (Ctrl+P → Guardar como PDF)
"""

import markdown
from pathlib import Path
from datetime import datetime


def generate_html():
    """Convierte THEORY_COMPLETE.md a HTML con formato profesional para PDF"""

    # Leer markdown
    md_file = Path("THEORY_COMPLETE.md")
    md_content = md_file.read_text(encoding="utf-8")

    # Configurar markdown con extensiones
    md = markdown.Markdown(
        extensions=[
            "extra",  # tablas, footnotes, etc
            "codehilite",  # syntax highlighting
            "toc",  # tabla de contenidos
            "fenced_code",  # bloques de código con ```
            "tables",  # soporte de tablas
        ]
    )

    # Convertir a HTML
    html_body = md.convert(md_content)

    # Template HTML con CSS embebido para PDF
    html_template = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NLP Koans - Teoría Completa</title>
    <style>
        /* === Configuración de impresión para PDF === */
        @page {{
            size: A4;
            margin: 2.5cm 2cm;
        }}
        
        @media print {{
            h1 {{ page-break-before: always; }}
            h1:first-of-type {{ page-break-before: avoid; }}
            h2, h3, h4 {{ page-break-after: avoid; }}
            pre, table, blockquote {{ page-break-inside: avoid; }}
            .no-print {{ display: none !important; }}
            a {{ text-decoration: none; color: #2c3e50; }}
        }}
        
        /* === Estilos generales === */
        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.7;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 2.2em;
            border-bottom: 4px solid #3498db;
            padding-bottom: 12px;
            margin-top: 50px;
            margin-bottom: 25px;
        }}
        
        h2 {{
            color: #34495e;
            font-size: 1.8em;
            border-bottom: 2px solid #95a5a6;
            padding-bottom: 10px;
            margin-top: 40px;
            margin-bottom: 20px;
        }}
        
        h3 {{
            color: #555;
            font-size: 1.4em;
            margin-top: 30px;
            margin-bottom: 15px;
        }}
        
        p {{
            margin: 12px 0;
            text-align: justify;
        }}
        
        code {{
            background-color: #f4f6f8;
            border: 1px solid #e1e4e8;
            border-radius: 3px;
            padding: 2px 8px;
            font-family: 'Consolas', 'Courier New', monospace;
            font-size: 0.88em;
            color: #e83e8c;
        }}
        
        pre {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-left: 5px solid #3498db;
            border-radius: 5px;
            padding: 18px;
            margin: 20px 0;
            overflow-x: auto;
        }}
        
        pre code {{
            background: none;
            border: none;
            padding: 0;
            font-size: 0.9em;
            color: #333;
        }}
        
        ul, ol {{ margin: 15px 0; padding-left: 35px; }}
        li {{ margin: 8px 0; }}
        
        blockquote {{
            border-left: 5px solid #3498db;
            padding: 15px 20px;
            margin: 20px 0;
            background-color: #f8f9fa;
            color: #555;
            font-style: italic;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 25px 0;
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 14px;
            text-align: left;
            border: 1px solid #5568d3;
        }}
        
        td {{
            border: 1px solid #dee2e6;
            padding: 12px;
        }}
        
        tr:nth-child(even) td {{
            background-color: #f8f9fa;
        }}
        
        a {{
            color: #3498db;
            text-decoration: none;
        }}
        
        .cover {{
            text-align: center;
            padding: 100px 0;
            border-bottom: 3px solid #3498db;
            margin-bottom: 50px;
        }}
        
        .cover h1 {{
            font-size: 3em;
            border: none;
            margin: 0;
        }}
        
        .utility-buttons {{
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
        }}
        
        .btn {{
            background: #3498db;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }}
        
        .btn:hover {{
            background: #2980b9;
        }}
        
        @media print {{
            .utility-buttons {{ display: none; }}
        }}
    </style>
</head>
<body>
    <div class="utility-buttons no-print">
        <button class="btn" onclick="window.print()">🖨️ Imprimir / Guardar PDF</button>
    </div>
    
    <div class="cover no-print">
        <h1>📚 NLP Koans</h1>
        <p style="font-size: 1.3em; color: #7f8c8d; margin: 20px 0;">Teoría Completa</p>
        <p style="font-size: 1.1em; color: #95a5a6;">Del procesamiento básico de texto a los LLMs modernos</p>
        <p style="font-size: 1em; color: #95a5a6; margin-top: 30px;">Generado: {datetime.now().strftime("%d de %B, %Y")}</p>
    </div>
    
    <div class="content">
        {html_body}
    </div>
</body>
</html>"""

    # Guardar HTML
    output_file = "NLP_Koans_Teoria_Completa.html"
    Path(output_file).write_text(html_template, encoding="utf-8")

    print(f"✅ HTML generado exitosamente: {output_file}")
    print(f"📊 Tamaño: {Path(output_file).stat().st_size / 1024:.2f} KB")
    print()
    print("📖 Pasos para generar PDF:")
    print("   1. Abre el archivo HTML en tu navegador")
    print("   2. Presiona Ctrl+P (o Cmd+P en Mac)")
    print("   3. Selecciona 'Guardar como PDF'")
    print("   4. Ajusta los márgenes si es necesario")
    print("   5. Guarda el PDF")


if __name__ == "__main__":
    generate_html()
