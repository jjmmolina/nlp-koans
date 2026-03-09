#!/usr/bin/env python3
"""
Genera un EPUB a partir de THEORY_COMPLETE.md
"""

from ebooklib import epub
import markdown
import re


def create_epub():
    # Crear libro
    book = epub.EpubBook()

    # Metadatos
    book.set_identifier("nlp-koans-teoria-completa-001")
    book.set_title("NLP Koans - Teoría Completa")
    book.set_language("es")
    book.add_author("jjmmolina")

    # Leer el archivo markdown
    with open("THEORY_COMPLETE.md", "r", encoding="utf-8") as f:
        content = f.read()

    # Separar por secciones principales (## emojis)
    sections = re.split(r"\n(?=## [0-9️⃣]+)", content)

    chapters = []
    toc = []
    spine = ["nav"]

    for i, section in enumerate(sections):
        if not section.strip():
            continue

        # Obtener título de la sección
        title_match = re.match(r"## ([0-9️⃣]+.*?)(?:\n|$)", section)
        if title_match:
            title = title_match.group(1).strip()
        else:
            title = f"Sección {i}"

        # Convertir markdown a HTML
        html_content = markdown.markdown(
            section, extensions=["fenced_code", "codehilite", "tables", "toc", "nl2br"]
        )

        # Crear capítulo
        chapter = epub.EpubHtml(
            title=title, file_name=f"chapter_{i:02d}.xhtml", lang="es"
        )

        # Añadir CSS básico para código
        css = """
        pre {
            background-color: #f4f4f4;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: monospace;
        }
        pre code {
            background-color: transparent;
            padding: 0;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f4f4f4;
        }
        h1, h2, h3 {
            color: #333;
            margin-top: 20px;
        }
        """

        chapter.set_content(
            f"""
        <html>
        <head>
            <style>{css}</style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        )

        book.add_item(chapter)
        chapters.append(chapter)
        spine.append(chapter)
        toc.append(chapter)

    # Añadir tabla de contenidos
    book.toc = tuple(toc)

    # Añadir navegación
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    # Definir spine (orden de lectura)
    book.spine = spine

    # Escribir archivo EPUB
    output_file = "NLP_Koans_Teoria_Completa.epub"
    epub.write_epub(output_file, book, {})

    print(f"✅ EPUB generado exitosamente: {output_file}")
    print(f"📖 Capítulos creados: {len(chapters)}")
    print(f"📄 Tamaño: {len(content):,} caracteres")

    return output_file


if __name__ == "__main__":
    try:
        create_epub()
    except Exception as e:
        print(f"❌ Error generando EPUB: {e}")
        import traceback

        traceback.print_exc()
