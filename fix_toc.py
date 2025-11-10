#!/usr/bin/env python3
"""
Script para generar automáticamente el TOC correcto del documento THEORY_COMPLETE.md
"""

import re
from pathlib import Path


def generate_anchor(title):
    """Genera anchor de GitHub a partir del título"""
    # GitHub: minúsculas, sin emojis/puntuación especial, espacios → guiones
    anchor = title.lower()
    # Remover emojis y caracteres especiales
    anchor = re.sub(r"[^\w\s-]", "", anchor)
    # Espacios → guiones
    anchor = anchor.strip().replace(" ", "-")
    # Múltiples guiones → uno
    anchor = re.sub(r"-+", "-", anchor)
    return anchor


def extract_headers():
    """Extrae todos los headers del documento"""
    md_file = Path("THEORY_COMPLETE.md")
    content = md_file.read_text(encoding="utf-8")

    headers = []
    for line in content.split("\n"):
        if line.startswith("#"):
            # Contar nivel
            match = re.match(r"^(#+)\s+(.+)$", line)
            if match:
                level = len(match.group(1))
                title = match.group(2).strip()

                # Saltar el título principal y el TOC mismo
                if level == 1 or "Tabla de Contenidos" in title:
                    continue

                anchor = generate_anchor(title)
                headers.append({"level": level, "title": title, "anchor": anchor})

    return headers


def generate_toc():
    """Genera el TOC en formato Markdown"""
    headers = extract_headers()

    toc_lines = ["### 📑 Tabla de Contenidos"]

    for h in headers:
        # Calcular indentación (nivel 2 = sin indent, nivel 3 = 4 espacios, etc)
        indent = "    " * (h["level"] - 2)
        # Crear enlace
        link = f"{indent}- [{h['title']}](#{h['anchor']})"
        toc_lines.append(link)

    return "\n".join(toc_lines)


def update_toc_in_file():
    """Actualiza el TOC en el archivo"""
    md_file = Path("THEORY_COMPLETE.md")
    lines = md_file.read_text(encoding="utf-8").split("\n")

    # Encontrar inicio y fin del TOC
    toc_start = None
    toc_end = None

    for i, line in enumerate(lines):
        if "### 📑 Tabla de Contenidos" in line or "###  Tabla de Contenidos" in line:
            toc_start = i
        if toc_start is not None and line.startswith("> Nota:"):
            toc_end = i
            break

    if toc_start is None or toc_end is None:
        print("❌ No se pudo encontrar el TOC en el archivo")
        return

    # Generar nuevo TOC
    new_toc = generate_toc()

    # Reemplazar
    new_lines = lines[:toc_start] + new_toc.split("\n") + [""] + lines[toc_end:]

    # Guardar
    md_file.write_text("\n".join(new_lines), encoding="utf-8")

    print("✅ TOC actualizado correctamente")
    print(f"📊 Headers procesados: {len(extract_headers())}")
    print(f"📝 Líneas del nuevo TOC: {len(new_toc.split(chr(10)))}")


if __name__ == "__main__":
    update_toc_in_file()
