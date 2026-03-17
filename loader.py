import os
from pathlib import Path
from pypdf import PdfReader
from docx import Document
import markdown
from bs4 import BeautifulSoup


def load_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_docx(path):
    doc = Document(path)
    return "\n".join([p.text for p in doc.paragraphs])

def load_md(path):
    with open(path, "r", encoding="utf-8") as f:
        html = markdown.markdown(f.read())
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text()

def load_documents(folder):
    docs = []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        content = ""

        if file.endswith(".txt"):
            content = load_txt(path)
        elif file.endswith(".pdf"):
            content = load_pdf(path)
        elif file.endswith(".docx"):
            content = load_docx(path)
        elif file.endswith(".md"):
            content = load_md(path)
        else:
            continue

        docs.append({
            "text": content,
            "filename": file
        })
    return docs


if __name__ == "__main__":
    folder = Path("documents")
    content = load_documents(folder)
    for i, doc in enumerate(content):
        print(f"\n--- Document {i+1} ({doc['filename']}) ---")
        print(doc["text"][:100] + "...")
