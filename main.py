import os
import uuid
import json
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import anthropic
import chromadb
from chromadb.utils import embedding_functions

# 檔案處理
import fitz  # PyMuPDF - 處理PDF
from pptx import Presentation  # 處理PPT
from docx import Document  # 處理Word
from PIL import Image
import io

app = FastAPI(title="Knowledge Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 初始化 ──────────────────────────────────────────
UPLOAD_DIR = Path("uploads")
PAGES_DIR = Path("page_images")
UPLOAD_DIR.mkdir(exist_ok=True)
PAGES_DIR.mkdir(exist_ok=True)

# ChromaDB - 向量資料庫
chroma_client = chromadb.PersistentClient(path="./chroma_db")
embedding_fn = embedding_functions.DefaultEmbeddingFunction()
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# Claude API
claude_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

# ── 工具函數 ──────────────────────────────────────────

def pdf_to_page_images(pdf_path: Path, doc_id: str) -> list[Path]:
    """把PDF每一頁轉成圖片，回傳圖片路徑列表"""
    doc = fitz.open(str(pdf_path))
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        mat = fitz.Matrix(2.0, 2.0)  # 2x解析度
        pix = page.get_pixmap(matrix=mat)
        img_path = PAGES_DIR / f"{doc_id}_page_{page_num}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)
    doc.close()
    return image_paths


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """從PDF抽取每頁文字，回傳 [{page: int, text: str}]"""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i, "text": text})
    doc.close()
    return pages


def convert_pptx_to_pdf(pptx_path: Path) -> Path:
    """將PPT轉換為PDF（需要LibreOffice）"""
    pdf_path = pptx_path.with_suffix(".pdf")
    os.system(f'libreoffice --headless --convert-to pdf "{pptx_path}" --outdir "{pptx_path.parent}"')
    return pdf_path


def convert_docx_to_pdf(docx_path: Path) -> Path:
    """將Word轉換為PDF（需要LibreOffice）"""
    pdf_path = docx_path.with_suffix(".pdf")
    os.system(f'libreoffice --headless --convert-to pdf "{docx_path}" --outdir "{docx_path.parent}"')
    return pdf_path


def process_image_file(img_path: Path) -> list[dict]:
    """圖片檔案：用Claude Vision描述內容"""
    with open(img_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    suffix = img_path.suffix.lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_map.get(suffix, "image/png")
    
    response = claude_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": img_data}},
                {"type": "text", "text": "請詳細描述這張圖片的所有內容，包含文字、圖表、數據等，用繁體中文回答。"}
            ]
        }]
    )
    description = response.content[0].text
    return [{"page": 0, "text": description}]


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """將長文字切成小段，讓搜尋更精準"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── API 路由 ──────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """上傳並處理檔案，存入向量資料庫"""
    try:
        doc_id = str(uuid.uuid4())[:8]
        suffix = Path(file.filename).suffix.lower()
        saved_path = UPLOAD_DIR / f"{doc_id}{suffix}"
        
        # 儲存原始檔案
        with open(saved_path, "wb") as f:
            f.write(await file.read())
        
        # 根據檔案類型處理
        pages_data = []
        pdf_path = None
        
        if suffix == ".pdf":
            pdf_path = saved_path
            pages_data = extract_text_from_pdf(saved_path)
            
        elif suffix in [".pptx", ".ppt"]:
            pdf_path = convert_pptx_to_pdf(saved_path)
            if pdf_path.exists():
                pages_data = extract_text_from_pdf(pdf_path)
            
        elif suffix in [".docx", ".doc"]:
            pdf_path = convert_docx_to_pdf(saved_path)
            if pdf_path.exists():
                pages_data = extract_text_from_pdf(pdf_path)
                
        elif suffix in [".jpg", ".jpeg", ".png", ".gif", ".webp"]:
            pages_data = process_image_file(saved_path)
            img_dest = PAGES_DIR / f"{doc_id}_page_0.png"
            img = Image.open(saved_path)
            img.save(str(img_dest))
        
        if not pages_data:
            raise HTTPException(status_code=400, detail="無法解析此檔案內容")
        
        if pdf_path and pdf_path.exists():
            pdf_to_page_images(pdf_path, doc_id)
        
        # 存入向量資料庫
        total_chunks = 0
        for page_info in pages_data:
            page_num = page_info["page"]
            text = page_info["text"]
            chunks = chunk_text(text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_p{page_num}_c{chunk_idx}"
                collection.add(
                    documents=[chunk],
                    ids=[chunk_id],
                    metadatas=[{
                        "doc_id": doc_id,
                        "filename": file.filename,
                        "page": page_num,
                        "chunk_idx": chunk_idx
                    }]
                )
                total_chunks += 1
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": len(pages_data),
            "chunks": total_chunks,
            "message": f"✅ 成功處理 {len(pages_data)} 頁，建立 {total_chunks} 個索引"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"處理失敗：{str(e)}")


@app.post("/ask")
async def ask_question(question: str = Form(...), top_k: int = Form(5)):
    """問問題，從資料庫找答案"""
    try:
        results = collection.query(
            query_texts=[question],
            n_results=min(top_k, collection.count())
        )
        
        if not results["documents"][0]:
            return {"answer": "資料庫中找不到相關資料，請先上傳文件。", "sources": []}
        
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        
        context_parts = []
        sources = []
        seen = set()
        
        for doc, meta in zip(docs, metas):
            context_parts.append(f"【來源：{meta['filename']} 第{meta['page']+1}頁】\n{doc}")
            
            key = f"{meta['doc_id']}_p{meta['page']}"
            if key not in seen:
                seen.add(key)
                img_path = PAGES_DIR / f"{meta['doc_id']}_page_{meta['page']}.png"
                sources.append({
                    "filename": meta["filename"],
                    "page": meta["page"] + 1,
                    "doc_id": meta["doc_id"],
                    "has_image": img_path.exists(),
                    "relevant_text": doc[:200]
                })
        
        context = "\n\n---\n\n".join(context_parts)
        
        system_prompt = """你是一個封閉式知識庫助手。
你只能根據使用者提供的文件內容來回答問題。
如果提供的資料中找不到答案，請明確說「資料庫中無此資訊」。
絕對不能使用任何外部知識或自行推測。
請用繁體中文回答，並標明資訊來自哪個文件的哪一頁。"""
        
        response = claude_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            system=system_prompt,
            messages=[{
                "role": "user",
                "content": f"以下是資料庫中找到的相關內容：\n\n{context}\n\n問題：{question}"
            }]
        )
        
        answer = response.content[0].text
        
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查詢失敗：{str(e)}")


@app.get("/page-image/{doc_id}/{page_num}")
async def get_page_image(doc_id: str, page_num: int):
    img_path = PAGES_DIR / f"{doc_id}_page_{page_num}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="頁面圖片不存在")
    
    with open(img_path, "rb") as f:
        img_data = base64.standard_b64encode(f.read()).decode("utf-8")
    
    return {"image_base64": img_data}


@app.get("/documents")
async def list_documents():
    try:
        all_items = collection.get()
        docs = {}
        for meta in all_items["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "pages": set()
                }
            docs[doc_id]["pages"].add(meta["page"])
        
        result = []
        for doc in docs.values():
            result.append({
                "doc_id": doc["doc_id"],
                "filename": doc["filename"],
                "page_count": len(doc["pages"])
            })
        
        return {"documents": result}
    except Exception as e:
        return {"documents": []}


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    try:
        all_items = collection.get()
        ids_to_delete = [
            id_ for id_, meta in zip(all_items["ids"], all_items["metadatas"])
            if meta["doc_id"] == doc_id
        ]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        
        for img_file in PAGES_DIR.glob(f"{doc_id}_*.png"):
            img_file.unlink()
        
        return {"success": True, "message": f"已刪除文件 {doc_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 靜態檔案
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/page-images", StaticFiles(directory="page_images"), name="page_images")
