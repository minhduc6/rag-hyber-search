from app.models import KnowledgeBase, User
from app.schemas import KnowledgeBaseCreate, UserCreate
import hashlib
from typing import List, Any, Dict
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, Query
from sqlalchemy.orm import Session
from langchain_chroma import Chroma
from sqlalchemy import text
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel
from sqlalchemy.orm import selectinload
import os
import time
import asyncio

from app.db.session import get_db
from app.core.security import get_current_user
from app.models.knowledge import KnowledgeBase, Document, ProcessingTask, DocumentChunk, DocumentUpload
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    DocumentResponse,
    PreviewRequest
)
from app.models.chat import Chat, Message
from app.services.document_processor import process_document_background, upload_document, preview_document, PreviewResult
from app.core.config import settings
from app.core.minio import get_minio_client
from minio.error import MinioException
from app.services.vector_store import VectorStoreFactory
from app.services.embedding.embedding_factory import EmbeddingsFactory
from app.core import security
import io  # Thêm import này

logger = logging.getLogger(__name__)

def seed_upload_documents(file_paths: List[str], kb_id: int, db: Session):
    minio_client = get_minio_client()
    results = []

    for file_path in file_paths:
        with open(file_path, "rb") as f:
            file_content = f.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        file_name = os.path.basename(file_path)
        temp_path = f"kb_{kb_id}/temp/{file_name}"

        # Chuyển bytes thành file-like object bằng io.BytesIO
        file_stream = io.BytesIO(file_content)

        # Upload file lên MinIO dùng stream
        minio_client.put_object(
            bucket_name=settings.MINIO_BUCKET_NAME,
            object_name=temp_path,
            data=file_stream,
            length=len(file_content),
            content_type="application/pdf"  # hoặc tự detect mime
        )

        # Tạo record upload
        upload = DocumentUpload(
            knowledge_base_id=kb_id,
            file_name=file_name,
            file_hash=file_hash,
            file_size=len(file_content),
            content_type="application/pdf",
            temp_path=temp_path,
            status="pending"
        )
        db.add(upload)
        db.commit()
        db.refresh(upload)

        results.append(upload)

    return results

async def process_kb_documents_sync(
    kb_id: int,
    upload_results: List[dict],
    db: Session,
    current_user: User
):
    task_info = []
    upload_ids = []

    for result in upload_results:
        if result.get("skip_processing"):
            continue
        upload_ids.append(result["upload_id"])

    if not upload_ids:
        return {"tasks": []}

    uploads = db.query(DocumentUpload).filter(DocumentUpload.id.in_(upload_ids)).all()
    uploads_dict = {upload.id: upload for upload in uploads}

    all_tasks = []
    for upload_id in upload_ids:
        upload = uploads_dict.get(upload_id)
        if not upload:
            continue

        task = ProcessingTask(
            document_upload_id=upload_id,
            knowledge_base_id=kb_id,
            status="pending"
        )
        all_tasks.append(task)

    db.add_all(all_tasks)
    db.commit()

    for task in all_tasks:
        db.refresh(task)

    task_data = []
    for i, upload_id in enumerate(upload_ids):
        if i < len(all_tasks):
            task = all_tasks[i]
            upload = uploads_dict.get(upload_id)

            task_info.append({
                "upload_id": upload_id,
                "task_id": task.id
            })

            if upload:
                task_data.append({
                    "task_id": task.id,
                    "upload_id": upload_id,
                    "temp_path": upload.temp_path,
                    "file_name": upload.file_name
                })

    # Vòng lặp xử lý đồng bộ documents
    for data in task_data:
        try:
            await process_document_background(
                    data["temp_path"],
                    data["file_name"],
                    kb_id,
                    data["task_id"],
                    None
                )
            task = db.query(ProcessingTask).get(data["task_id"])
            task.status = "completed"
            db.commit()
        except Exception as e:
            logger.error(f"Error processing task {data['task_id']}: {str(e)}")
            task = db.query(ProcessingTask).get(data["task_id"])
            task.status = "error"
            db.commit()
    
    # # Chờ tất cả task hoàn thành
    # await asyncio.gather(*tasks, return_exceptions=True)
            
    return {"tasks": task_info}

async def seed_knowledge_base(db: Session):
    kb_name = "My Knowledge Base nhanan sự v1"
    username = "seed_user"

    # Tạo user nếu chưa có
    user = db.query(User).filter(User.username == username).first()
    if not user:
        user_in = UserCreate(username=username, email="seed_user123@example.com", password="1234565665")
        user = User(
            email=user_in.email,
            username=user_in.username,
            hashed_password=security.get_password_hash(user_in.password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Created seed user {user.username}")

    # Tạo knowledge base nếu chưa có
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.name == kb_name,
        KnowledgeBase.user_id == user.id
    ).first()

    if kb:
        logger.info(f"KnowledgeBase '{kb_name}' already exists for user {user.id}")
        return kb

    kb_in = KnowledgeBaseCreate(name=kb_name, description="Mô tả knowledge base")
    kb = KnowledgeBase(
        name=kb_in.name,
        description=kb_in.description,
        user_id=user.id
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    logger.info(f"Knowledge base created: {kb.name} for user {user.id}")
    
    print(os.path.exists("uploads/20250601_Sổ tay nhân sự.pdf"))

        # Upload một số tài liệu seed nếu muốn
    sample_files = [
        "uploads/20250601_Sổ tay nhân sự.pdf"
    ]

    uploads = seed_upload_documents(sample_files, kb.id, db)
    logger.info(f"Uploaded {len(uploads)} documents to knowledge base: {kb.name}")

    upload_results = [{
        "upload_id": upload.id,
        "file_name": upload.file_name,
        "temp_path": upload.temp_path,
        "status": upload.status,
        "skip_processing": False
    } for upload in uploads]

    await process_kb_documents_sync(kb.id, upload_results, db, user)

    logger.info(f"Processed documents for knowledge base: {kb.name}")


    existing_chat = db.query(Chat).get(1)  # lấy theo primary key
    if existing_chat:
        db.delete(existing_chat)
        db.commit()
        logger.info("Deleted existing Chat with id=1")

    # Tạo mới Chat với id=1
    chat = Chat(id=1, title="Chat Nhân Sự", user_id=1)
    chat.knowledge_bases.append(kb)
    db.add(chat)
    db.flush()   # đảm bảo insert trước commit
    db.commit()
    db.refresh(chat)
    logger.info(f"Created seed Chat with id=1 for user {user.id}, KB {kb.id}")