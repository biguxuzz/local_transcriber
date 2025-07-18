# -*- coding: utf-8 -*-
"""
Модели данных Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import uuid
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from config.logging_config import get_logger

logger = get_logger('models')

class ProcessingStatus(Enum):
    """Статусы обработки."""
    PENDING = "pending"
    UPLOADING = "uploading"
    CONVERTING = "converting"
    COMBINING = "combining"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    COMPLETED = "completed"
    ERROR = "error"

class FileType(Enum):
    """Типы файлов."""
    WAV = "wav"
    MP3 = "mp3"
    MP4 = "mp4"
    MKV = "mkv"

@dataclass
class UploadedFile:
    """Модель загруженного файла."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    original_filename: str = ""
    file_type: FileType = FileType.WAV
    size_bytes: int = 0
    duration_seconds: Optional[float] = None
    upload_time: datetime = field(default_factory=datetime.now)
    file_path: str = ""
    
    def __post_init__(self):
        """Логирование создания файла."""
        logger.debug(f"Создан объект UploadedFile: {self.original_filename}")

@dataclass
class SpeakerSegment:
    """Сегмент речи спикера."""
    
    speaker_id: str
    start_time: float
    end_time: float
    text: str = ""
    confidence: float = 0.0
    
    @property
    def duration(self) -> float:
        """Длительность сегмента."""
        return self.end_time - self.start_time
    
    @property
    def formatted_time(self) -> str:
        """Отформатированное время в формате HH:MM:SS."""
        start_hours = int(self.start_time // 3600)
        start_minutes = int((self.start_time % 3600) // 60)
        start_seconds = int(self.start_time % 60)
        
        end_hours = int(self.end_time // 3600)
        end_minutes = int((self.end_time % 3600) // 60)
        end_end_seconds = int(self.end_time % 60)
        
        return f"[{start_hours:02d}:{start_minutes:02d}:{start_seconds:02d}-{end_hours:02d}:{end_minutes:02d}:{end_end_seconds:02d}]"

@dataclass
class ProcessingJob:
    """Задача обработки."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    files: List[UploadedFile] = field(default_factory=list)
    status: ProcessingStatus = ProcessingStatus.PENDING
    current_step: str = ""
    progress_percentage: float = 0.0
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Результаты обработки
    combined_audio_path: Optional[str] = None
    speaker_segments: List[SpeakerSegment] = field(default_factory=list)
    result_file_path: Optional[str] = None
    
    # Метаданные
    total_duration: Optional[float] = None
    speaker_count: int = 0
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Логирование создания задачи."""
        logger.info(f"Создана задача обработки: {self.id}")
    
    def start_processing(self):
        """Начать обработку."""
        self.status = ProcessingStatus.UPLOADING
        self.started_at = datetime.now()
        logger.info(f"Начата обработка задачи: {self.id}")
    
    def update_progress(self, status: ProcessingStatus, step: str, percentage: float):
        """Обновить прогресс обработки."""
        self.status = status
        self.current_step = step
        self.progress_percentage = percentage
        logger.info(f"Задача {self.id}: {status.value} - {step} ({percentage:.1f}%)")
    
    def complete_processing(self, result_path: str):
        """Завершить обработку."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result_file_path = result_path
        
        if self.started_at:
            self.processing_time = (self.completed_at - self.started_at).total_seconds()
        
        logger.info(f"Задача {self.id} завершена успешно за {self.processing_time:.2f}с")
    
    def set_error(self, error_message: str):
        """Установить ошибку обработки."""
        self.status = ProcessingStatus.ERROR
        self.error_message = error_message
        self.completed_at = datetime.now()
        logger.error(f"Ошибка в задаче {self.id}: {error_message}")
    
    def add_file(self, file: UploadedFile):
        """Добавить файл к задаче."""
        self.files.append(file)
        logger.debug(f"Файл {file.original_filename} добавлен к задаче {self.id}")
    
    def get_total_size(self) -> int:
        """Получить общий размер всех файлов."""
        return sum(file.size_bytes for file in self.files)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразовать в словарь для JSON."""
        return {
            'id': self.id,
            'job_id': self.id,  # Для совместимости с фронтендом
            'status': self.status.value,
            'current_step': self.current_step,
            'progress_percentage': self.progress_percentage,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'files_count': len(self.files),
            'total_size': self.get_total_size(),
            'total_duration': self.total_duration,
            'speaker_count': self.speaker_count,
            'processing_time': self.processing_time,
            'result_available': self.result_file_path is not None
        }

class JobManager:
    """Менеджер задач обработки."""
    
    def __init__(self):
        self.jobs: Dict[str, ProcessingJob] = {}
        self.logger = get_logger('job_manager')
        self.logger.info("Инициализирован менеджер задач")
    
    def create_job(self) -> ProcessingJob:
        """Создать новую задачу."""
        job = ProcessingJob()
        self.jobs[job.id] = job
        self.logger.info(f"Создана новая задача: {job.id}")
        return job
    
    def get_job(self, job_id: str) -> Optional[ProcessingJob]:
        """Получить задачу по ID."""
        return self.jobs.get(job_id)
    
    def delete_job(self, job_id: str) -> bool:
        """Удалить задачу."""
        if job_id in self.jobs:
            del self.jobs[job_id]
            self.logger.info(f"Задача {job_id} удалена")
            return True
        return False
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Очистить старые задачи."""
        current_time = datetime.now()
        jobs_to_delete = []
        
        for job_id, job in self.jobs.items():
            age_hours = (current_time - job.created_at).total_seconds() / 3600
            if age_hours > max_age_hours:
                jobs_to_delete.append(job_id)
        
        for job_id in jobs_to_delete:
            self.delete_job(job_id)
        
        if jobs_to_delete:
            self.logger.info(f"Очищено {len(jobs_to_delete)} старых задач")

# Глобальный экземпляр менеджера задач
job_manager = JobManager() 