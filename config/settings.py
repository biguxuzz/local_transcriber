# -*- coding: utf-8 -*-
"""
Настройки приложения Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import logging
from pathlib import Path

class Config:
    """Базовая конфигурация приложения."""
    
    # Основные настройки Flask
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Пути директорий
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_FOLDER = BASE_DIR / 'uploads'
    RESULTS_FOLDER = BASE_DIR / 'results'
    LOGS_FOLDER = BASE_DIR / 'logs'
    
    # Создание директорий если их нет
    UPLOAD_FOLDER.mkdir(exist_ok=True)
    RESULTS_FOLDER.mkdir(exist_ok=True)
    LOGS_FOLDER.mkdir(exist_ok=True)
    
    # Настройки загрузки файлов
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.mp4', '.mkv'}
    
    # Настройки обработки аудио
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHANNELS = 1  # моно
    AUDIO_FORMAT = 'wav'
    
    # Настройки моделей
    WHISPER_MODEL = os.environ.get('WHISPER_MODEL') or 'medium'  # Улучшенная модель для лучшего качества
    DIARIZATION_MODEL = os.environ.get('DIARIZATION_MODEL') or 'pyannote/speaker-diarization'
    
    # Настройки Hugging Face
    HF_TOKEN = os.environ.get('HF_TOKEN')
    
    # Настройки ONNX Runtime для CPU оптимизации
    USE_ONNX_RUNTIME = True
    ONNX_CPU_THREADS = os.cpu_count()
    FORCE_CPU_INFERENCE = True
    
    # Настройки обработки
    CHUNK_SIZE = 30  # секунд для обработки
    MIN_SPEAKER_DURATION = 1.0  # минимальная длительность сегмента спикера
    
    # Настройки логирования
    LOG_LEVEL = getattr(logging, os.environ.get('LOG_LEVEL', 'INFO').upper())
    
    # Настройки очистки файлов
    CLEANUP_INTERVAL = int(os.environ.get('CLEANUP_INTERVAL', '3600'))  # секунды
    FILE_RETENTION = int(os.environ.get('FILE_RETENTION', '86400'))  # 24 часа
    
    @staticmethod
    def init_app(app):
        """Инициализация приложения с конфигурацией."""
        pass

class DevelopmentConfig(Config):
    """Конфигурация для разработки."""
    
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    
    # Более детальное логирование в разработке
    ENABLE_FUNCTION_LOGGING = True
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Настройка логирования для разработки
        from config.logging_config import setup_logging
        setup_logging(logging.DEBUG)

class ProductionConfig(Config):
    """Конфигурация для продакшена."""
    
    DEBUG = False
    LOG_LEVEL = logging.INFO
    
    # Оптимизация для продакшена
    ENABLE_FUNCTION_LOGGING = False
    
    @staticmethod
    def init_app(app):
        Config.init_app(app)
        
        # Проверка SECRET_KEY в продакшене
        if not app.config.get('SECRET_KEY') or app.config['SECRET_KEY'] == 'dev-secret-key-change-in-production':
            raise ValueError('SECRET_KEY должен быть установлен в продакшене')
        
        # Настройка логирования для продакшена
        from config.logging_config import setup_logging
        setup_logging(logging.INFO)

class TestingConfig(Config):
    """Конфигурация для тестирования."""
    
    TESTING = True
    DEBUG = True
    LOG_LEVEL = logging.WARNING
    
    # Тестовые пути
    UPLOAD_FOLDER = Config.BASE_DIR / 'test_uploads'
    RESULTS_FOLDER = Config.BASE_DIR / 'test_results'
    
    # Более быстрые настройки для тестов
    WHISPER_MODEL = 'tiny'
    CHUNK_SIZE = 10

# Маппинг конфигураций
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Импорт функций логирования
from config.logging_config import setup_logging, get_logger 