# -*- coding: utf-8 -*-
"""
Конфигурация логирования для Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import logging
import logging.handlers
import os
from datetime import datetime

def setup_logging(log_level=logging.INFO):
    """
    Настройка логирования для приложения.
    
    Args:
        log_level: Уровень логирования (по умолчанию INFO)
    """
    # Создаем директорию для логов если её нет
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    
    # Формат логов
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Основной логгер
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Удаляем существующие хэндлеры
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Консольный хэндлер
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Файловый хэндлер с ротацией
    log_filename = os.path.join(logs_dir, 'transcriber.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_filename, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Отдельный файл для ошибок
    error_filename = os.path.join(logs_dir, 'errors.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_filename,
        maxBytes=5*1024*1024,  # 5MB
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(log_format, date_format)
    error_handler.setFormatter(error_formatter)
    logger.addHandler(error_handler)
    
    # Отдельный файл для процесса обработки
    process_filename = os.path.join(logs_dir, 'processing.log')
    process_handler = logging.handlers.RotatingFileHandler(
        process_filename,
        maxBytes=20*1024*1024,  # 20MB
        backupCount=3,
        encoding='utf-8'
    )
    process_handler.setLevel(logging.DEBUG)
    process_formatter = logging.Formatter(log_format, date_format)
    process_handler.setFormatter(process_formatter)
    
    # Фильтр для логгера обработки
    class ProcessFilter(logging.Filter):
        def filter(self, record):
            return record.name.startswith('transcriber')
    
    process_handler.addFilter(ProcessFilter())
    logger.addHandler(process_handler)
    
    logging.info("Система логирования инициализирована")
    return logger

def get_logger(name):
    """
    Получить логгер для модуля.
    
    Args:
        name: Имя модуля
        
    Returns:
        logging.Logger: Настроенный логгер
    """
    return logging.getLogger(f'transcriber.{name}')

def log_function_call(func):
    """
    Декоратор для логирования вызовов функций.
    
    Args:
        func: Функция для логирования
        
    Returns:
        wrapper: Обёртка функции с логированием
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Вызов функции {func.__name__} с args={args}, kwargs={kwargs}")
        try:
            start_time = datetime.now()
            result = func(*args, **kwargs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.debug(f"Функция {func.__name__} выполнена успешно за {duration:.2f}с")
            return result
        except Exception as e:
            logger.error(f"Ошибка в функции {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper

def log_processing_step(step_name, details=None):
    """
    Логирование шага обработки.
    
    Args:
        step_name: Название шага
        details: Дополнительные детали (опционально)
    """
    logger = get_logger('processing')
    message = f"ЭТАП: {step_name}"
    if details:
        message += f" - {details}"
    logger.info(message)

def log_progress(current, total, message="Обработка"):
    """
    Логирование прогресса.
    
    Args:
        current: Текущий элемент
        total: Общее количество
        message: Сообщение
    """
    logger = get_logger('processing')
    percentage = (current / total * 100) if total > 0 else 0
    logger.info(f"{message}: {current}/{total} ({percentage:.1f}%)") 