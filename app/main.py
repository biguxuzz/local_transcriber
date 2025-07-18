# -*- coding: utf-8 -*-
"""
Основной файл запуска Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import os
import atexit
import threading
import time
from flask import Flask

from config import setup_logging, get_logger
from app import create_app
from app.models import job_manager

# Инициализация логирования
setup_logging()
logger = get_logger('main')

def cleanup_old_jobs():
    """Периодическая очистка старых задач."""
    while True:
        try:
            job_manager.cleanup_old_jobs()
            time.sleep(3600)  # Каждый час
        except Exception as e:
            logger.error(f"Ошибка при очистке задач: {str(e)}")
            time.sleep(60)  # Повторить через минуту при ошибке

def setup_cleanup_thread():
    """Настройка потока очистки."""
    cleanup_thread = threading.Thread(target=cleanup_old_jobs, daemon=True)
    cleanup_thread.start()
    logger.info("Поток очистки старых задач запущен")

def create_application():
    """
    Создание и настройка приложения.
    
    Returns:
        Flask: Настроенное приложение
    """
    config_name = os.environ.get('FLASK_ENV', 'development')
    logger.info(f"Создание приложения с конфигурацией: {config_name}")
    
    app = create_app(config_name)
    
    # Запуск потока очистки
    setup_cleanup_thread()
    
    # Регистрация функции очистки при завершении
    def cleanup_on_exit():
        logger.info("Завершение работы приложения")
    
    atexit.register(cleanup_on_exit)
    
    return app

# Создание приложения
app = create_application()

if __name__ == '__main__':
    logger.info("Запуск приложения Local Transcriber")
    
    # Параметры запуска
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Сервер запускается на {host}:{port}, debug={debug}")
    
    try:
        app.run(
            host=host,
            port=port,
            debug=debug,
            threaded=True,
            use_reloader=False  # Отключаем перезагрузку для стабильности
        )
    except KeyboardInterrupt:
        logger.info("Приложение остановлено пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка при запуске: {str(e)}", exc_info=True)
        raise 