# -*- coding: utf-8 -*-
"""
Flask приложение Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from flask import Flask
from config.settings import config, setup_logging, get_logger

def create_app(config_name='default'):
    """
    Фабрика приложений Flask.
    
    Args:
        config_name: Название конфигурации ('development', 'production', 'testing')
        
    Returns:
        Flask: Настроенное приложение Flask
    """
    # Создаем приложение с правильными путями к шаблонам и статическим файлам
    app = Flask(__name__, 
                template_folder='../templates',
                static_folder='../static')
    
    # Загрузка конфигурации
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)
    
    # Настройка логирования
    logger = get_logger('app')
    logger.info(f"Инициализация приложения с конфигурацией: {config_name}")
    
    # Регистрация маршрутов
    from app.routes import main
    app.register_blueprint(main)
    
    logger.info("Приложение успешно инициализировано")
    return app 