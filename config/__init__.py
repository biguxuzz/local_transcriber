# -*- coding: utf-8 -*-
"""
Модуль конфигурации Local Transcriber
Copyright (C) 2025 Local Transcriber Project

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

from .logging_config import setup_logging, get_logger
from .settings import Config, DevelopmentConfig, ProductionConfig

__all__ = ['setup_logging', 'get_logger', 'Config', 'DevelopmentConfig', 'ProductionConfig'] 