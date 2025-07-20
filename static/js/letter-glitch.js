class LetterGlitch {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            console.error(`Container with id "${containerId}" not found`);
            return;
        }

        // Настройки по умолчанию
        this.options = {
            glitchColors: options.glitchColors || ['#2b4539', '#61dca3', '#61b3dc'],
            glitchSpeed: options.glitchSpeed || 50,
            centerVignette: options.centerVignette || true,
            outerVignette: options.outerVignette || false,
            smooth: options.smooth !== undefined ? options.smooth : true,
            ...options
        };

        this.canvas = null;
        this.context = null;
        this.animationFrame = null;
        this.letters = [];
        this.grid = { columns: 0, rows: 0 };
        this.lastGlitchTime = Date.now();

        this.fontSize = 16;
        this.charWidth = 10;
        this.charHeight = 20;

        this.lettersAndSymbols = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            '!', '@', '#', '$', '&', '*', '(', ')', '-', '_', '+', '=', '/',
            '[', ']', '{', '}', ';', ':', '<', '>', ',', '0', '1', '2', '3',
            '4', '5', '6', '7', '8', '9'
        ];

        this.init();
    }

    init() {
        this.createCanvas();
        this.setupContainer();
        this.createVignettes();
        this.resizeCanvas();
        this.startAnimation();
        this.setupEventListeners();
    }

    createCanvas() {
        this.canvas = document.createElement('canvas');
        this.canvas.style.cssText = `
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
        `;
        this.context = this.canvas.getContext('2d');
        this.container.appendChild(this.canvas);
    }

    setupContainer() {
        // Сохраняем исходную позицию контейнера
        const originalPosition = getComputedStyle(this.container).position;
        if (originalPosition === 'static') {
            this.container.style.position = 'relative';
        }

        // Устанавливаем фоновый цвет
        this.container.style.backgroundColor = '#000000';
        this.container.style.overflow = 'hidden';
    }

    createVignettes() {
        if (this.options.outerVignette) {
            const outerVignette = document.createElement('div');
            outerVignette.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                background: radial-gradient(circle, rgba(0,0,0,0) 60%, rgba(0,0,0,1) 100%);
                z-index: 0;
            `;
            this.container.appendChild(outerVignette);
        }

        if (this.options.centerVignette) {
            const centerVignette = document.createElement('div');
            centerVignette.style.cssText = `
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
                background: radial-gradient(circle, rgba(0,0,0,0.8) 0%, rgba(0,0,0,0) 60%);
                z-index: 0;
            `;
            this.container.appendChild(centerVignette);
        }
    }

    getRandomChar() {
        return this.lettersAndSymbols[Math.floor(Math.random() * this.lettersAndSymbols.length)];
    }

    getRandomColor() {
        return this.options.glitchColors[Math.floor(Math.random() * this.options.glitchColors.length)];
    }

    hexToRgb(hex) {
        const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
        hex = hex.replace(shorthandRegex, (m, r, g, b) => {
            return r + r + g + g + b + b;
        });

        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16),
            g: parseInt(result[2], 16),
            b: parseInt(result[3], 16)
        } : null;
    }

    interpolateColor(start, end, factor) {
        const result = {
            r: Math.round(start.r + (end.r - start.r) * factor),
            g: Math.round(start.g + (end.g - start.g) * factor),
            b: Math.round(start.b + (end.b - start.b) * factor),
        };
        return `rgb(${result.r}, ${result.g}, ${result.b})`;
    }

    calculateGrid(width, height) {
        const columns = Math.ceil(width / this.charWidth);
        const rows = Math.ceil(height / this.charHeight);
        return { columns, rows };
    }

    initializeLetters(columns, rows) {
        this.grid = { columns, rows };
        const totalLetters = columns * rows;
        this.letters = Array.from({ length: totalLetters }, () => ({
            char: this.getRandomChar(),
            color: this.getRandomColor(),
            targetColor: this.getRandomColor(),
            colorProgress: 1,
        }));
    }

    resizeCanvas() {
        if (!this.canvas || !this.container) return;

        const dpr = window.devicePixelRatio || 1;
        const rect = this.container.getBoundingClientRect();

        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;

        this.canvas.style.width = `${rect.width}px`;
        this.canvas.style.height = `${rect.height}px`;

        if (this.context) {
            this.context.setTransform(dpr, 0, 0, dpr, 0, 0);
        }

        const { columns, rows } = this.calculateGrid(rect.width, rect.height);
        this.initializeLetters(columns, rows);

        this.drawLetters();
    }

    drawLetters() {
        if (!this.context || this.letters.length === 0) return;
        
        const rect = this.container.getBoundingClientRect();
        this.context.clearRect(0, 0, rect.width, rect.height);
        this.context.font = `${this.fontSize}px monospace`;
        this.context.textBaseline = 'top';

        this.letters.forEach((letter, index) => {
            const x = (index % this.grid.columns) * this.charWidth;
            const y = Math.floor(index / this.grid.columns) * this.charHeight;
            this.context.fillStyle = letter.color;
            this.context.fillText(letter.char, x, y);
        });
    }

    updateLetters() {
        if (!this.letters || this.letters.length === 0) return;

        const updateCount = Math.max(1, Math.floor(this.letters.length * 0.05));

        for (let i = 0; i < updateCount; i++) {
            const index = Math.floor(Math.random() * this.letters.length);
            if (!this.letters[index]) continue;

            this.letters[index].char = this.getRandomChar();
            this.letters[index].targetColor = this.getRandomColor();

            if (!this.options.smooth) {
                this.letters[index].color = this.letters[index].targetColor;
                this.letters[index].colorProgress = 1;
            } else {
                this.letters[index].colorProgress = 0;
            }
        }
    }

    handleSmoothTransitions() {
        let needsRedraw = false;
        this.letters.forEach((letter) => {
            if (letter.colorProgress < 1) {
                letter.colorProgress += 0.05;
                if (letter.colorProgress > 1) letter.colorProgress = 1;

                const startRgb = this.hexToRgb(letter.color);
                const endRgb = this.hexToRgb(letter.targetColor);
                if (startRgb && endRgb) {
                    letter.color = this.interpolateColor(startRgb, endRgb, letter.colorProgress);
                    needsRedraw = true;
                }
            }
        });

        if (needsRedraw) {
            this.drawLetters();
        }
    }

    animate() {
        const now = Date.now();
        if (now - this.lastGlitchTime >= this.options.glitchSpeed) {
            this.updateLetters();
            this.drawLetters();
            this.lastGlitchTime = now;
        }

        if (this.options.smooth) {
            this.handleSmoothTransitions();
        }

        this.animationFrame = requestAnimationFrame(() => this.animate());
    }

    startAnimation() {
        this.animate();
    }

    setupEventListeners() {
        let resizeTimeout;

        const handleResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.stop();
                this.resizeCanvas();
                this.startAnimation();
            }, 100);
        };

        window.addEventListener('resize', handleResize);
        this.handleResize = handleResize;
    }

    stop() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
            this.animationFrame = null;
        }
    }

    destroy() {
        this.stop();
        if (this.handleResize) {
            window.removeEventListener('resize', this.handleResize);
        }
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }

    updateOptions(newOptions) {
        this.options = { ...this.options, ...newOptions };
        this.resizeCanvas();
    }
}

// Экспорт для использования в других скриптах
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LetterGlitch;
} else if (typeof window !== 'undefined') {
    window.LetterGlitch = LetterGlitch;
} 