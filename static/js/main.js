/**
 * Local Transcriber JavaScript
 * Copyright (C) 2025 Local Transcriber Project
 * Licensed under GPL-3.0
 */

// Global variables
let currentJobId = null;
let statusCheckInterval = null;
let selectedFiles = [];
let backgroundGlitch = null;

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Local Transcriber interface loaded');
    
    // Initialize background glitch effect
    initializeBackgroundGlitch();
    
    // Setup drag and drop for file upload
    setupDragAndDrop();
    
    // Setup file sorting if SortableJS is available
    setupFileSorting();
    
    // Setup form submission
    setupFormSubmission();
});

/**
 * Initialize background glitch effect
 */
function initializeBackgroundGlitch() {
    // Create background container
    createBackgroundContainer();
    
    // Initialize glitch effect if LetterGlitch is available
    if (typeof LetterGlitch !== 'undefined') {
        try {
            backgroundGlitch = new LetterGlitch('background-glitch', {
                glitchSpeed: 50,
                centerVignette: true,
                outerVignette: false,
                smooth: true,
                glitchColors: ['#2b4539', '#61dca3', '#61b3dc']
            });
            console.log('Background glitch effect initialized');
        } catch (error) {
            console.error('Failed to initialize background glitch:', error);
        }
    } else {
        console.warn('LetterGlitch class not found');
    }
}

/**
 * Create background container for glitch effect
 */
function createBackgroundContainer() {
    // Check if container already exists
    if (document.getElementById('background-glitch')) {
        return;
    }
    
    const backgroundContainer = document.createElement('div');
    backgroundContainer.id = 'background-glitch';
    backgroundContainer.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: -999;
        pointer-events: none;
    `;
    
    // Insert at the beginning of body
    document.body.insertBefore(backgroundContainer, document.body.firstChild);
}

/**
 * Control background glitch effect
 * @param {boolean} enabled - Whether to enable or disable the effect
 */
function toggleBackgroundGlitch(enabled) {
    if (backgroundGlitch) {
        if (enabled) {
            backgroundGlitch.startAnimation();
        } else {
            backgroundGlitch.stop();
        }
    }
}

/**
 * Update glitch effect colors based on processing state
 * @param {string} state - Processing state
 */
function updateGlitchColors(state) {
    if (!backgroundGlitch) return;
    
    const colorSchemes = {
        'idle': ['#2b4539', '#61dca3', '#61b3dc'],
        'processing': ['#856404', '#ffc107', '#fd7e14'],
        'completed': ['#198754', '#20c997', '#0dcaf0'],
        'error': ['#dc3545', '#fd5e53', '#e74c3c']
    };
    
    const colors = colorSchemes[state] || colorSchemes.idle;
    backgroundGlitch.updateOptions({ glitchColors: colors });
}

/**
 * Setup drag and drop functionality for file upload
 */
function setupDragAndDrop() {
    const fileInput = document.getElementById('files');
    const uploadSection = document.getElementById('upload-section');
    
    if (!fileInput || !uploadSection) return;
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadSection.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadSection.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadSection.addEventListener('drop', handleDrop, false);
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        uploadSection.classList.add('drag-over');
    }
    
    function unhighlight() {
        uploadSection.classList.remove('drag-over');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        fileInput.files = files;
        displayFileList(Array.from(files));
    }
}

/**
 * Setup file sorting functionality
 */
function setupFileSorting() {
    // This would require SortableJS library
    // For now, we'll implement basic functionality
    const fileItems = document.getElementById('file-items');
    if (fileItems && typeof Sortable !== 'undefined') {
        new Sortable(fileItems, {
            animation: 150,
            ghostClass: 'sortable-ghost',
            chosenClass: 'sortable-chosen',
            onEnd: function(evt) {
                updateFileOrder();
            }
        });
    }
}

/**
 * Setup form submission handler
 */
function setupFormSubmission() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('files');
    
    if (!uploadForm || !fileInput) {
        console.error('Upload form or file input not found');
        return;
    }
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        const files = Array.from(e.target.files);
        displayFileList(files);
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const files = fileInput.files;
        if (files.length === 0) {
            showAlert('Пожалуйста, выберите файлы для загрузки', 'warning');
            return;
        }
        
        console.log('Starting upload of', files.length, 'files');
        uploadFiles(files);
    });
}

/**
 * Display selected files in the interface
 * @param {Array} files - Array of File objects
 */
function displayFileList(files) {
    selectedFiles = files;
    const fileList = document.getElementById('file-list');
    const fileItems = document.getElementById('file-items');
    
    if (!fileList || !fileItems) return;
    
    // Clear existing items
    fileItems.innerHTML = '';
    
    if (files.length === 0) {
        fileList.style.display = 'none';
        return;
    }
    
    // Show file list
    fileList.style.display = 'block';
    
    // Add each file
    files.forEach((file, index) => {
        const fileItem = createFileItem(file, index);
        fileItems.appendChild(fileItem);
    });
}

/**
 * Create a file item element
 * @param {File} file - File object
 * @param {number} index - File index
 * @returns {HTMLElement} File item element
 */
function createFileItem(file, index) {
    const item = document.createElement('div');
    item.className = 'file-item';
    item.dataset.index = index;
    
    const fileSize = formatFileSize(file.size);
    const fileType = getFileType(file.name);
    
    item.innerHTML = `
        <div class="file-info">
            <div class="file-name">
                <i class="bi bi-${getFileIcon(fileType)} me-2"></i>
                ${file.name}
            </div>
            <div class="file-size">${fileSize} • ${fileType.toUpperCase()}</div>
        </div>
        <div class="file-actions">
            <button type="button" class="btn btn-sm btn-outline-danger" onclick="removeFile(${index})">
                <i class="bi bi-trash"></i>
            </button>
        </div>
    `;
    
    return item;
}

/**
 * Get file type from filename
 * @param {string} filename - Filename
 * @returns {string} File type
 */
function getFileType(filename) {
    const extension = filename.split('.').pop().toLowerCase();
    return extension;
}

/**
 * Get Bootstrap icon for file type
 * @param {string} fileType - File type
 * @returns {string} Icon name
 */
function getFileIcon(fileType) {
    const iconMap = {
        'wav': 'file-music',
        'mp3': 'file-music',
        'mp4': 'file-play',
        'mkv': 'file-play'
    };
    return iconMap[fileType] || 'file-earmark';
}

/**
 * Format file size in human readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Remove file from selection
 * @param {number} index - File index to remove
 */
function removeFile(index) {
    selectedFiles.splice(index, 1);
    
    // Update file input
    const fileInput = document.getElementById('files');
    const dt = new DataTransfer();
    selectedFiles.forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
    
    // Refresh display
    displayFileList(selectedFiles);
}

/**
 * Upload files and start processing
 * @param {FileList} files - Files to upload
 */
async function uploadFiles(files) {
    try {
        // Show loading state
        setUploadButtonLoading(true);
        
        // Create form data
        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });
        
        // Upload files
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Start monitoring job
            currentJobId = result.job_id;
            showProcessingSection();
            startStatusPolling();
            
            showAlert(`Загружено ${result.files_count} файлов. Обработка началась...`, 'success');
        } else {
            throw new Error(result.error || 'Ошибка загрузки файлов');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showAlert(error.message, 'danger');
    } finally {
        setUploadButtonLoading(false);
    }
}

/**
 * Set loading state for upload button
 * @param {boolean} loading - Loading state
 */
function setUploadButtonLoading(loading) {
    const uploadBtn = document.getElementById('upload-btn');
    if (!uploadBtn) return;
    
    if (loading) {
        uploadBtn.disabled = true;
        uploadBtn.innerHTML = `
            <span class="spinner-border spinner-border-sm me-2" role="status"></span>
            Загрузка...
        `;
    } else {
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = `
            <i class="bi bi-upload me-2"></i>
            Начать обработку
        `;
    }
}

/**
 * Show processing section
 */
function showProcessingSection() {
    hideAllSections();
    
    const processingSection = document.getElementById('processing-section');
    if (processingSection) {
        processingSection.style.display = 'block';
        processingSection.classList.add('fade-in');
    }
    
    // Set job ID
    const jobIdElement = document.getElementById('job-id');
    if (jobIdElement && currentJobId) {
        jobIdElement.textContent = currentJobId;
    }
    
    // Update glitch colors for processing state
    updateGlitchColors('processing');
}

/**
 * Start polling job status
 */
function startStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    statusCheckInterval = setInterval(checkJobStatus, 2000); // Check every 2 seconds
    checkJobStatus(); // Check immediately
}

/**
 * Stop status polling
 */
function stopStatusPolling() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

/**
 * Check job status
 */
async function checkJobStatus() {
    if (!currentJobId) return;
    
    try {
        const response = await fetch(`/status/${currentJobId}`);
        const job = await response.json();
        
        if (response.ok) {
            updateJobStatus(job);
            
            // Stop polling if job is completed or failed
            if (job.status === 'completed') {
                stopStatusPolling();
                showResultsSection(job);
            } else if (job.status === 'error') {
                stopStatusPolling();
                showErrorSection(job.error_message);
            }
        } else {
            throw new Error(job.error || 'Ошибка получения статуса');
        }
    } catch (error) {
        console.error('Status check error:', error);
        // Don't show alert for status errors, just log them
    }
}

/**
 * Update job status in the interface
 * @param {Object} job - Job status object
 */
function updateJobStatus(job) {
    // Update progress bar
    const progressBar = document.getElementById('progress-bar');
    const progressPercentage = document.getElementById('progress-percentage');
    
    if (progressBar && progressPercentage) {
        const percentage = Math.round(job.progress_percentage || 0);
        progressBar.style.width = percentage + '%';
        progressBar.setAttribute('aria-valuenow', percentage);
        progressPercentage.textContent = percentage + '%';
    }
    
    // Update current step
    const currentStep = document.getElementById('current-step');
    if (currentStep) {
        currentStep.textContent = job.current_step || 'Обработка...';
    }
    
    // Update job status
    const jobStatus = document.getElementById('job-status');
    if (jobStatus) {
        jobStatus.textContent = getStatusText(job.status);
    }
    
    // Update processing steps
    updateProcessingSteps(job.status, job.progress_percentage);
}

/**
 * Get human-readable status text
 * @param {string} status - Job status
 * @returns {string} Human-readable status
 */
function getStatusText(status) {
    const statusMap = {
        'pending': 'Ожидание',
        'uploading': 'Загрузка',
        'converting': 'Конвертация',
        'combining': 'Объединение',
        'diarizing': 'Диаризация',
        'transcribing': 'Транскрибация',
        'completed': 'Завершено',
        'error': 'Ошибка'
    };
    return statusMap[status] || status;
}

/**
 * Update processing steps visual state
 * @param {string} status - Current status
 * @param {number} percentage - Progress percentage
 */
function updateProcessingSteps(status, percentage) {
    const steps = ['upload', 'convert', 'diarize', 'transcribe'];
    const statusMap = {
        'uploading': 0,
        'converting': 1,
        'combining': 1,
        'diarizing': 2,
        'transcribing': 3,
        'completed': 4
    };
    
    const currentStepIndex = statusMap[status] || 0;
    
    steps.forEach((step, index) => {
        const stepElement = document.getElementById(`step-${step}`);
        if (stepElement) {
            stepElement.classList.remove('active', 'completed');
            
            if (index < currentStepIndex) {
                stepElement.classList.add('completed');
            } else if (index === currentStepIndex) {
                stepElement.classList.add('active');
            }
        }
    });
}

/**
 * Show results section
 * @param {Object} job - Completed job object
 */
function showResultsSection(job) {
    hideAllSections();
    
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        resultsSection.classList.add('fade-in');
    }
    
    // Update statistics
    updateResultsInfo(job);
    
    // Update glitch colors for completed state
    updateGlitchColors('completed');
    
    showAlert('Транскрибация завершена успешно!', 'success');
}

/**
 * Update results information
 * @param {Object} job - Job object
 */
function updateResultsInfo(job) {
    // Speaker count
    const speakerCount = document.getElementById('speaker-count');
    if (speakerCount) {
        speakerCount.textContent = job.speaker_count || 0;
    }
    
    // Duration
    const duration = document.getElementById('duration');
    if (duration && job.total_duration) {
        duration.textContent = formatDuration(job.total_duration);
    }
    
    // Processing time
    const processingTime = document.getElementById('processing-time');
    if (processingTime && job.processing_time) {
        processingTime.textContent = formatDuration(job.processing_time);
    }
}

/**
 * Format duration in human readable format
 * @param {number} seconds - Duration in seconds
 * @returns {string} Formatted duration
 */
function formatDuration(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}ч ${minutes}м ${secs}с`;
    } else if (minutes > 0) {
        return `${minutes}м ${secs}с`;
    } else {
        return `${secs}с`;
    }
}

/**
 * Show error section
 * @param {string} errorMessage - Error message
 */
function showErrorSection(errorMessage) {
    hideAllSections();
    
    const errorSection = document.getElementById('error-section');
    const errorMessageElement = document.getElementById('error-message');
    
    if (errorSection) {
        errorSection.style.display = 'block';
        errorSection.classList.add('fade-in');
    }
    
    if (errorMessageElement) {
        errorMessageElement.textContent = errorMessage || 'Неизвестная ошибка';
    }
    
    // Update glitch colors for error state
    updateGlitchColors('error');
    
    showAlert('Произошла ошибка при обработке', 'danger');
}

/**
 * Hide all sections
 */
function hideAllSections() {
    const sections = ['upload-section', 'processing-section', 'results-section', 'error-section'];
    sections.forEach(sectionId => {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'none';
            section.classList.remove('fade-in');
        }
    });
}

/**
 * Download result file
 */
function downloadResult() {
    if (currentJobId) {
        window.location.href = `/download/${currentJobId}`;
    }
}

/**
 * Start new job (reset interface)
 */
function startNewJob() {
    // Reset state
    currentJobId = null;
    selectedFiles = [];
    stopStatusPolling();
    
    // Reset file input
    const fileInput = document.getElementById('files');
    if (fileInput) {
        fileInput.value = '';
    }
    
    // Hide file list
    const fileList = document.getElementById('file-list');
    if (fileList) {
        fileList.style.display = 'none';
    }
    
    // Show upload section
    hideAllSections();
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
        uploadSection.style.display = 'block';
        uploadSection.classList.add('fade-in');
    }
    
    // Reset glitch colors to idle state
    updateGlitchColors('idle');
}

/**
 * Cancel current processing
 */
function cancelProcessing() {
    if (currentJobId) {
        if (confirm('Вы уверены, что хотите отменить обработку?')) {
            stopStatusPolling();
            
            // Optionally delete the job
            fetch(`/delete/${currentJobId}`, { method: 'DELETE' })
                .then(() => {
                    showAlert('Обработка отменена', 'info');
                    startNewJob();
                })
                .catch(error => {
                    console.error('Cancel error:', error);
                    startNewJob();
                });
        }
    }
}

/**
 * Show jobs list (placeholder for future implementation)
 */
function showJobsList() {
    // This would show a modal with all jobs
    alert('Список задач - функция в разработке');
}

/**
 * Show alert message
 * @param {string} message - Alert message
 * @param {string} type - Alert type (success, danger, warning, info)
 */
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts-container');
    if (!alertsContainer) return;
    
    const alertId = 'alert-' + Date.now();
    const alertElement = document.createElement('div');
    alertElement.id = alertId;
    alertElement.className = `alert alert-${type} alert-dismissible fade show`;
    alertElement.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    alertsContainer.appendChild(alertElement);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        const alert = document.getElementById(alertId);
        if (alert) {
            alert.remove();
        }
    }, 5000);
}

/**
 * Update file order after sorting
 */
function updateFileOrder() {
    const fileItems = document.querySelectorAll('.file-item');
    const newOrder = [];
    
    fileItems.forEach(item => {
        const index = parseInt(item.dataset.index);
        newOrder.push(selectedFiles[index]);
    });
    
    selectedFiles = newOrder;
    
    // Update file input
    const fileInput = document.getElementById('files');
    const dt = new DataTransfer();
    selectedFiles.forEach(file => dt.items.add(file));
    fileInput.files = dt.files;
} 