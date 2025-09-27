// Main JavaScript for IDMS

// Sidebar functionality
const sidebar = {
    init: function() {
        this.setupToggle();
        this.setupOverlay();
        this.setupResponsive();
    },

    setupToggle: function() {
        const toggleBtn = document.getElementById('sidebar-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', this.toggle);
        }
    },

    setupOverlay: function() {
        const overlay = document.getElementById('sidebar-overlay');
        if (overlay) {
            overlay.addEventListener('click', this.close);
        }
    },

    setupResponsive: function() {
        // Handle window resize
        window.addEventListener('resize', () => {
            if (window.innerWidth >= 1024) {
                // On desktop, ensure sidebar is visible and remove mobile classes
                const sidebar = document.getElementById('sidebar');
                const overlay = document.getElementById('sidebar-overlay');
                
                if (sidebar) {
                    sidebar.classList.remove('-translate-x-full');
                }
                if (overlay) {
                    overlay.classList.add('hidden');
                }
            }
        });
    },

    toggle: function() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.toggle('-translate-x-full');
            overlay.classList.toggle('hidden');
        }
    },

    close: function() {
        const sidebar = document.getElementById('sidebar');
        const overlay = document.getElementById('sidebar-overlay');
        
        if (sidebar && overlay) {
            sidebar.classList.add('-translate-x-full');
            overlay.classList.add('hidden');
        }
    }
};

// Utility functions
const utils = {
    // Format file size
    formatFileSize: function(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    // Show toast notification
    showToast: function(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(toast);
            }, 300);
        }, 3000);
    },

    // Copy to clipboard
    copyToClipboard: function(text) {
        navigator.clipboard.writeText(text).then(() => {
            this.showToast('Copied to clipboard!', 'success');
        }).catch(() => {
            this.showToast('Failed to copy to clipboard', 'error');
        });
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
};

// File upload handling
const fileUpload = {
    init: function() {
        this.setupDragAndDrop();
        this.setupFileInputs();
    },

    setupDragAndDrop: function() {
        const dropZones = document.querySelectorAll('[data-drop-zone]');
        
        dropZones.forEach(zone => {
            zone.addEventListener('dragover', (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });

            zone.addEventListener('dragleave', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
            });

            zone.addEventListener('drop', (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
                
                const files = Array.from(e.dataTransfer.files);
                this.handleFiles(files, zone);
            });
        });
    },

    setupFileInputs: function() {
        const fileInputs = document.querySelectorAll('input[type="file"]');
        
        fileInputs.forEach(input => {
            input.addEventListener('change', (e) => {
                const files = Array.from(e.target.files);
                this.handleFiles(files, input);
            });
        });
    },

    handleFiles: function(files, target) {
        // Validate file types and sizes
        const validFiles = this.validateFiles(files);
        
        if (validFiles.length === 0) {
            utils.showToast('No valid files selected', 'error');
            return;
        }

        // Update UI to show selected files
        this.updateFileList(validFiles, target);
    },

    validateFiles: function(files) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = [
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.oasis.opendocument.text',
            'text/csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.ms-excel',
            'image/png',
            'image/jpeg',
            'image/jpg',
            'application/zip',
            'application/x-7z-compressed',
            'text/plain',
            'application/json',
            'application/x-yaml',
            'text/yaml'
        ];

        return files.filter(file => {
            if (file.size > maxSize) {
                utils.showToast(`File ${file.name} is too large (max 100MB)`, 'error');
                return false;
            }
            
            if (!allowedTypes.includes(file.type)) {
                utils.showToast(`File type ${file.type} is not supported`, 'error');
                return false;
            }
            
            return true;
        });
    },

    updateFileList: function(files, target) {
        const container = target.closest('.file-upload-container');
        if (!container) return;

        let fileList = container.querySelector('.file-list');
        if (!fileList) {
            fileList = document.createElement('div');
            fileList.className = 'file-list mt-4';
            container.appendChild(fileList);
        }

        fileList.innerHTML = '';
        
        files.forEach(file => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-list-item';
            fileItem.innerHTML = `
                <i class="fas fa-file ${this.getFileIcon(file.type)}"></i>
                <div class="flex-1 ml-3">
                    <div class="text-sm font-medium text-gray-900">${file.name}</div>
                    <div class="text-xs text-gray-500">${utils.formatFileSize(file.size)}</div>
                </div>
                <button type="button" class="text-red-500 hover:text-red-700" onclick="this.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            `;
            fileList.appendChild(fileItem);
        });
    },

    getFileIcon: function(mimeType) {
        const iconMap = {
            'application/pdf': 'fa-file-pdf text-red-500',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'fa-file-word text-blue-500',
            'application/vnd.oasis.opendocument.text': 'fa-file-alt text-orange-500',
            'text/csv': 'fa-file-csv text-green-500',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'fa-file-excel text-green-500',
            'application/vnd.ms-excel': 'fa-file-excel text-green-500',
            'image/png': 'fa-file-image text-purple-500',
            'image/jpeg': 'fa-file-image text-purple-500',
            'image/jpg': 'fa-file-image text-purple-500',
            'application/zip': 'fa-file-archive text-gray-500',
            'application/x-7z-compressed': 'fa-file-archive text-gray-500',
            'text/plain': 'fa-file-alt text-gray-500',
            'application/json': 'fa-file-code text-yellow-500',
            'application/x-yaml': 'fa-file-code text-yellow-500',
            'text/yaml': 'fa-file-code text-yellow-500'
        };

        return iconMap[mimeType] || 'fa-file text-gray-500';
    }
};

// Form handling
const formHandler = {
    init: function() {
        this.setupFormValidation();
        this.setupFormSubmission();
    },

    setupFormValidation: function() {
        const forms = document.querySelectorAll('form');
        
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                }
            });
        });
    },

    setupFormSubmission: function() {
        const forms = document.querySelectorAll('form[data-async]');
        
        forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                e.preventDefault();
                this.submitFormAsync(form);
            });
        });
    },

    validateForm: function(form) {
        const requiredFields = form.querySelectorAll('[required]');
        let isValid = true;

        requiredFields.forEach(field => {
            if (!field.value.trim()) {
                this.showFieldError(field, 'This field is required');
                isValid = false;
            } else {
                this.clearFieldError(field);
            }
        });

        return isValid;
    },

    showFieldError: function(field, message) {
        const errorElement = field.parentNode.querySelector('.field-error');
        if (errorElement) {
            errorElement.textContent = message;
        } else {
            const error = document.createElement('div');
            error.className = 'field-error text-red-500 text-sm mt-1';
            error.textContent = message;
            field.parentNode.appendChild(error);
        }
        
        field.classList.add('border-red-500');
    },

    clearFieldError: function(field) {
        const errorElement = field.parentNode.querySelector('.field-error');
        if (errorElement) {
            errorElement.remove();
        }
        
        field.classList.remove('border-red-500');
    },

    submitFormAsync: function(form) {
        const submitBtn = form.querySelector('button[type="submit"]');
        const originalText = submitBtn.innerHTML;
        
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Processing...';

        const formData = new FormData(form);
        
        fetch(form.action, {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(html => {
            document.body.innerHTML = html;
        })
        .catch(error => {
            utils.showToast('An error occurred: ' + error.message, 'error');
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        });
    }
};

// Results table functionality
const resultsTable = {
    init: function() {
        this.setupSorting();
        this.setupFiltering();
        this.setupRowActions();
    },

    setupSorting: function() {
        const headers = document.querySelectorAll('th[data-sortable]');
        
        headers.forEach(header => {
            header.addEventListener('click', () => {
                this.sortTable(header);
            });
        });
    },

    setupFiltering: function() {
        const filterInputs = document.querySelectorAll('[data-filter]');
        
        filterInputs.forEach(input => {
            input.addEventListener('input', utils.debounce((e) => {
                this.filterTable(e.target);
            }, 300));
        });
    },

    setupRowActions: function() {
        // Copy filename functionality
        window.copyToClipboard = function(filename) {
            utils.copyToClipboard(filename);
        };

        // Toggle reasoning functionality
        window.toggleReasoning = function(filename) {
            const element = document.getElementById('reasoning-' + filename);
            if (element) {
                element.classList.toggle('hidden');
            }
        };
    },

    sortTable: function(header) {
        const table = header.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const columnIndex = Array.from(header.parentNode.children).indexOf(header);
        const isAscending = header.classList.contains('sort-asc');

        // Remove existing sort classes
        header.parentNode.querySelectorAll('th').forEach(th => {
            th.classList.remove('sort-asc', 'sort-desc');
        });

        // Add new sort class
        header.classList.add(isAscending ? 'sort-desc' : 'sort-asc');

        // Sort rows
        rows.sort((a, b) => {
            const aText = a.children[columnIndex].textContent.trim();
            const bText = b.children[columnIndex].textContent.trim();
            
            if (isAscending) {
                return bText.localeCompare(aText);
            } else {
                return aText.localeCompare(bText);
            }
        });

        // Reorder rows in DOM
        rows.forEach(row => tbody.appendChild(row));
    },

    filterTable: function(input) {
        const filterValue = input.value.toLowerCase();
        const table = input.closest('.table-container').querySelector('table');
        const rows = table.querySelectorAll('tbody tr');

        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            if (text.includes(filterValue)) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }
};

// Initialize everything when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    sidebar.init();
    fileUpload.init();
    formHandler.init();
    resultsTable.init();
    
    // Add fade-in animation to main content
    const mainContent = document.querySelector('main');
    if (mainContent) {
        mainContent.classList.add('fade-in-up');
    }
    
    // Initialize tooltips if needed
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(tooltip => {
        tooltip.addEventListener('mouseenter', function() {
            // Simple tooltip implementation
            const text = this.getAttribute('data-tooltip');
            const tooltipElement = document.createElement('div');
            tooltipElement.className = 'absolute z-50 px-2 py-1 text-sm text-white bg-gray-900 rounded shadow-lg';
            tooltipElement.textContent = text;
            tooltipElement.style.top = this.offsetTop - 30 + 'px';
            tooltipElement.style.left = this.offsetLeft + 'px';
            
            document.body.appendChild(tooltipElement);
            this._tooltipElement = tooltipElement;
        });
        
        tooltip.addEventListener('mouseleave', function() {
            if (this._tooltipElement) {
                document.body.removeChild(this._tooltipElement);
                this._tooltipElement = null;
            }
        });
    });
});
