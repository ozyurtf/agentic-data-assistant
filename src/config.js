// Configuration file for the application
// Modify these values as needed

export const config = {
    // API Configuration
    API_BASE_URL: process.env.VUE_APP_API_BASE_URL || 'http://127.0.0.1:8001',
    
    // Other configuration options can be added here
    // UPLOAD_MAX_SIZE: 100 * 1024 * 1024, // 100MB
    // TIMEOUT: 30000, // 30 seconds
}

// Export individual values for convenience
export const API_BASE_URL = config.API_BASE_URL