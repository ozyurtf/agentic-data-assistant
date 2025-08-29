// Configuration file for the application
// Modify these values as needed

export const config = {
    // API Configuration - construct URL from host and port
    API_BASE_URL: process.env.VUE_API_HOST && process.env.VUE_API_PORT 
        ? `http://${process.env.VUE_API_HOST}:${process.env.VUE_API_PORT}`
        : process.env.VUE_APP_API_BASE_URL || 'http://127.0.0.1:8001',
    
    // Chatbot Configuration - construct URL from host and port
    CHATBOT_URL: process.env.CHATBOT_HOST && process.env.CHATBOT_PORT
        ? `http://${process.env.CHATBOT_HOST}:${process.env.CHATBOT_PORT}`
        : process.env.VUE_APP_CHATBOT_URL || 'http://localhost:8000',
    
    // Other configuration options can be added here
    // UPLOAD_MAX_SIZE: 100 * 1024 * 1024, // 100MB
    // TIMEOUT: 30000, // 30 seconds
}

// Export individual values for convenience
export const API_BASE_URL = config.API_BASE_URL
export const CHATBOT_URL = config.CHATBOT_URL