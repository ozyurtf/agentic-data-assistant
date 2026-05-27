// Configuration file for the application
// Modify these values as needed

export const config = {
    // Same-origin defaults — requests are proxied through Vue's dev server
    // (see /config/index.js proxyTable). Override via env vars only if you need
    // to point the browser at a different host (e.g., production build).
    API_BASE_URL: process.env.VUE_APP_API_BASE_URL || '',

    CHATBOT_URL: process.env.VUE_APP_CHATBOT_URL || '/chatbot'

    // Other configuration options can be added here
    // UPLOAD_MAX_SIZE: 100 * 1024 * 1024, // 100MB
    // TIMEOUT: 30000, // 30 seconds
}

// Export individual values for convenience
export const API_BASE_URL = config.API_BASE_URL
export const CHATBOT_URL = config.CHATBOT_URL
