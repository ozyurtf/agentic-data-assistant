'use strict'
module.exports = {
  NODE_ENV: '"production"',
  VUE_APP_CESIUM_TOKEN: JSON.stringify(process.env.VUE_APP_CESIUM_TOKEN || ''),
  VUE_APP_CESIUM_RESOURCE_ID: JSON.stringify(process.env.VUE_APP_CESIUM_RESOURCE_ID || 3),
  VUE_APP_MAPTILER_KEY: JSON.stringify(process.env.VUE_APP_MAPTILER_KEY || 'o3JREHNnXex8WSPPm2BU'),
  // Environment variables for browser access
  VUE_APP_API_BASE_URL: JSON.stringify(process.env.VUE_APP_API_BASE_URL || 'http://localhost:8001'),
  VUE_APP_CHATBOT_URL: JSON.stringify(process.env.VUE_APP_CHATBOT_URL || '/chatbot'),
  // Legacy format for server-side access
  API_HOST: JSON.stringify(process.env.API_HOST || 'localhost'),
  API_PORT: JSON.stringify(process.env.API_PORT || '8001'),
  CHATBOT_HOST: JSON.stringify(process.env.CHATBOT_HOST || 'localhost'),
  CHATBOT_PORT: JSON.stringify(process.env.CHATBOT_PORT || '8000')
}