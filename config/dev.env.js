'use strict'
require('dotenv').config()
const { merge } = require('webpack-merge')
const prodEnv = require('./prod.env')

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  VUE_APP_CESIUM_TOKEN: JSON.stringify(process.env.VUE_APP_CESIUM_TOKEN || ''),
  VUE_APP_CESIUM_RESOURCE_ID: JSON.stringify(process.env.VUE_APP_CESIUM_RESOUR_ID || 3),
  VUE_APP_MAPTILER_KEY: JSON.stringify(process.env.VUE_APP_MAPTILER_KEY || ''),
  // Environment variables for browser access - use explicit values from .env if available
  VUE_APP_API_BASE_URL: JSON.stringify(process.env.VUE_APP_API_BASE_URL || `http://${process.env.API_HOST || 'localhost'}:${process.env.API_PORT || '8001'}`),
  VUE_APP_CHATBOT_URL: JSON.stringify(process.env.VUE_APP_CHATBOT_URL || `http://${process.env.CHATBOT_HOST || 'localhost'}:${process.env.CHATBOT_PORT || '8000'}`),
  // Legacy format for server-side access
  API_HOST: JSON.stringify(process.env.API_HOST || 'localhost'),
  API_PORT: JSON.stringify(process.env.API_PORT || '8001'),
  CHATBOT_HOST: JSON.stringify(process.env.CHATBOT_HOST || 'localhost'),
  CHATBOT_PORT: JSON.stringify(process.env.CHATBOT_PORT || '8000')
})