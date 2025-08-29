'use strict'
require('dotenv').config()
const { merge } = require('webpack-merge')
const prodEnv = require('./prod.env')

module.exports = merge(prodEnv, {
  NODE_ENV: '"development"',
  VUE_APP_CESIUM_TOKEN: JSON.stringify(process.env.VUE_APP_CESIUM_TOKEN || ''),
  VUE_APP_CESIUM_RESOURCE_ID: JSON.stringify(process.env.VUE_APP_CESIUM_RESOUR_ID || 3),
  VUE_APP_MAPTILER_KEY: JSON.stringify(process.env.VUE_APP_MAPTILER_KEY || ''),
  // New environment variable format
  VUE_API_HOST: JSON.stringify(process.env.VUE_API_HOST || 'localhost'),
  VUE_API_PORT: JSON.stringify(process.env.VUE_API_PORT || '8001'),
  CHATBOT_HOST: JSON.stringify(process.env.CHATBOT_HOST || 'localhost'),
  CHATBOT_PORT: JSON.stringify(process.env.CHATBOT_PORT || '8000')
})