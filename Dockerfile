## STAGE 1: BUILDING

# Use node:20 as the base image (a ready-made image with Node.js 20 and npm installed)
FROM node:20 AS build

# Create app directory
WORKDIR /usr/src/app

# Install git (needed for submodules)
RUN apt-get update && apt-get install -y git

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Bundle app source
COPY . .

# Initialize git submodules
# The chain: 
# 1) The project includes .gitmodules file
# 2) COPY . . brings it into the image
# 3) git submodule update reads it, goes to that GitHub URL, downloads JsDataflashParser's files into src/tools/parsers/JsDataflashParser/
# 4) Now the build has the parser code it depends on at src/tools/parsers/JsDataflashParser/ in the container, rather than an empty folder.
RUN git init
RUN git submodule init
RUN git submodule update

# Run the update-browserslist-db as suggested in the warning
RUN npx update-browserslist-db@latest

# Configure git to trust the mounted directory
RUN git config --global --add safe.directory /usr/src/app

# 
ARG VUE_APP_CESIUM_TOKEN
ARG VUE_APP_CESIUM_RESOURCE_ID
ARG VUE_APP_GOOGLE_MAPS_KEY
ARG VUE_APP_MAPTILER_KEY

ENV VUE_APP_CESIUM_TOKEN=$VUE_APP_CESIUM_TOKEN
ENV VUE_APP_CESIUM_RESOURCE_ID=$VUE_APP_CESIUM_RESOURCE_ID
ENV VUE_APP_GOOGLE_MAPS_KEY=$VUE_APP_GOOGLE_MAPS_KEY
ENV VUE_APP_MAPTILER_KEY=$VUE_APP_MAPTILER_KEY

# Run the build script defined in package.json (the "build" entry) 
# 1) Take all the source code (e.g, .vue files, JavaScript, CSS)
# 2) Bundle/compile it into plain static files (HTML, JS, CSS) that a browser can actually use
# 3) Place the static files in the dist/ folder.
RUN npm run build

## STAGE 2: SERVING

# Start a fresh image from the nginx base
FROM nginx:alpine

# Copy the static files created by the Stage 1 in the dist/ folder to /usr/share/nginx/html path (default path nginx serves files from)
COPY --from=build /usr/src/app/dist /usr/share/nginx/html/dist/

# Copy the nginx.conf file into the image at /etc/nginx/conf.d/default.conf (the location where nginx looks for its server configuration)
COPY nginx.conf.template /etc/nginx/templates/default.conf.template

# nginx forks itself into the background and the original process exits.
# This results in a dead container.
# So we run nginx with "daemon off;" to keep it in the foreground, and 
# this keeps the container alive and serving.
CMD ["nginx", "-g", "daemon off;"]
