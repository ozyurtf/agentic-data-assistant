### Stage 1: Building
FROM node:20 AS build

# Define build argument for UI port
ARG UI_PORT=443
ENV UI_PORT=${UI_PORT}
ENV PORT=${UI_PORT}

# Create app directory
WORKDIR /usr/src/app

# Install git (needed for submodules)
RUN apt-get update && apt-get install -y git

# Create a non-root user
RUN useradd -m -u 2000 nodeuser

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Bundle app source
COPY . .

# Initialize and update git submodules
RUN git init
RUN git submodule init
RUN git submodule update

# Run the update-browserslist-db as suggested in the warning
RUN npx update-browserslist-db@latest

# Change ownership of the app directory to the non-root user
RUN chown -R nodeuser:nodeuser /usr/src/app

# Switch to non-root user
USER nodeuser

# Configure git to trust the mounted directory
RUN git config --global --add safe.directory /usr/src/app

EXPOSE ${UI_PORT}
RUN npm run build


### Stage 2: Serving
FROM nginx:alpine
COPY --from=build /usr/src/app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80 443
CMD ["nginx", "-g", "daemon off;"]
