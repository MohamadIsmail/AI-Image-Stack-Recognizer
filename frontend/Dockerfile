# Stage 1: Build the React app
FROM node:18-alpine AS build

# Set working directory inside container
WORKDIR /app
COPY package.json ./
COPY package-lock.json ./

# Install dependencies
RUN npm install --legacy-peer-deps

# Copy the rest of the app code
COPY . .

# Build the production-ready app
RUN npm run build

# Stage 2: Build the Production image
FROM nginx:alpine

# Copy build files to nginx served folder
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"] 
