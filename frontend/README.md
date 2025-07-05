# AI Image Stack Recognizer - Frontend

A React application with Tailwind CSS for uploading and viewing image stacks interactively.

## Features

- **File Upload**: Select and upload ZIP files containing PNG images
- **Interactive Viewer**: Browse through uploaded images with navigation controls
- **Image Information**: Display filename, resolution, and position in stack
- **Thumbnail Navigation**: Quick access to any image in the stack
- **Responsive Design**: Works on desktop and mobile devices

## Setup

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start the development server**:
   ```bash
   npm start
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

## Usage

1. **Upload Images**:
   - Click "Select ZIP File" to choose a ZIP file containing PNG images
   - Click "Upload Images" to upload the file to the backend
   - Wait for the upload to complete

2. **View Images**:
   - After successful upload, the Image Stack Viewer will appear
   - Use the arrow buttons or arrow keys to navigate through images
   - Click on thumbnails to jump to specific images
   - View image information including filename, resolution, and position

## API Endpoints

The frontend communicates with the backend API:

- `POST /upload` - Upload ZIP file containing images
- `GET /uploaded_images/{filename}` - Retrieve uploaded images

## Development

- **Port**: Runs on `http://localhost:3000`
- **Backend**: Expects backend to run on `http://localhost:8000`
- **Proxy**: Configured to proxy API requests to the backend

## Technologies Used

- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **Create React App** - Build tool 