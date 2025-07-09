import React, { useState, useRef } from 'react';
import axios from 'axios';
import ImageStackViewer from './components/ImageStackViewer';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [runningInference, setRunningInference] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Add document-level drag event listeners
  React.useEffect(() => {
    const handleDocumentDragOver = (e) => {
      e.preventDefault();
    };

    const handleDocumentDrop = (e) => {
      e.preventDefault();
      console.log('Document drop event');
    };

    document.addEventListener('dragover', handleDocumentDragOver);
    document.addEventListener('drop', handleDocumentDrop);

    return () => {
      document.removeEventListener('dragover', handleDocumentDragOver);
      document.removeEventListener('drop', handleDocumentDrop);
    };
  }, []);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === 'application/zip' || file.name.endsWith('.zip'))) {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please select a valid .zip file');
      setSelectedFile(null);
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag enter - target:', e.target.className);
    setIsDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag leave - target:', e.target.className);
    // Only set drag over to false if we're leaving the drop zone entirely
    if (!e.currentTarget.contains(e.relatedTarget)) {
      setIsDragOver(false);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    console.log('Drag over');
    // This is required to allow dropping
    e.dataTransfer.dropEffect = 'copy';
  };

  const handleDrop = (e) => {
    console.log('=== DROP EVENT START ===');
    console.log('Event target:', e.target);
    console.log('Event currentTarget:', e.currentTarget);
    console.log('Event type:', e.type);
    
    e.preventDefault();
    e.stopPropagation();
    console.log('Drop event triggered - target:', e.target.className);
    setIsDragOver(false);
    
    const files = e.dataTransfer.files;
    console.log('Dropped files:', files);
    console.log('Files length:', files.length);
    
    if (files.length > 0) {
      const file = files[0];
      console.log('File type:', file.type, 'File name:', file.name);
      if (file.type === 'application/zip' || file.name.endsWith('.zip')) {
        setSelectedFile(file);
        setError(null);
        console.log('File accepted:', file.name);
      } else {
        setError('Please drop a valid .zip file');
        setSelectedFile(null);
        console.log('Invalid file type rejected');
      }
    } else {
      console.log('No files in drop event');
    }
    console.log('=== DROP EVENT END ===');
  };

  // Test click handler to verify the drop zone is working
  const handleDropZoneClick = () => {
    console.log('Drop zone clicked - this confirms the element is working');
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setUploading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setUploadResult(response.data);
      setSelectedFile(null);
      // Reset file input
      document.getElementById('file-input').value = '';
    } catch (err) {
      setError(err.response?.data?.error || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const handleInferenceSelected = async (selectedFilenames) => {
    if (!selectedFilenames || selectedFilenames.length === 0) {
      setError('No images selected for inference');
      return;
    }

    setRunningInference(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/predict_opt_batch', {
        filenames: selectedFilenames
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      setInferenceResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Inference failed. Please try again.');
    } finally {
      setRunningInference(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-4xl font-bold text-center text-gray-800 mb-8">
          AI Image Stack Recognizer
        </h1>
        
        {/* Upload Section */}
        <div 
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={handleDropZoneClick}
          className={`max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6 mb-8 transition-colors ${
            isDragOver 
              ? 'border-2 border-blue-500 bg-blue-50' 
              : 'border-2 border-transparent'
          }`}
          style={{ minHeight: '200px' }}
        >
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">
            Upload Image Stack
          </h2>
          
          <div className="space-y-4">
            {/* File Selection */}
            <div 
              className={`border-2 border-dashed rounded-lg p-6 transition-colors relative ${
                isDragOver 
                  ? 'border-blue-500 bg-blue-50' 
                  : 'border-gray-300 hover:border-gray-400'
              }`}
              style={{ minHeight: '120px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}
              onDragEnter={(e) => e.preventDefault()}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => e.preventDefault()}
            >
              {/* Overlay for drag and drop */}
              {isDragOver && (
                <div 
                  className="absolute inset-0 bg-blue-50 border-2 border-blue-500 rounded-lg flex items-center justify-center z-10"
                  style={{ pointerEvents: 'none' }}
                >
                  <p className="text-lg font-medium text-blue-600">
                    Drop your .zip file here!
                  </p>
                </div>
              )}
              
              <label 
                className="block text-sm font-medium text-gray-700 mb-2"
                style={{ pointerEvents: isDragOver ? 'none' : 'auto' }}
              >
                Select ZIP File
              </label>
              <input
                ref={fileInputRef}
                id="file-input"
                type="file"
                accept=".zip"
                onChange={handleFileSelect}
                style={{ pointerEvents: isDragOver ? 'none' : 'auto' }}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
              <p 
                className="mt-2 text-sm text-gray-500"
                style={{ pointerEvents: isDragOver ? 'none' : 'auto' }}
              >
                Or drag and drop a .zip file here
              </p>
              {selectedFile && (
                <div 
                  className="mt-2 p-2 bg-green-50 border border-green-200 rounded"
                  style={{ pointerEvents: isDragOver ? 'none' : 'auto' }}
                >
                  <p className="text-sm text-green-700">
                    <strong>Selected file:</strong> {selectedFile.name}
                  </p>
                </div>
              )}
            </div>

            {/* Upload Button */}
            <button
              onClick={handleUpload}
              disabled={!selectedFile || uploading}
              className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
                !selectedFile || uploading
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {uploading ? 'Uploading...' : 'Upload Images'}
            </button>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-md">
                {error}
              </div>
            )}

            {/* Upload Result */}
            {uploadResult && (
              <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded-md">
                <p className="font-medium">Upload successful!</p>
                <p>Number of images: {uploadResult.num_images}</p>
                <p>Filenames: {uploadResult.filenames.join(', ')}</p>
              </div>
            )}
          </div>
        </div>

        {/* Image Stack Viewer */}
        {uploadResult && (
          <ImageStackViewer 
            filenames={uploadResult.filenames}
            resolutions={uploadResult.resolutions}
            inferenceResults={inferenceResults}
            onRunInference={handleInferenceSelected}
          />
        )}
      </div>
    </div>
  );
}

export default App; 