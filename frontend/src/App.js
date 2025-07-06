import React, { useState } from 'react';
import axios from 'axios';
import ImageStackViewer from './components/ImageStackViewer';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  const [inferenceResults, setInferenceResults] = useState(null);
  const [runningInference, setRunningInference] = useState(false);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/zip') {
      setSelectedFile(file);
      setError(null);
    } else {
      setError('Please select a valid .zip file');
      setSelectedFile(null);
    }
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

  const handleInference = async () => {
    if (!uploadResult || !uploadResult.filenames) {
      setError('No images available for inference');
      return;
    }

    setRunningInference(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:8000/predict_opt', {
        filenames: uploadResult.filenames
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
        <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6 mb-8">
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">
            Upload Image Stack
          </h2>
          
          <div className="space-y-4">
            {/* File Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Select ZIP File
              </label>
              <input
                id="file-input"
                type="file"
                accept=".zip"
                onChange={handleFileSelect}
                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />
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
          />
        )}

        {/* Inference Section */}
        {uploadResult && (
          <div className="max-w-2xl mx-auto bg-white rounded-lg shadow-md p-6 mb-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">
              Run Inference
            </h2>
            
            <div className="space-y-4">
              <button
                onClick={handleInference}
                disabled={runningInference}
                className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
                  runningInference
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-green-600 text-white hover:bg-green-700'
                }`}
              >
                {runningInference ? 'Running Inference...' : 'Run Inference on All Images'}
              </button>

              {/* Inference Results */}
              {inferenceResults && (
                <div className="bg-blue-50 border border-blue-200 text-blue-700 px-4 py-3 rounded-md">
                  <p className="font-medium mb-2">Inference Results:</p>
                  <div className="space-y-2">
                    {inferenceResults.map((result, index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="font-medium">{result.filename}:</span>
                        <span>
                          {result.class} ({Math.round(result.confidence * 100)}%)
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App; 