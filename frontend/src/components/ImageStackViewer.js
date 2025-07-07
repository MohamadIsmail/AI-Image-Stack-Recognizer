import React, { useState, useEffect } from 'react';

const ImageStackViewer = ({ filenames, resolutions, inferenceResults, onRunInference }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedImages, setSelectedImages] = useState([]);

  useEffect(() => {
    // Load images from the backend
    const loadImages = async () => {
      setLoading(true);
      const imagePromises = filenames.map(async (filename) => {
        try {
          const response = await fetch(`http://localhost:8000/uploaded_images/${filename}`);
          if (response.ok) {
            const blob = await response.blob();
            return {
              src: URL.createObjectURL(blob),
              filename,
              resolution: resolutions[filenames.indexOf(filename)]
            };
          }
          return null;
        } catch (error) {
          console.error(`Error loading image ${filename}:`, error);
          return null;
        }
      });

      const loadedImages = await Promise.all(imagePromises);
      setImages(loadedImages.filter(img => img !== null));
      setLoading(false);
    };

    loadImages();
  }, [filenames, resolutions]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyPress = (event) => {
      if (images.length === 0) return;
      
      switch (event.key) {
        case 'ArrowLeft':
          event.preventDefault();
          prevImage();
          break;
        case 'ArrowRight':
          event.preventDefault();
          nextImage();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [images.length]);

  const nextImage = () => {
    setCurrentIndex((prev) => (prev + 1) % images.length);
  };

  const prevImage = () => {
    setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const goToImage = (index) => {
    setCurrentIndex(index);
  };

  // Selection logic
  const toggleSelectImage = (filename) => {
    setSelectedImages((prev) =>
      prev.includes(filename)
        ? prev.filter((f) => f !== filename)
        : [...prev, filename]
    );
  };

  const selectAll = () => {
    setSelectedImages(images.map((img) => img.filename));
  };

  const deselectAll = () => {
    setSelectedImages([]);
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
        <div className="text-center text-gray-500">
          No images available to display.
        </div>
      </div>
    );
  }

  const currentImage = images[currentIndex];

  return (
    <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
      <h2 className="text-2xl font-semibold text-gray-700 mb-4">
        Image Stack Viewer
      </h2>
      {/* Selection Controls */}
      <div className="mb-4 flex gap-2 items-center">
        <button
          onClick={selectAll}
          className="px-3 py-1 bg-blue-100 text-blue-700 rounded hover:bg-blue-200 text-sm"
        >
          Select All
        </button>
        <button
          onClick={deselectAll}
          className="px-3 py-1 bg-gray-100 text-gray-700 rounded hover:bg-gray-200 text-sm"
        >
          Deselect All
        </button>
        <span className="text-xs text-gray-500 ml-2">
          {selectedImages.length} selected
        </span>
      </div>
      {/* Image Display */}
      <div className="relative mb-6">
        <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
          <img
            src={currentImage.src}
            alt={currentImage.filename}
            className="max-w-full max-h-full object-contain"
          />
        </div>
        
        {/* Navigation Buttons */}
        <button
          onClick={prevImage}
          className="absolute left-4 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-opacity"
        >
          ←
        </button>
        <button
          onClick={nextImage}
          className="absolute right-4 top-1/2 transform -translate-y-1/2 bg-black bg-opacity-50 text-white p-2 rounded-full hover:bg-opacity-75 transition-opacity"
        >
          →
        </button>
      </div>

      {/* Image Info */}
      <div className="mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">Filename:</span>
            <p className="text-gray-600">{currentImage.filename}</p>
          </div>
          <div>
            <span className="font-medium text-gray-700">Resolution:</span>
            <p className="text-gray-600">{currentImage.resolution[0]} × {currentImage.resolution[1]}</p>
          </div>
          <div>
            <span className="font-medium text-gray-700">Position:</span>
            <p className="text-gray-600">{currentIndex + 1} of {images.length}</p>
          </div>
        </div>
      </div>

      {/* Inference Results for Current Image */}
      {inferenceResults && (
        <div className="mb-6">
          <h3 className="text-lg font-medium text-gray-700 mb-3">Inference Result</h3>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            {(() => {
              const result = inferenceResults.find(r => r.filename === currentImage.filename);
              if (result) {
                return (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <span className="font-medium text-gray-700">Class:</span>
                      <p className="text-blue-600 font-semibold">{result.class}</p>
                    </div>
                    <div>
                      <span className="font-medium text-gray-700">Confidence:</span>
                      <p className="text-blue-600 font-semibold">{Math.round(result.confidence * 100)}%</p>
                    </div>
                  </div>
                );
              } else {
                return (
                  <p className="text-gray-500">No inference result available for this image</p>
                );
              }
            })()}
          </div>
        </div>
      )}

      {/* Run Inference Button */}
      <div className="mb-6 flex justify-end">
        <button
          onClick={() => onRunInference(selectedImages)}
          disabled={selectedImages.length === 0}
          className={`px-4 py-2 rounded bg-blue-600 text-white font-semibold shadow hover:bg-blue-700 transition-colors disabled:bg-gray-300 disabled:cursor-not-allowed`}
        >
          Run Inference on Selected
        </button>
      </div>

      {/* Thumbnail Navigation */}
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-700 mb-3">Thumbnails</h3>
        <div className="flex gap-2 overflow-x-auto pb-2">
          {images.map((image, index) => (
            <div key={index} className="relative flex-shrink-0">
              <button
                onClick={() => goToImage(index)}
                className={`w-20 h-20 rounded-lg overflow-hidden border-2 transition-colors focus:outline-none ${
                  index === currentIndex
                    ? 'border-blue-500'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <img
                  src={image.src}
                  alt={image.filename}
                  className="w-full h-full object-cover"
                />
                {/* Selection Checkbox Overlay */}
                <input
                  type="checkbox"
                  checked={selectedImages.includes(image.filename)}
                  onChange={() => toggleSelectImage(image.filename)}
                  className="absolute top-1 left-1 w-4 h-4 bg-white bg-opacity-80 rounded border border-gray-400 cursor-pointer z-10"
                  onClick={e => e.stopPropagation()}
                  aria-label={`Select image ${image.filename}`}
                />
              </button>
            </div>
          ))}
        </div>
      </div>

      {/* Keyboard Navigation Info */}
      <div className="text-xs text-gray-500 text-center">
        Use arrow keys or click the navigation buttons to browse through images
      </div>
    </div>
  );
};

export default ImageStackViewer; 