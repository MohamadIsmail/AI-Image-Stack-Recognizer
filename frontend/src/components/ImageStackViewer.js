import React, { useState, useEffect } from 'react';

const ImageStackViewer = ({ filenames, resolutions }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load images from the backend
    const loadImages = async () => {
      setLoading(true);
      const imagePromises = filenames.map(async (filename) => {
        try {
          const response = await fetch(`/uploaded_images/${filename}`);
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

  const nextImage = () => {
    setCurrentIndex((prev) => (prev + 1) % images.length);
  };

  const prevImage = () => {
    setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);
  };

  const goToImage = (index) => {
    setCurrentIndex(index);
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

      {/* Thumbnail Navigation */}
      <div className="mb-4">
        <h3 className="text-lg font-medium text-gray-700 mb-3">Thumbnails</h3>
        <div className="flex gap-2 overflow-x-auto pb-2">
          {images.map((image, index) => (
            <button
              key={index}
              onClick={() => goToImage(index)}
              className={`flex-shrink-0 w-20 h-20 rounded-lg overflow-hidden border-2 transition-colors ${
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
            </button>
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