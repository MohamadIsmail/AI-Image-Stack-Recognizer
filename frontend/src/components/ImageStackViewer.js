import React, { useState, useEffect } from 'react';

const ImageStackViewer = ({ filenames, resolutions, inferenceResults, onRunInference }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedImages, setSelectedImages] = useState([]);
  // Zoom state
  const [zoom, setZoom] = useState(1);
  // Rotate state
  const [rotate, setRotate] = useState(0);
  const handleRotateLeft = () => setRotate((r) => (r - 90 + 360) % 360);
  const handleRotateRight = () => setRotate((r) => (r + 90) % 360);
  const handleRotateReset = () => setRotate(0);

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

  // Zoom handlers
  const handleZoomIn = () => setZoom((z) => Math.min(z + 0.05, 5));
  const handleZoomOut = () => setZoom((z) => Math.max(z - 0.05, 0.25));
  const handleResetZoom = () => setZoom(1);
  const handleSliderChange = (e) => setZoom(Number(e.target.value));
  const handleZoomInputChange = (e) => {
    let val = parseFloat(e.target.value.replace(/[^0-9.]/g, ''));
    if (isNaN(val)) val = 100;
    setZoom(Math.max(0.25, Math.min(val / 100, 5)));
  };
  const handleZoomInputBlur = (e) => {
    if (!e.target.value) setZoom(1);
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
      {/* Main active image display (for keyboard navigation, zoom, rotate, etc.) */}
      <div className="relative mb-6">
        {/* Zoom & Rotate Controls */}
        <div className="absolute top-2 right-2 z-10 flex gap-2 items-center bg-white bg-opacity-80 rounded p-2 shadow">
          {/* Rotate Left */}
          <button
            onClick={handleRotateLeft}
            className="px-2 py-1 text-lg font-bold bg-gray-200 rounded hover:bg-gray-300"
            title="Rotate Left"
          >
            ⟲
          </button>
          {/* Rotate Right */}
          <button
            onClick={handleRotateRight}
            className="px-2 py-1 text-lg font-bold bg-gray-200 rounded hover:bg-gray-300"
            title="Rotate Right"
          >
            ⟳
          </button>
          {/* Zoom Out Button */}
          <button
            onClick={handleZoomOut}
            className="px-2 py-1 text-lg font-bold bg-gray-200 rounded hover:bg-gray-300 flex items-center"
            title="Zoom Out"
            style={{ minWidth: 32 }}
          >
            <span role="img" aria-label="Zoom Out">🔍−</span>
          </button>
          {/* Zoom Slider */}
          <input
            type="range"
            min={0.25}
            max={5}
            step={0.01}
            value={zoom}
            onChange={handleSliderChange}
            className="w-32 mx-2 accent-blue-600"
            title="Zoom Slider"
          />
          {/* Zoom In Button */}
          <button
            onClick={handleZoomIn}
            className="px-2 py-1 text-lg font-bold bg-gray-200 rounded hover:bg-gray-300 flex items-center"
            title="Zoom In"
            style={{ minWidth: 32 }}
          >
            <span role="img" aria-label="Zoom In">🔍＋</span>
          </button>
          {/* Zoom Percentage Input */}
          <input
            type="text"
            value={`${Math.round(zoom * 100)}`}
            onChange={handleZoomInputChange}
            onBlur={handleZoomInputBlur}
            className="w-14 text-center border border-gray-300 rounded px-1 py-0.5 mx-2 text-sm"
            style={{ width: 48 }}
            title="Zoom Percentage"
          />
          <span className="text-sm text-gray-600">%</span>
          {/* Reset All Button */}
          <button
            onClick={() => { handleResetZoom(); handleRotateReset(); }}
            className="px-2 py-1 text-sm bg-gray-100 rounded hover:bg-gray-200 ml-2"
            title="Reset Zoom & Rotation"
          >
            Reset
          </button>
          <span className="text-sm text-gray-600 ml-2">{rotate}&deg;</span>
        </div>
        <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center select-none">
          <img
            src={currentImage.src}
            alt={currentImage.filename}
            className="max-w-full max-h-full object-contain transition-transform duration-200"
            style={{ transform: `scale(${zoom}) rotate(${rotate}deg)` }}
          />
        </div>
      </div>
      {/* Thumbnail strip */}
      <div className="flex overflow-x-auto gap-2 py-2 mb-4 scrollbar-thin scrollbar-thumb-gray-300 scrollbar-track-gray-100">
        {images.map((img, idx) => (
          <div
            key={img.filename}
            className={`flex-shrink-0 w-20 h-20 rounded border-2 cursor-pointer relative ${idx === currentIndex ? 'border-blue-500' : 'border-transparent'}`}
            style={{ background: '#f3f4f6' }}
            onClick={() => setCurrentIndex(idx)}
          >
            <img
              src={img.src}
              alt={img.filename}
              className="w-full h-full object-contain rounded"
              style={{ opacity: idx === currentIndex ? 1 : 0.7 }}
            />
            {/* Selection Checkbox */}
            <input
              type="checkbox"
              checked={selectedImages.includes(img.filename)}
              onChange={() => toggleSelectImage(img.filename)}
              className="absolute top-1 left-1 w-4 h-4 accent-blue-600 bg-white rounded border border-gray-300"
              onClick={e => e.stopPropagation()}
            />
          </div>
        ))}
      </div>
      {/* Slider for image navigation */}
      <div className="flex items-center gap-4 mb-6">
        <span className="text-sm text-gray-600">Image</span>
        <input
          type="range"
          min={0}
          max={images.length - 1}
          value={currentIndex}
          onChange={e => setCurrentIndex(Number(e.target.value))}
          className="flex-1 accent-blue-600"
        />
        <span className="text-sm text-gray-600">{currentIndex + 1} / {images.length}</span>
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
          Run Inference on Selected Images
        </button>
      </div>

      {/* Keyboard Navigation Info */}
      <div className="text-xs text-gray-500 text-center">
        Use arrow keys or click the navigation buttons to browse through images
      </div>
    </div>
  );
};

export default ImageStackViewer; 