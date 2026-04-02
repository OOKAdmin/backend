import React, { useRef, useState, useEffect, Suspense } from 'react';
import './VideoPlayerSection.css';
import { MdOutlineFitScreen } from "react-icons/md";
import { GoScreenFull } from "react-icons/go";

// Lazy import (this doesn’t change how your component works)
const ReactPlayer = React.lazy(() => import("react-player/youtube"));


export default function VideoPlayerSection() {
  const containerRef = useRef(null); // Reference to the video container
  const playerRef = useRef(null); // Reference to the ReactPlayer instance
  const [isTheaterMode, setIsTheaterMode] = useState(false); // State for theater mode
  const [isPlaying, setIsPlaying] = useState(false); // State for play/pause
  const [currentTime, setCurrentTime] = useState(0); // Current playback time
  const [duration, setDuration] = useState(0); // Video duration

  // Function to toggle play/pause
  const handlePlayPause = () => {
    setIsPlaying(!isPlaying);
  };

  // Function to toggle theater mode
  const toggleTheaterMode = () => {
    setIsTheaterMode(!isTheaterMode);
  };

  // Function to toggle full-screen mode
  const handleFullScreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) {
        document.exitFullscreen();
      } else {
        containerRef.current.requestFullscreen();
      }
    }
  };

  // Function to handle timeline changes (when user interacts with the timeline)
  const handleTimelineChange = (e) => {
    const newTime = parseFloat(e.target.value); // Get new time from the slider
    setCurrentTime(newTime); // Update the currentTime state
    if (playerRef.current) {
      playerRef.current.seekTo(newTime, "seconds"); // Seek the video to the new time
    }
  };

  // Function to format time in MM:SS format
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  return (
    <>
      <h2 className="display-4 text-center mb-5" style={{ fontWeight: "600" }}>
        How to use Beam Properties Calculator.
      </h2>
      <br />

      <div
        className={`video-container ${isTheaterMode ? "theater-mode" : ""}`}
        ref={containerRef}
        data-volume-level="high"
      >
        <Suspense fallback={<div style={{ height: '80vh', background: '#000' }} />}>
          <ReactPlayer
            ref={playerRef}
            url="https://www.youtube.com/embed/-uyWQ_gSKu4?si=Dsgb06rz6cdhCL5V"
            playing={isPlaying}
            controls={false}
            onProgress={({ playedSeconds }) => setCurrentTime(playedSeconds)}
            onDuration={(videoDuration) => setDuration(videoDuration)}
            width="100%"
            height="80vh"
            className="video-frame"
            config={{
              youtube: {
                playerVars: { modestbranding: 1, rel: 0 },
              },
            }}
          />
        </Suspense>


        <div className="video-controls-container">
          <div className="timeline-container">
            <input
              type="range"
              className="timeline"
  aria-label="Video Progress"
              min="0"
              max={duration || 100}
              value={currentTime || 0}
              step="0.1"
              onChange={handleTimelineChange} // Handle timeline change
            />
          </div>

          <div className="controls">
            <button className="play-pause-btn" aria-label="Play or pause video" onClick={handlePlayPause}>
              {isPlaying ? (
                <svg className="pause-icon" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M14,19H18V5H14M6,19H10V5H6V19Z" />
                </svg>
              ) : (
                <svg className="play-icon" viewBox="0 0 24 24">
                  <path fill="currentColor" d="M8,5.14V19.14L19,12.14L8,5.14Z" />
                </svg>
              )}
            </button>
            <div className="duration-container">
              <div className="current-time">{formatTime(currentTime)}</div>/
              <div className="total-time">{formatTime(duration || 100)}</div>
            </div>
            <button className="theater-mode-btn" onClick={toggleTheaterMode} aria-label="Toggle theater mode">
              <MdOutlineFitScreen />
            </button>
            <button className="fullscreen-btn" onClick={handleFullScreen} aria-label="Enter fullscreen">
              <GoScreenFull style={{ color: "#fff" }} />
            </button>
          </div>
        </div>
      </div>
      <style>
        {`
          .video-container.theater-mode {
            position: relative;
            left: 0;
            right: 0;
            bottom: 50%;
            background-color: #000;
            display: flex;
            max-width: -webkit-fill-available;
            justify-content: center;
            align-items: center;
          }
          .video-container {
            position: relative;
            left: 0;
            right: 0;
            bottom: 50%;
            background-color: #000;
            display: flex;
            justify-content: center;
            align-items: center;
          }

          .video-container.theater-mode .video-frame {
            width: 100%;
            height: 80vh;
          }

          .video-frame {
            width: 100%;
            height: 80vh;
          }

          .timeline {
            width: 100%;
            -webkit-appearance: none;
            height: 5px;
            background: #444;
            outline: none;
            cursor: pointer;
            border-radius: 2px;
          }

          .play-pause-btn {
            background: none;
            border: none;
            color: #fff;
            cursor: pointer;
          }
        `}
      </style>
    </>
  );
}
