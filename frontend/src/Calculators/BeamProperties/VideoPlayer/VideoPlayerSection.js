import React, { useRef, useState, useEffect, Suspense } from 'react';
import './VideoPlayerSection.css';
import { MdOutlineFitScreen } from "react-icons/md";
import { GoScreenFull } from "react-icons/go";

// ✅ Lazy-load ReactPlayer to reduce main-thread blocking
const ReactPlayer = React.lazy(() => import("react-player/youtube"));

export default function VideoPlayerSection() {
  const containerRef = useRef(null);
  const playerRef = useRef(null);
  const [isTheaterMode, setIsTheaterMode] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [showPlayer, setShowPlayer] = useState(false); // ✅ For lazy mount when visible

  // ✅ Load the player only when visible on screen
  useEffect(() => {
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setShowPlayer(true);
        observer.disconnect();
      }
    });
    if (containerRef.current) observer.observe(containerRef.current);
  }, []);

  const handlePlayPause = () => setIsPlaying(!isPlaying);
  const toggleTheaterMode = () => setIsTheaterMode(!isTheaterMode);

  const handleFullScreen = () => {
    if (containerRef.current) {
      if (document.fullscreenElement) document.exitFullscreen();
      else containerRef.current.requestFullscreen();
    }
  };

  const handleTimelineChange = (e) => {
    const newTime = parseFloat(e.target.value);
    setCurrentTime(newTime);
    if (playerRef.current) playerRef.current.seekTo(newTime, "seconds");
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
  };

  return (
    <>
      <h3 className="display-4 text-center mb-5" style={{ fontWeight: "600" }}>
        How to use Beam Properties Calculator.
      </h3>
      <br />

      <div
        className={`video-container ${isTheaterMode ? "theater-mode" : ""}`}
        ref={containerRef}
        data-volume-level="high"
      >
        {/* ✅ Lazy-load the ReactPlayer only when visible */}
        {showPlayer ? (
          <Suspense fallback={<div style={{ height: '80vh', background: '#000' }} />}>
            <ReactPlayer
              ref={playerRef}
              url="https://www.youtube.com/watch?v=fYJuOTQLrDM"
              playing={isPlaying}
              controls={false}
              onProgress={({ playedSeconds }) => setCurrentTime(playedSeconds)}
              onDuration={(videoDuration) => setDuration(videoDuration)}
              width="100%"
              height="80vh"
              className="video-frame"
              config={{
                youtube: { playerVars: { modestbranding: 1, rel: 0 } },
              }}
            />
          </Suspense>
        ) : (
          <div style={{ height: '80vh', background: '#000', width: '100%' }} />
        )}

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
              onChange={handleTimelineChange}
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
          .video-container {
            position: relative;
            width: 100%;
            background-color: #000;
            aspect-ratio: 16 / 9; /* ✅ Prevent layout shift */
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
          }

          .video-container.theater-mode {
            position: relative;
            background-color: #000;
            width: 100%;
            justify-content: center;
            align-items: center;
          }

          .video-container.theater-mode .video-frame,
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
