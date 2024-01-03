// Script.js 

// Select the elements 
const video = document.getElementById("video"); 
const videoThumbnail = document.getElementById("video-thumbnail"); 
const playpause = document.getElementById("play-pause"); 
const frwd = document.getElementById("skip-10"); 
const bkwrd = document.getElementById("skipminus-10"); 
const volume = document.getElementById("volume"); 
const mutebtn = document.getElementById("mute"); 
const videoContainer = document.querySelector(".video-container"); 
const controls = document.querySelector(".controls"); 
const progressBar = document.querySelector(".progress-bar"); 
const playbackline = document.querySelector(".playback-line"); 
const currentTimeRef = document.getElementById("current-time"); 
const maxDuration = document.getElementById("max-duration"); 

const timeFormatter = (timeInput) => { 
    let minute = Math.floor(timeInput / 60); 
    minute = minute < 10 ? "0" + minute : minute; 
    let second = Math.floor(timeInput % 60); 
    second = second < 10 ? "0" + second : second; 
    return `${minute}:${second}`; 
}; 

// Function to toggle play/pause 
function togglePlayPause() { 
    if (video.paused) { 
        videoThumbnail.style.display = "none"; 
        video.play(); 
        playpause.innerHTML = '<i class="fa-solid fa-pause"></i>'; 
    } else { 
        video.pause(); 
        playpause.innerHTML = '<i class="fa-solid fa-play"></i>'; 
    } 
}

// Function to seek to a specific frame
function seekToFrame(frameNumber) {
    const fps = 30; // Assuming 30 frames per second
    const seekTime = Math.round(frameNumber / fps * 100) / 100;

    console.log('Seeking to frame:', frameNumber, 'Seek time:', seekTime);
    video.currentTime = seekTime;

    // Wait for the video to seek, then play
    video.addEventListener('seeked', function onSeeked() {
        video.removeEventListener('seeked', onSeeked); // Remove the listener to avoid multiple calls
        // Play the video from the new time
        video.play().then(() => {
            // After play is initiated, toggle play/pause
            togglePlayPause();
        }).catch(error => {
            console.error('Error playing video:', error);
        });
    });
}

// Play-Pause 
playpause.addEventListener("click", function () { 
    togglePlayPause();
});

// Event listener for the video to update the isPlaying flag 
video.addEventListener("play", function () { 
    isPlaying = true; 
});

video.addEventListener("pause", function () { 
    isPlaying = false; 
});

video.addEventListener("ended", function () { 
    playpause.innerHTML = '<i class="fa-solid fa-play"></i>'; 
});

// Forward 5 sec or backward 5 sec 
frwd.addEventListener("click", function () { 
    video.currentTime += 5; 
});

bkwrd.addEventListener("click", function () { 
    video.currentTime -= 5; 
});

// Mute or Unmute 
mutebtn.addEventListener("click", function () { 
    if (video.muted) { 
        video.muted = false; 
        mutebtn.innerHTML = '<i class="fas fa-volume-up"></i>'; 
        volume.value = video.volume; 
    } else { 
        video.muted = true; 
        mutebtn.innerHTML = '<i class="fa-solid fa-volume-xmark"></i>'; 
        volume.value = 0; 
    } 
});

// Event listener for both frame circles and progress bar
document.querySelector('.playback-line').addEventListener('click', (e) => {
    const timelineWidth = playbackline.clientWidth;
    const offsetX = e.clientX - playbackline.getBoundingClientRect().left;
    const percentage = (offsetX / timelineWidth);
    const seekTime = percentage * video.duration;

    console.log('Click on progress bar - Seek time:', seekTime);
    
    // Update the video currentTime
    video.currentTime = seekTime;

    // Wait for the video to seek, then play
    video.addEventListener('seeked', function onSeeked() {
        video.removeEventListener('seeked', onSeeked); // Remove the listener to avoid multiple calls
        // Play the video from the new time
        video.play().then(() => {
            // After play is initiated, toggle play/pause
            togglePlayPause();
        }).catch(error => {
            console.error('Error playing video:', error);
        });
    });
});

// Update the playback line as the video plays 
video.addEventListener("timeupdate", () => { 
    const currentTime = video.currentTime; 
    const duration = video.duration; 
    const percentage = (currentTime / duration) * 100; 
    progressBar.style.width = percentage + "%"; 
});

// Hide or unhide controllers div 
videoContainer.addEventListener("mouseenter", () => { 
    controls.style.opacity = 1; 
});

videoContainer.addEventListener("mouseleave", () => { 
    controls.style.opacity = 1; 
});

// Reseting the playback line when the video ends 
video.addEventListener("ended", () => { 
    progressBar.style.width = "0%"; 
    showThumbnail(); 
});

// Function to show thumbnail
function showThumbnail() { 
    videoThumbnail.style.display = "block"; 
}

// Update the current time and max duration every 1 millisecond
setInterval(() => { 
    currentTimeRef.innerHTML = timeFormatter(video.currentTime); 
    maxDuration.innerText = timeFormatter(video.duration); 
}, 1);
