document.addEventListener('DOMContentLoaded', () => {
    // 1. REFERENCIAS A ELEMENTOS
    const audio = document.getElementById('audio');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const playIcon = playPauseBtn ? playPauseBtn.querySelector('i') : null;
    const progressBar = document.getElementById('progressBar');
    
    const btnComenzar = document.getElementById('btn-comenzar');
    const transitionContainer = document.getElementById('transition-container');
    const transitionVideo = document.getElementById('transition-video');

    // 2. LÓGICA DEL REPRODUCTOR DE AUDIO
    if (playPauseBtn && audio) {
        playPauseBtn.addEventListener('click', () => {
            if (audio.paused) {
                audio.play().catch(e => console.log("Interacción requerida para audio"));
                playIcon.className = 'fa-solid fa-pause';
            } else {
                audio.pause();
                playIcon.className = 'fa-solid fa-play';
            }
        });

        audio.addEventListener('timeupdate', () => {
            if (!isNaN(audio.duration)) {
                progressBar.value = (audio.currentTime / audio.duration) * 100;
            }
        });

        progressBar.addEventListener('input', () => {
            audio.currentTime = (progressBar.value / 100) * audio.duration;
        });
    }
});