document.addEventListener('DOMContentLoaded', () => {
    const btnComenzar = document.getElementById('btn-comenzar');
    const transitionContainer = document.getElementById('transition-container');
    const transitionVideo = document.getElementById('transition-video');

    // PRUEBA DE CONEXIÓN: Si esto no sale en consola, el ID está mal.
    if (!btnComenzar) {
        console.error("ERROR: No encontré el botón con id='btn-comenzar'");
        return;
    }

    btnComenzar.onclick = function() {
        console.log("¡Botón presionado!");
        
        // 1. Mostrar el contenedor
        transitionContainer.style.display = 'block';
        
        // 2. Intentar reproducir
        transitionVideo.play().then(() => {
            console.log("Video en marcha...");
        }).catch(err => {
            console.warn("Video bloqueado o no encontrado, saltando a Pipeline", err);
            window.location.href = "Pipeline.html";
        });

        // 3. Al terminar, ir a la siguiente fase del proyecto
        transitionVideo.onended = () => {
            window.location.href = "Pipeline.html";
        };
    };
});