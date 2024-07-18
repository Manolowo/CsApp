
document.addEventListener('DOMContentLoaded', function() {
    const pageContent = document.querySelector('.page-info');

    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            pageContent.classList.add('scrolled-up');
        } else {
            pageContent.classList.remove('scrolled-up');
        }
    });
});

function focusEconomia() {
    console.log("Haciendo foco en Economía...");
    var economiaDiv = document.getElementById('economia');
    if (economiaDiv) {
        economiaDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

function focusVictoria() {
    console.log("Haciendo foco en Victoria...");
    var victoriaDiv = document.getElementById('victoria');
    if (victoriaDiv) {
        victoriaDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

function focusEstadistica() {
    console.log("Haciendo foco en estadistica...");
    var estadisticaDiv = document.getElementById('estadisticas');
    if (estadisticaDiv) {
        estadisticaDiv.scrollIntoView({ behavior: 'smooth' });
    }
}

// Obtener el botón
const scrollToTopBtn = document.getElementById('scrollToTopBtn');

// Mostrar el botón cuando se desplaza hacia abajo
window.onscroll = function() {
    if (document.body.scrollTop > 100 || document.documentElement.scrollTop > 100) {
        scrollToTopBtn.style.display = 'block';
    } else {
        scrollToTopBtn.style.display = 'none';
    }
};

// Hacer que el botón funcione
scrollToTopBtn.onclick = function() {
    window.scrollTo({ top: 0, behavior: 'smooth' });
};
