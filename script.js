document.addEventListener("DOMContentLoaded", () => {
    const card = document.querySelector('.glass-card');
    
    // Subtle 3D tilt effect for the glass card
    if (card && window.matchMedia("(min-width: 768px)").matches) {
        document.addEventListener('mousemove', (e) => {
            const xAxis = (window.innerWidth / 2 - e.pageX) / 45;
            const yAxis = (window.innerHeight / 2 - e.pageY) / 45;
            
            // Adding a slight glow effect based on mouse position
            const xRect = e.clientX - card.getBoundingClientRect().left;
            const yRect = e.clientY - card.getBoundingClientRect().top;
            
            card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg) translateY(-5px)`;
            
            // Optional: Dynamic reflection using a pseudoelement or box-shadow
            card.style.boxShadow = `
                ${-xAxis}px ${-yAxis}px 50px -12px rgba(0, 0, 0, 0.5), 
                0 0 40px rgba(45, 212, 191, 0.05)
            `;
        });
        
        // Reset transform smoothly when the mouse stops or leaves the window
        document.body.addEventListener('mouseleave', () => {
            card.style.transform = `rotateY(0deg) rotateX(0deg) translateY(0)`;
            card.style.boxShadow = `0 25px 50px -12px rgba(0, 0, 0, 0.5)`;
        });
    }
});
