document.addEventListener("DOMContentLoaded", function() {
    const prevBtn = document.querySelector(".prev");
    const nextBtn = document.querySelector(".next");
    const carouselImages = document.querySelector(".carousel-images");
    const images = carouselImages.querySelectorAll(".image-container");
    const numVisibleImages = 3;
    let currentIndex = 0;

    function showImages(startIndex) {
        for (let i = 0; i < images.length; i++) {
            images[i].style.display = "none";
        }
        for (let i = startIndex; i < Math.min(startIndex + numVisibleImages, images.length); i++) {
            images[i].style.display = "block";
        }
    }

    function updateButtons() {
        prevBtn.disabled = currentIndex === 0;
        nextBtn.disabled = currentIndex + numVisibleImages >= images.length;
    }

    prevBtn.addEventListener("click", function() {
        currentIndex = Math.max(0, currentIndex - numVisibleImages);
        showImages(currentIndex);
        updateButtons();
    });

    nextBtn.addEventListener("click", function() {
        currentIndex = Math.min(currentIndex + numVisibleImages, images.length - numVisibleImages);
        showImages(currentIndex);
        updateButtons();
    });

    showImages(currentIndex);
    updateButtons();
});
