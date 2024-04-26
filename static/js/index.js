document.querySelector('.btn').addEventListener('click', function() {
    document.body.classList.add('pixelize-out');
    // Optionally, redirect to the new page after a delay
    setTimeout(function() {
        window.location.href = '/story-options'; // Change '/new-page' to the URL of the new page
    }, 1000); // Adjust the delay to match the duration of the animation
});