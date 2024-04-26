document.addEventListener("DOMContentLoaded", function() {
    const prevBtn = document.querySelector(".prev");
    const nextBtn = document.querySelector(".next");
    var carouselTexts = document.querySelector(".carousel-texts");
    var texts = carouselTexts.querySelectorAll(".text-container");
    const numVisibleTexts = 1;
    let currentIndex = 0;

    function handleOptionSelection(selectedOption, text, pdf_path, currentIndex) {
        const apiUrl = 'http://127.0.0.1:5000/path'

        fetch(apiUrl, {
            method: "POST",
            body: JSON.stringify({
                text: text,
                pdf_path: pdf_path,
                decision: selectedOption
            }),
            headers: {
                "Content-type": "application/json; charset=UTF-8"
            }
        })
        .then(response => response.json())
        .then(data => {
            const path = data['path'];
            const nextIndex = currentIndex + 1;

            if (nextIndex < texts.length) {
                var nextContainer = texts[nextIndex];
                var nextParagraph = nextContainer.querySelector("p");
                if (!nextParagraph) {
                    nextContainer = document.createElement("div");
                    nextContainer.classList.add("text-container");
                    nextParagraph = document.createElement("p");
                    nextParagraph.textContent = path;
                    nextContainer.appendChild(nextParagraph);

                } else {
                    // Update the text content of the next paragraph with the generated path
                    nextParagraph.textContent = path;
                }

            } else {
                console.log("No more text containers to show.");
            }
            showTexts(nextIndex, path);
            updateButtons();

            // window.location.href
        })
    
    
    }   


    function fetchOptions(text, startIndex) {
        const apiUrl = 'http://127.0.0.1:5000/options'

        fetch(apiUrl, {
            method: "POST",
            body: JSON.stringify({
                text: text,
                pdf_path: pdf_path
            }),
            headers: {
                "Content-type": "application/json; charset=UTF-8"
            }
        })
        .then(response => response.json())
        .then(data => {
            var options = data['options'];

            let optionsFormHTML = '<form class="text-container">';

            options.forEach((option, index) => {
                const optionId = `option${index}`;
                optionsFormHTML += `<input type="radio" name="options" id="${optionId}" value="${option}">`;
                optionsFormHTML += `<label for="${optionId}">${option}</label>`;
                optionsFormHTML += '<br>';
            });
        
            optionsFormHTML += '</form>';


            const currentContainer = texts[startIndex];

            currentContainer.innerHTML = `<p class="opt-container-title"> Ponder the question.... What If? </p?`;

            currentContainer.innerHTML += optionsFormHTML;

            const optionsForm = currentContainer.querySelector("form");
            optionsForm.addEventListener("click", function(event) {
                event.preventDefault();
                const selectedOption = optionsForm.querySelector("input[name='options']:checked");
                if (selectedOption) {
                    console.log(`Selected Option: ${selectedOption.value}`);
                    console.log("\n\n");
                    handleOptionSelection(selectedOption.value, text, pdf_path, currentIndex);
                    currentContainer.innerHTML = "";
                    // currentContainer.querySelector("p").innerHTML = '<div class="spinner"></div>';
                } else {
                    // Handle case where no option is selected
                    console.log("Please select an option.");
                }
            });
        })
        .catch(error => {
            console.error('Error:', error);
            // Handle error, e.g., display an error message
        });
    }

    function classifyText(text, startIndex){

        const apiUrl = 'http://127.0.0.1:5000/clf';
       
        // Call the summarization API
        fetch(apiUrl, {
            method: "POST",
            body: JSON.stringify({
                text: text
            }),
            headers: {
                "Content-type": "application/json; charset=UTF-8"
            }
        })
        .then(response => response.json())
        .then(data => {

            decision = data["decision"];

            console.log(`Does the excerpt contain a decision?: ${decision}`)

            if (decision == "yes") {
                fetchOptions(text, startIndex)
                var optionText = `
                <br>
                <p class="opt-container-yes">
                    You have reached a nexus point. New story paths are being generated.
                    <br>
                    If you do not wish to make a decision, click ❯ to continue reading.
                </p>
                <br>
                `
                
            }
            else {
                const nothing = 1;  

                var optionText = `
                <br>
                <p class="opt-container-no">
                Click ❯ to continue reading.
                </p>
                <br>

                `
            }

            var currentContainer = texts[startIndex];
            var currentParagraph = currentContainer.querySelector("p");
            currentParagraph.innerHTML += optionText;


        })
        .catch(error => {
            console.error('Error:', error);
            // Handle error, e.g., display an error message
        });
    }

    function showTexts(startIndex, path = null) {
        for (let i = 0; i < texts.length; i++) {
            texts[i].style.display = "none";
        }
        for (let i = startIndex; i < Math.min(startIndex + numVisibleTexts, texts.length); i++) {
            texts[i].style.display = "block";
        }
        for (let i = startIndex; i < Math.min(startIndex + numVisibleTexts, texts.length); i++) {
            const currentContainer = texts[startIndex];

            var paragraphElement = currentContainer.querySelector("p");
            
            if (!paragraphElement) {


                paragraphElement = document.createElement("p");
                paragraphElement.textContent = path;
                currentContainer.appendChild(paragraphElement);
            }

            const text = paragraphElement.textContent;
            
            // Show loading spinner while API call is in progress
            paragraphElement.innerHTML = '<div class="spinner"></div>';
            
            const apiUrl = 'http://127.0.0.1:5000/summ';
            
            // Call the summarization API
            fetch(apiUrl, {
                method: "POST",
                body: JSON.stringify({
                    text: text
                }),
                headers: {
                    "Content-type": "application/json; charset=UTF-8"
                }
            })
            .then(response => response.json())
            .then(data => {
                paragraphElement.textContent = data['summary'];
                classifyText(text,startIndex);
            })
            .catch(error => {
                console.error('Error:', error);
                // Handle error, e.g., display an error message
            });
        }
    }
    
    function updateButtons() {
        prevBtn.disabled = currentIndex === 0;
        nextBtn.disabled = currentIndex + numVisibleTexts >= texts.length;
    }

    prevBtn.addEventListener("click", function() {
        currentIndex = Math.max(0, currentIndex - numVisibleTexts);
        showTexts(currentIndex);
        updateButtons();
    });

    nextBtn.addEventListener("click", function() {
        // applyModificationToNextPage();
        currentIndex = Math.min(currentIndex + numVisibleTexts, texts.length - numVisibleTexts);
        showTexts(currentIndex);
        updateButtons();
    });

    showTexts(currentIndex);
    updateButtons();
});
