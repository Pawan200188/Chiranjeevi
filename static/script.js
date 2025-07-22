document.addEventListener("DOMContentLoaded", function () {
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const previewContainer = document.getElementById("previewContainer");
    const previewImage = document.getElementById("previewImage");
    const progressBar = document.getElementById("progressBar");
    const progress = document.getElementById("progress");
    const analyzeButton = document.getElementById("analyzeButton");
    const confidenceLevel = document.getElementById("confidenceLevel");
    const confidenceText = document.getElementById("confidenceText");
    const helpButton = document.getElementById("helpButton");
    const chatbotButton = document.getElementById("chatbotButton");
    const chatbotWindow = document.getElementById("chatbotWindow");
    const closeChatbot = document.getElementById("closeChatbot");
    const sendMessageButton = document.getElementById("sendMessage");
    const userInput = document.getElementById("userInput");
    const chatBody = document.getElementById("chatBody");

    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.background = "rgba(0, 180, 216, 0.1)";
    });

    dropzone.addEventListener("dragleave", () => {
        dropzone.style.background = "transparent";
    });

    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        handleFile(e.dataTransfer.files[0]);
    });

    dropzone.addEventListener("click", () => {
        fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
        handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if (file && (file.type.startsWith("image/") || file.name.endsWith(".dcm") || file.name.endsWith(".nii"))) {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                previewContainer.style.display = "block";
                dropzone.style.background = "transparent";
                resetAnalysis();
            };
            reader.readAsDataURL(file);
        }
    }

    function resetAnalysis() {
        confidenceLevel.style.width = "0%";
        confidenceText.textContent = "Ready for analysis";
    }

    async function analyzeScan() {
        try {
            if (!fileInput.files[0]) {
                throw new Error("Please select a file first");
            }

            progressBar.style.display = "block";
            progress.style.width = "0%";
            analyzeButton.disabled = true;
            confidenceText.textContent = "Uploading scan...";

            const formData = new FormData();
            formData.append("file", fileInput.files[0]);

            let progressValue = 0;
            const progressInterval = setInterval(() => {
                progressValue += 2;
                if (progressValue <= 50) {
                    progress.style.width = `${progressValue}%`;
                }
            }, 100);

            const uploadResponse = await fetch("http://127.0.0.1:8000/upload/", {
                method: "POST",
                body: formData,
            });

            if (!uploadResponse.ok) {
                throw new Error("Upload failed");
            }

            const uploadResult = await uploadResponse.json();
            confidenceText.textContent = "Processing scan...";

            progressValue = 50;
            const processInterval = setInterval(() => {
                progressValue += 2;
                if (progressValue <= 90) {
                    progress.style.width = `${progressValue}%`;
                }
            }, 100);

            const predictResponse = await fetch(`http://127.0.0.1:8000/predict/${uploadResult.filename}`, {
                method: "POST",
            });

            if (!predictResponse.ok) {
                throw new Error("Prediction failed");
            }

            clearInterval(progressInterval);
            clearInterval(processInterval);
            progress.style.width = "100%";

            const predictResult = await predictResponse.json();

            setTimeout(() => {
                showResults(predictResult);
                analyzeButton.disabled = false;
            }, 500);
        } catch (error) {
            console.log("Script.js loaded successfully!");
            confidenceText.textContent = `Error: ${error.message}`;
            analyzeButton.disabled = false;
            progressBar.style.display = "none";
        }
    }

    analyzeButton.addEventListener("click", analyzeScan);

    function showResults(result) {
        const { filename, "Classification Type": classification_type, "Confidence": confidence } = result;

        confidenceText.innerHTML = `
            <strong>Classification: ${classification_type}</strong><br>
            Confidence: ${confidence}<br>
        `;

        progressBar.style.display = "none";

        addToHistory(filename, classification_type, confidence);
    }

    function addToHistory(filename, classification_type, confidence) {
        const historyBody = document.getElementById("historyBody");
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${filename}</td>
            <td>${classification_type}</td>
            <td>${confidence}</td>
        `;

        historyBody.prepend(row);
    };

    helpButton.addEventListener("click", () => {
        const message = `
    NeuroScan AI Help:
    1. Drag & drop or click to upload MRI scans
    2. Supported formats: JPG, PNG
    3. Analysis typically takes 15-30 seconds
    4. Results include confidence level and recommendations

    For technical support:-
    Email-ID: neuroninjas@gmail.com  
    Contact: 7887941386

    We value your feedback! Click OK to share your experience.
        `;
    
        if (confirm(message)) {
            window.open("https://forms.gle/6cttzT9mS81VkowJA", "_blank");
        }
    });
    
    let isChatbotOpen = false;
    
    chatbotButton.addEventListener("click", function () {
        if (!isChatbotOpen) {
            chatbotWindow.style.display = "block";
            setTimeout(() => chatbotWindow.classList.add("fade-in"), 10);
            chatbotWindow.classList.remove("fade-out");
        } else {
            chatbotWindow.classList.remove("fade-in");
            chatbotWindow.classList.add("fade-out");
            setTimeout(() => chatbotWindow.style.display = "none", 300);
        }
        isChatbotOpen = !isChatbotOpen;
    });
    
    closeChatbot.addEventListener("click", function () {
        chatbotWindow.classList.remove("fade-in");
        chatbotWindow.classList.add("fade-out");
        setTimeout(() => chatbotWindow.style.display = "none", 300);
        isChatbotOpen = false;
    });
    
    function scrollToBottom() {
        const chatBody = document.getElementById("chatBody");
        chatBody.scrollTop = chatBody.scrollHeight;
    }
    
    sendMessageButton.addEventListener("click", () => {
        sendMessageToChatbot();
        setTimeout(scrollToBottom, 100);
    });
    
    async function sendMessageToChatbot() {
        const message = userInput.value.trim();
        if (!message) return;
    
        const userMessage = document.createElement("p");
        userMessage.classList.add("user-message");
        userMessage.textContent = message;
        chatBody.appendChild(userMessage);
    
        userInput.value = "";
        scrollToBottom();
    
        const loadingMessage = document.createElement("div");
        loadingMessage.classList.add("bot-message", "loading-message");
        loadingMessage.innerHTML = `<span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>`;
        chatBody.appendChild(loadingMessage);
        scrollToBottom();
    
        try {
            const response = await fetch("http://127.0.0.1:8000/chatbot/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: message }),
            });
    
            const data = await response.json();
    
            if (response.ok) {
                chatBody.removeChild(loadingMessage);

                const botMessage = document.createElement("p");
                botMessage.classList.add("bot-message");
                botMessage.innerHTML = data.response
                    .replace(/\n/g, "<br>")
                    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
                    .replace(/\*(.*?)\*/g, "<em>$1</em>")
                    .replace(/- (.*?)(<br>|$)/g, "<ul><li>$1</li></ul>")
                    .replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank">$1</a>');
    
                chatBody.appendChild(botMessage);
                scrollToBottom();
            } else {
                chatBody.removeChild(loadingMessage);
                alert("Server Error: " + data.detail);
            }
        } catch (error) {
            chatBody.removeChild(loadingMessage);
            console.error("Chatbot Request Failed:", error);
            alert("An error occurred: " + error.message);
        }
    }
    
    sendMessageButton.addEventListener("click", sendMessageToChatbot);
    
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessageToChatbot();
        }
    });          

    const buttons = document.querySelectorAll(".tool-button");
    buttons.forEach((button) => {
        button.addEventListener("mouseover", (e) => {
            const tooltip = document.createElement("div");
            tooltip.className = "tooltip";
            tooltip.textContent = button.textContent;
            tooltip.style.position = "absolute";
            tooltip.style.background = "rgba(0, 0, 0, 0.8)";
            tooltip.style.padding = "5px 10px";
            tooltip.style.borderRadius = "4px";
            tooltip.style.fontSize = "12px";
            tooltip.style.zIndex = "1000";
            document.body.appendChild(tooltip);

            const rect = button.getBoundingClientRect();
            tooltip.style.top = `${rect.top - 30}px`;
            tooltip.style.left = `${rect.left}px`;

            button.addEventListener("mouseout", () => tooltip.remove());
        });
    });
});
