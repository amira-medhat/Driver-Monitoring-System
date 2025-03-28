let currentMode = "idle";
let dangerousTimeout = null;
let monitoringInterval = null;
let recognition = null;
let assistantTimeout = null;
let isWakeWordActive = false;

function initRecognition() {
    recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    recognition.lang = "en-US";
    recognition.continuous = true;

    recognition.onresult = function (event) {
        const transcript = event.results[event.results.length - 1][0].transcript.trim().toLowerCase();
        console.log("[DEBUG] Heard:", transcript);

        if (currentMode === "monitoring" && transcript.includes("hey neon")) {
            console.log("[DEBUG] Wake word matched. Switching to assistant mode...");
            currentMode = "assistant";
            isWakeWordActive = false;
            recognition.stop(); // Stop wake-word listener

            clearInterval(monitoringInterval);
            document.getElementById("response").innerText = "Neon: Hey driver, how can I help you?";

            const prompt = new SpeechSynthesisUtterance("Hey driver, how can I help you?");
            prompt.onend = () => {
                listenForQuestion();
            };
            speechSynthesis.speak(prompt);
        }
    };

    recognition.onerror = function (e) {
        console.error("[ERROR] Mic error (Wake Mode):", e.error);
        isWakeWordActive = false;
    };

    recognition.onend = () => {
        isWakeWordActive = false;
        if (currentMode === "monitoring") {
            console.log("[DEBUG] Wake word listener stopped. Restarting...");
            if (!isWakeWordActive) {
                try {
                    recognition.start();
                    isWakeWordActive = true;
                } catch (e) {
                    console.warn("[WARN] Could not restart wake-word listener:", e.message);
                }
            }
        }
    };
}

function listenForQuestion() {
    console.log("[DEBUG] Listening for user question...");

    const eq = document.getElementById("equalizer");
    if (eq) eq.classList.remove("hidden");

    let speechDetected = false;
    let hasResult = false;
    let timeoutHandle;
    let questionRecognizer;

    const startRecognizer = () => {
        speechDetected = false;
        hasResult = false;

        timeoutHandle = setTimeout(() => {
            console.log("[DEBUG] Timeout: No speech detected within 1 minute.");
            if (eq) eq.classList.add("hidden");
            currentMode = "monitoring";
            console.log("[DEBUG] currentMode is monitoring (from timeout)");
            startMonitoring();
            if (!isWakeWordActive) {
                try {
                    recognition.start();
                    isWakeWordActive = true;
                    console.log("[DEBUG] Wake word restarted.");
                } catch (e) {
                    console.warn("[WARN] Wake word restart failed:", e.message);
                }
            }
        }, 60000);

        questionRecognizer = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
        questionRecognizer.lang = "en-US";
        questionRecognizer.continuous = false;
        questionRecognizer.interimResults = false;

        questionRecognizer.start();
        console.log("[DEBUG] SpeechRecognition started.");

        questionRecognizer.onspeechstart = () => {
            console.log("[DEBUG] User started speaking...");
            speechDetected = true;
            clearTimeout(timeoutHandle);
            //document.getElementById("speakNow").style.display = "block";
            document.getElementById("speakNow").classList.add("show");

        };

        questionRecognizer.onresult = function (event) {
            hasResult = true;
            const question = event.results[0][0].transcript.trim();
            console.log("[DEBUG] Heard:", question);

            if (eq) eq.classList.add("hidden");
            questionRecognizer.stop();

            fetch("/neon_assistant", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ question })
            })
            .then(res => res.json())
            .then(data => {
                console.log("[DEBUG] Assistant response:", data.response);
                document.getElementById("response").innerText = "Neon: " + data.response;

                const reply = new SpeechSynthesisUtterance(data.response);
                reply.onend = () => {
                    console.log("[DEBUG] Response done. Listening for follow-up question...");
                    setTimeout(() => listenForQuestion(), 10000);
                };

                speechSynthesis.speak(reply);
            });
        };

        questionRecognizer.onend = () => {
            console.log("[DEBUG] SpeechRecognition ended.");
            //document.getElementById("speakNow").style.display = "none";
            document.getElementById("speakNow").classList.remove("show");

            if (!speechDetected || !hasResult) {
                clearTimeout(timeoutHandle);
                console.log("[DEBUG] No result or speech. Switching to monitoring.");
                if (eq) eq.classList.add("hidden");
                currentMode = "monitoring";
                startMonitoring();
                if (!isWakeWordActive) {
                    try {
                        recognition.start();
                        isWakeWordActive = true;
                        console.log("[DEBUG] Wake word restarted.");
                    } catch (e) {
                        console.warn("[WARN] Wake word restart failed:", e.message);
                    }
                }
            }
        };

        questionRecognizer.onerror = (e) => {
            console.error("[ERROR] SpeechRecognition error:", e.error);
            if (eq) eq.classList.add("hidden");
            clearTimeout(timeoutHandle);
            currentMode = "monitoring";
            startMonitoring();
            if (!isWakeWordActive) {
                try {
                    recognition.start();
                    isWakeWordActive = true;
                    console.log("[DEBUG] Wake word restarted.");
                } catch (e) {
                    console.warn("[WARN] Wake word restart failed:", e.message);
                }
            }
        };
    };

    startRecognizer();
}

function getSafetyInstruction() {
    if (currentMode !== "monitoring") return;

    console.log("[DEBUG] Checking safety...");
    fetch("/analyze")
        .then(res => res.json())
        .then(data => {
            if (currentMode !== "monitoring") return;
            console.log("[DEBUG] Safety result:", data);

            if (data.mode === "sleep") {
                currentMode = "sleep";
            
                try {
                    recognition.stop();  // 🛑 Stop wake-word recognizer
                    isWakeWordActive = false;
                    console.log("[DEBUG] Wake word recognizer stopped before sleep check.");
                } catch (e) {
                    console.warn("[WARN] Could not stop recognition:", e.message);
                }
            
                document.getElementById("response").innerText = "Dangerous";
                document.getElementById("response").style.color = "red";
            
                dangerousTimeout = setTimeout(() => {
                    if (currentMode === "sleep") {
                        document.getElementById("response").innerText = "Dangerous";
                        document.getElementById("response").style.color = "red";
                        currentMode = "monitoring";
                    }
                }, 150000);
            
                startListeningSleep();
            }
            
            else {
                document.getElementById("response").innerText = "Assistant: " + data.message;
                speechSynthesis.speak(new SpeechSynthesisUtterance(data.message));
            }
        })
        .catch(error => console.error("[ERROR] Error calling /analyze:", error));
}

function startListeningSleep() {
    const sleepRecognizer = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    sleepRecognizer.lang = "en-US";
    sleepRecognizer.continuous = false;
    sleepRecognizer.start();

    sleepRecognizer.onresult = function (event) {
        const speech = event.results[0][0].transcript.trim();
        console.log("[DEBUG] Sleep mode heard:", speech);
        document.getElementById("response").innerText = "You said: " + speech;
        speechSynthesis.speak(new SpeechSynthesisUtterance("Ok, I was just checking on you."));
        clearTimeout(dangerousTimeout);
        currentMode = "monitoring";
        
        // Restart wake word listener
        if (!isWakeWordActive) {
            try {
                recognition.start();
                isWakeWordActive = true;
                console.log("[DEBUG] Wake word restarted after sleep.");
            } catch (e) {
                console.warn("[WARN] Wake word restart failed:", e.message);
            }
        }
        
    };

    sleepRecognizer.onerror = function (e) {
        console.error("[ERROR] Mic error (Sleep Mode):", e.error);
    };
}

function startMonitoring() {
    console.log("[DEBUG] Monitoring started.");
    getSafetyInstruction();
    monitoringInterval = setInterval(() => {
        if (currentMode === "monitoring") getSafetyInstruction();
    }, 40000);
}

// START everything
document.getElementById("startBtn").addEventListener("click", () => {
    currentMode = "monitoring";
    document.getElementById("response").innerText = "System is running...";

    initRecognition();        // Setup the always-on wake word listener
    recognition.start();      // Start the listener
    isWakeWordActive = true;
    startMonitoring();        // Start periodic safety checks
});
