/**
 * Viva AI - Frontend Logic
 * Handles file upload, analysis requests, and results display.
 */

const uploadArea = document.getElementById("upload-area");
const fileInput = document.getElementById("file-input");
const fileInfo = document.getElementById("file-info");
const fileName = document.getElementById("file-name");
const clearFileBtn = document.getElementById("clear-file");
const analyzeBtn = document.getElementById("analyze-btn");
const inputSection = document.getElementById("input-section");
const processingSection = document.getElementById("processing-section");
const resultsSection = document.getElementById("results-section");
const progressFill = document.getElementById("progress-fill");
const statusText = document.getElementById("status-text");
const newAnalysisBtn = document.getElementById("new-analysis-btn");

let selectedFile = null;

// Upload area click
uploadArea.addEventListener("click", () => fileInput.click());

// Drag and drop
uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("drag-over");
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

// File input change
fileInput.addEventListener("change", () => {
    if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// Clear file
clearFileBtn.addEventListener("click", () => {
    selectedFile = null;
    fileInfo.style.display = "none";
    uploadArea.style.display = "block";
    analyzeBtn.disabled = true;
    fileInput.value = "";
});

// Handle file selection
function handleFile(file) {
    const validTypes = ["video/mp4", "video/avi", "video/mov", "video/webm",
                        "video/x-msvideo", "video/quicktime",
                        "audio/wav", "audio/mp3", "audio/mpeg", "audio/webm"];
    const ext = file.name.split(".").pop().toLowerCase();
    const validExts = ["mp4", "avi", "mov", "webm", "wav", "mp3"];

    if (!validTypes.includes(file.type) && !validExts.includes(ext)) {
        alert("Please upload a video or audio file (MP4, AVI, MOV, WebM, WAV, MP3).");
        return;
    }

    selectedFile = file;
    fileName.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)} MB)`;
    fileInfo.style.display = "flex";
    uploadArea.style.display = "none";
    analyzeBtn.disabled = false;
}

// Analyze button
analyzeBtn.addEventListener("click", async () => {
    console.log("Analyze button clicked, file:", selectedFile ? selectedFile.name : "none");
    if (!selectedFile) {
        alert("No file selected. Please upload a file first.");
        return;
    }

    // Show processing UI immediately
    inputSection.style.display = "none";
    processingSection.style.display = "block";
    resultsSection.style.display = "none";
    analyzeBtn.disabled = true;

    const formData = new FormData();
    formData.append("file", selectedFile);

    // Animate progress while waiting
    let fakeProgress = 5;
    const progressInterval = setInterval(() => {
        if (fakeProgress < 85) {
            fakeProgress += Math.random() * 3;
            updateProgress(Math.min(fakeProgress, 85),
                fakeProgress < 20 ? "Uploading file..." :
                fakeProgress < 50 ? "Analyzing video frames (this may take a minute)..." :
                fakeProgress < 70 ? "Transcribing audio with Whisper..." :
                "Computing final scores...");
        }
    }, 1000);

    try {
        updateProgress(5, "Uploading file...");
        console.log("Sending request to /analyze...");

        const response = await fetch("/analyze", {
            method: "POST",
            body: formData,
        });

        clearInterval(progressInterval);
        console.log("Response received, status:", response.status);

        if (!response.ok) {
            let errMsg = "Analysis failed";
            try {
                const err = await response.json();
                errMsg = err.error || errMsg;
            } catch (e) {
                errMsg = "Server error (status " + response.status + ")";
            }
            throw new Error(errMsg);
        }

        updateProgress(90, "Loading results...");

        const data = await response.json();
        console.log("Analysis result:", data);
        updateProgress(100, "Complete!");

        setTimeout(() => displayResults(data), 500);

    } catch (error) {
        clearInterval(progressInterval);
        console.error("Analysis error:", error);
        processingSection.style.display = "none";
        inputSection.style.display = "block";
        analyzeBtn.disabled = false;
        alert("Error: " + error.message);
    }
});

// New analysis
newAnalysisBtn.addEventListener("click", () => {
    resultsSection.style.display = "none";
    inputSection.style.display = "block";
    selectedFile = null;
    fileInfo.style.display = "none";
    uploadArea.style.display = "block";
    analyzeBtn.disabled = true;
    fileInput.value = "";
    progressFill.style.width = "0%";
});

function updateProgress(percent, text) {
    progressFill.style.width = percent + "%";
    statusText.textContent = text;
}

function displayResults(data) {
    processingSection.style.display = "none";
    resultsSection.style.display = "block";

    // Overall score and grade
    document.getElementById("overall-score").textContent = Math.round(data.overall_score);
    const gradeBadge = document.getElementById("grade-badge");
    gradeBadge.textContent = data.grade;
    gradeBadge.style.color = gradeColor(data.grade);
    gradeBadge.style.borderColor = gradeColor(data.grade);

    const circle = document.getElementById("overall-circle");
    circle.style.borderColor = gradeColor(data.grade);

    // Component scores
    const components = data.component_scores || {};
    setScoreBar("eye-contact", components.eye_contact);
    setScoreBar("head-stability", components.head_stability);
    setScoreBar("fluency", components.fluency);
    setScoreBar("clarity", components.clarity);
    setScoreBar("filler", components.filler_penalty);
    setScoreBar("pace", components.pace);

    // Feedback
    const strengthsList = document.getElementById("strengths-list");
    const improvementsList = document.getElementById("improvements-list");
    strengthsList.innerHTML = "";
    improvementsList.innerHTML = "";

    (data.strengths || []).forEach(s => {
        const li = document.createElement("li");
        li.textContent = s;
        strengthsList.appendChild(li);
    });

    (data.improvements || []).forEach(s => {
        const li = document.createElement("li");
        li.textContent = s;
        improvementsList.appendChild(li);
    });

    if (strengthsList.children.length === 0) {
        strengthsList.innerHTML = "<li>No specific strengths identified yet</li>";
    }
    if (improvementsList.children.length === 0) {
        improvementsList.innerHTML = "<li>No specific improvements needed</li>";
    }

    // Transcription
    const transcriptionEl = document.getElementById("transcription-text");
    const meta = data.metadata || {};
    transcriptionEl.textContent = meta.transcription || data.transcription || "No transcription available.";

    // Stats
    document.getElementById("stat-words").textContent = meta.audio_word_count || "--";
    document.getElementById("stat-wpm").textContent = meta.words_per_minute || "--";
    document.getElementById("stat-duration").textContent = meta.audio_duration_seconds || "--";
    document.getElementById("stat-fillers").textContent = meta.filler_word_count || "0";
}

function setScoreBar(id, value) {
    const val = Math.round(value || 0);
    const bar = document.getElementById("bar-" + id);
    const num = document.getElementById("score-" + id);

    num.textContent = val;

    let colorClass = "bar-high";
    if (val < 50) colorClass = "bar-low";
    else if (val < 75) colorClass = "bar-mid";

    bar.className = "score-bar " + colorClass;
    setTimeout(() => { bar.style.width = val + "%"; }, 100);
}

function gradeColor(grade) {
    if (grade.startsWith("A")) return "#2ecc71";
    if (grade.startsWith("B")) return "#6c63ff";
    if (grade.startsWith("C")) return "#f39c12";
    return "#e74c3c";
}
