// static/js/main.js

document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let currentFaceEmotion = null;
    let currentVoiceEmotion = null;
    let currentImagePath = null;
    let mediaRecorder;
    let audioChunks = [];
    let charts = {};

    // --- ELEMENT SELECTORS ---
    const video = document.getElementById('video-feed');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const faceStatus = document.getElementById('face-status');
    const detectorSelect = document.getElementById('detector-selector');
    
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const voiceStatus = document.getElementById('voice-status');

    const sleepSlider = document.getElementById('sleep-slider');
    const sleepValue = document.getElementById('sleep-value');
    const activitySelect = document.getElementById('activity-selector');

    const logCheckinBtn = document.getElementById('log-checkin-btn');
    const clearLogsBtn = document.getElementById('clear-logs-btn');

    const capturedImageDisplay = document.getElementById('captured-image-display');
    const currentCheckinInfo = document.getElementById('current-checkin-info');
    const emotionResultDisplay = document.getElementById('emotion-result-display');
    const voiceEmotionResultDisplay = document.getElementById('voice-emotion-result-display');

    const feedbackReportDisplay = document.getElementById('feedback-report-display');
    const stressMetric = document.getElementById('stress-metric');
    const feedbackText = document.getElementById('feedback-text');

    // --- INITIALIZATION ---
    setupWebcam();
    updateDashboard();

    // --- EVENT LISTENERS ---
    captureBtn.addEventListener('click', handleFaceCapture);
    recordBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    sleepSlider.addEventListener('input', () => sleepValue.textContent = sleepSlider.value);
    logCheckinBtn.addEventListener('click', handleLogCheckin);
    clearLogsBtn.addEventListener('click', handleClearLogs);

    // --- WEBCAM & FACE ANALYSIS ---
    function setupWebcam() {
        if (navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true, audio: false })
                .then(stream => {
                    video.srcObject = stream;
                })
                .catch(err => {
                    console.error("Error accessing webcam:", err);
                    faceStatus.textContent = "Error: Could not access webcam.";
                    faceStatus.className = 'status-message error';
                });
        }
    }

    function handleFaceCapture() {
        faceStatus.textContent = 'Analyzing...';
        faceStatus.className = 'status-message info';
        captureBtn.disabled = true;

        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        const imageDataUrl = canvas.toDataURL('image/jpeg');
        capturedImageDisplay.src = imageDataUrl;
        capturedImageDisplay.style.display = 'block';
        currentCheckinInfo.style.display = 'none';

        fetch('/analyze_face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataUrl, detector: detectorSelect.value }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            currentFaceEmotion = data.emotion;
            currentImagePath = data.image_path;
            updateStatus(faceStatus, `Success: ${data.emotion}`, 'success');
            updateResultDisplay(emotionResultDisplay, `Face Emotion: <strong>${data.emotion}</strong>`);
        })
        .catch(error => {
            currentFaceEmotion = null;
            currentImagePath = null;
            updateStatus(faceStatus, `Error: ${error.message}`, 'error');
            updateResultDisplay(emotionResultDisplay, `Face Analysis Failed`, true);
        })
        .finally(() => {
            captureBtn.disabled = false;
        });
    }

    // --- AUDIO RECORDING & ANALYSIS ---
    function startRecording() {
        navigator.mediaDevices.getUserMedia({ audio: true, video: false })
            .then(stream => {
                mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
                audioChunks = [];
                mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
                mediaRecorder.start();

                updateStatus(voiceStatus, 'Recording...', 'info');
                recordBtn.disabled = true;
                stopBtn.disabled = false;
            }).catch(err => {
                console.error("Error accessing microphone:", err);
                updateStatus(voiceStatus, 'Error: Mic access denied.', 'error');
            });
    }

    function stopRecording() {
        mediaRecorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' }); // Flask will handle conversion
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');

            updateStatus(voiceStatus, 'Analyzing voice...', 'info');
            stopBtn.disabled = true;

            fetch('/analyze_voice', { method: 'POST', body: formData })
                .then(response => response.json())
                .then(data => {
                    if (data.error) throw new Error(data.error);
                    currentVoiceEmotion = data.voice_emotion;
                    updateStatus(voiceStatus, `Success: ${data.voice_emotion}`, 'success');
                    updateResultDisplay(voiceEmotionResultDisplay, `Voice Emotion: <strong>${data.voice_emotion}</strong>`);
                })
                .catch(error => {
                    currentVoiceEmotion = null;
                    updateStatus(voiceStatus, `Error: ${error.message}`, 'error');
                    updateResultDisplay(voiceEmotionResultDisplay, `Voice Analysis Failed`, true);
                })
                .finally(() => {
                    recordBtn.disabled = false;
                });
        };
        mediaRecorder.stop();
    }

    // --- LOGGING & FEEDBACK ---
    function handleLogCheckin() {
        if (!currentFaceEmotion || currentFaceEmotion.startsWith("Error")) {
            alert("Please capture a valid face emotion before logging.");
            return;
        }

        const payload = {
            emotion: currentFaceEmotion,
            voice_emotion: currentVoiceEmotion || "N/A",
            sleep_hours: parseFloat(sleepSlider.value),
            activity_level: activitySelect.value,
            detector: detectorSelect.value,
            image_path: currentImagePath || "N/A"
        };
        
        logCheckinBtn.textContent = 'Logging...';
        logCheckinBtn.disabled = true;

        fetch('/log_checkin', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            displayFeedback(data.feedback, data.stress_score);
            alert("Check-in logged successfully!");
            updateDashboard(); // Refresh charts
        })
        .catch(error => {
            alert(`Error logging check-in: ${error.message}`);
        })
        .finally(() => {
            logCheckinBtn.textContent = '📊 Analyze and Log Check-in';
            logCheckinBtn.disabled = false;
        });
    }
    
    function handleClearLogs() {
        if (confirm("Are you sure you want to permanently delete all log data? This cannot be undone.")) {
            fetch('/clear_logs', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        alert(data.message);
                        // Reset UI state
                        currentFaceEmotion = null;
                        currentVoiceEmotion = null;
                        currentImagePath = null;
                        emotionResultDisplay.style.display = 'none';
                        voiceEmotionResultDisplay.style.display = 'none';
                        feedbackReportDisplay.style.display = 'none';
                        capturedImageDisplay.style.display = 'none';
                        currentCheckinInfo.style.display = 'block';
                        updateDashboard();
                    } else {
                        throw new Error(data.message);
                    }
                })
                .catch(error => alert(`Error clearing logs: ${error.message}`));
        }
    }


    // --- UI UPDATERS ---
    function updateStatus(element, message, type) {
        element.textContent = message;
        element.className = `status-message ${type}`;
    }

    function updateResultDisplay(element, message, isError = false) {
        element.innerHTML = message;
        element.style.color = isError ? 'var(--error-color)' : 'var(--text-color)';
        element.style.display = 'block';
    }
    
    function displayFeedback(feedback, score) {
        feedbackReportDisplay.style.display = 'block';
        stressMetric.innerHTML = `
            <div class="metric-label">Calculated Potential Stress Score</div>
            <div class="metric-value">${score}</div>
        `;
        // Replace markdown-like bolding with HTML tags
        feedbackText.innerHTML = feedback.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    }
    
    // --- DASHBOARD & CHARTS ---
    function updateDashboard() {
        fetch('/get_logs')
            .then(response => response.json())
            .then(logData => {
                if (logData.data && logData.data.length > 0) {
                    processLogDataForCharts(logData.data);
                } else {
                    // Clear charts if no data
                    Object.values(charts).forEach(chart => chart.destroy());
                    charts = {};
                }
            });
    }

    function processLogDataForCharts(data) {
        const labels = data.map(d => d.timestamp);
        
        // Stress Chart
        const stressScores = data.map(d => d.stress_score);
        renderChart('stress-chart', 'line', labels, 'Stress Score', stressScores, '#FF4B4B');
        
        // Sleep Chart
        const sleepHours = data.map(d => d.sleep_hours);
        renderChart('sleep-chart', 'line', labels, 'Sleep Hours', sleepHours, '#00BFFF');
        
        // Emotion Chart
        const emotionCounts = countOccurrences(data.map(d => d.emotion).filter(e => e && !e.startsWith('Error')));
        renderChart('emotion-chart', 'bar', Object.keys(emotionCounts), 'Count', Object.values(emotionCounts), '#4BC0C0');

        // Voice Emotion Chart
        const voiceEmotionCounts = countOccurrences(data.map(d => d.voice_emotion).filter(e => e && !e.startsWith('Error') && e !== 'N/A'));
        renderChart('voice-emotion-chart', 'bar', Object.keys(voiceEmotionCounts), 'Count', Object.values(voiceEmotionCounts), '#9966FF');
    }

    function renderChart(canvasId, type, labels, label, data, color) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        if (charts[canvasId]) {
            charts[canvasId].destroy();
        }
        charts[canvasId] = new Chart(ctx, {
            type: type,
            data: {
                labels: labels,
                datasets: [{
                    label: label,
                    data: data,
                    backgroundColor: color,
                    borderColor: color,
                    borderWidth: type === 'line' ? 2 : 1,
                    fill: false,
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }

    function countOccurrences(arr) {
        return arr.reduce((acc, curr) => {
            acc[curr] = (acc[curr] || 0) + 1;
            return acc;
        }, {});
    }
});

// Tab switching logic for dashboard
function openTab(evt, tabName) {
    let i, tabcontent, tablinks;
    tabcontent = document.getElementsByClassName("tab-content");
    for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    tablinks = document.getElementsByClassName("tab-link");
    for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}