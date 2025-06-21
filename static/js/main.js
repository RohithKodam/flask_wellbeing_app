document.addEventListener('DOMContentLoaded', () => {
    // --- STATE MANAGEMENT ---
    let state = {
        currentFaceEmotion: null,
        currentVoiceEmotion: null,
        currentImagePath: null,
        currentDetector: 'mediapipe',
        charts: {},
        audioContext: null,
        analyser: null,
        recorder: null,
        animationFrameId: null,
    };

    // --- ELEMENT SELECTORS ---
    const elements = {
        video: document.getElementById('video-feed'),
        canvas: document.getElementById('canvas'),
        captureBtn: document.getElementById('capture-btn'),
        detectorSelect: document.getElementById('detector-selector'),
        faceStatus: document.getElementById('face-status'),
        recordBtn: document.getElementById('record-btn'),
        stopBtn: document.getElementById('stop-btn'),
        voiceStatus: document.getElementById('voice-status'),
        voiceVisualizer: document.getElementById('voice-visualizer'),
        sleepSlider: document.getElementById('sleep-slider'),
        sleepValue: document.getElementById('sleep-value'),
        activitySelect: document.getElementById('activity-selector'),
        logCheckinBtn: document.getElementById('log-checkin-btn'),
        clearLogsBtn: document.getElementById('clear-logs-btn'),
        analysisCard: document.getElementById('current-analysis-card'),
        capturedImageDisplay: document.getElementById('captured-image-display'),
        emotionResultDisplay: document.getElementById('emotion-result-display'),
        voiceEmotionResultDisplay: document.getElementById('voice-emotion-result-display'),
        feedbackReportDisplay: document.getElementById('feedback-report-display'),
        stressMetric: document.getElementById('stress-metric'),
        feedbackText: document.getElementById('feedback-text'),
        recommendationsCard: document.getElementById('recommendations-card'),
        recommendationsContent: document.getElementById('recommendations-content'),
        logTableBody: document.querySelector('#log-table tbody'),
        noLogsMessage: document.getElementById('no-logs-message'),
    };

    // --- INITIALIZATION ---
    setupWebcam();
    updateDashboard();

    // --- EVENT LISTENERS ---
    elements.detectorSelect.addEventListener('change', (e) => state.currentDetector = e.target.value);
    elements.captureBtn.addEventListener('click', handleFaceCapture);
    elements.recordBtn.addEventListener('click', startRecording);
    elements.stopBtn.addEventListener('click', stopRecording);
    elements.sleepSlider.addEventListener('input', () => elements.sleepValue.textContent = elements.sleepSlider.value);
    elements.logCheckinBtn.addEventListener('click', handleLogCheckin);
    elements.clearLogsBtn.addEventListener('click', handleClearLogs);

    function setupWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { elements.video.srcObject = stream; })
                .catch(err => {
                    console.error("Webcam Error:", err);
                    updateStatus(elements.faceStatus, "Webcam access denied.", 'error');
                });
        }
    }

    function handleFaceCapture() {
        updateStatus(elements.faceStatus, 'Analyzing...', 'info');
        elements.captureBtn.disabled = true;
        const context = elements.canvas.getContext('2d');
        elements.canvas.width = elements.video.videoWidth;
        elements.canvas.height = elements.video.videoHeight;
        context.drawImage(elements.video, 0, 0, elements.canvas.width, elements.canvas.height);
        const imageDataUrl = elements.canvas.toDataURL('image/jpeg');

        fetch('/analyze_face', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageDataUrl, detector: state.currentDetector }),
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            state.currentFaceEmotion = data.emotion;
            state.currentImagePath = data.image_path;
            elements.analysisCard.style.display = 'block';
            elements.capturedImageDisplay.src = imageDataUrl;
            elements.capturedImageDisplay.style.display = 'block';
            updateStatus(elements.faceStatus, `Success!`, 'success');
            updateResultDisplay(elements.emotionResultDisplay, `Facial Expression: <strong>${data.emotion}</strong>`);
        })
        .catch(error => {
            state.currentFaceEmotion = null; state.currentImagePath = null;
            updateStatus(elements.faceStatus, `Analysis failed.`, 'error');
            updateResultDisplay(elements.emotionResultDisplay, `Analysis Failed`, true);
        })
        .finally(() => { elements.captureBtn.disabled = false; });
    }

    // --- VOICE RECORDING & VISUALIZATION ---
    async function startRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            updateStatus(elements.voiceStatus, 'Audio capture not supported.', 'error');
            return;
        }
        updateStatus(elements.voiceStatus, 'Starting...', 'info');
        elements.recordBtn.disabled = true;
        
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        state.recorder = new MediaRecorder(stream);
        const audioChunks = [];

        // Setup visualization
        if (!state.audioContext) {
            state.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        state.analyser = state.audioContext.createAnalyser();
        const source = state.audioContext.createMediaStreamSource(stream);
        source.connect(state.analyser);
        drawVoiceVisualizer();

        state.recorder.ondataavailable = event => audioChunks.push(event.data);
        state.recorder.onstop = () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            analyzeVoice(audioBlob);
            stream.getTracks().forEach(track => track.stop()); // Stop mic access
            cancelAnimationFrame(state.animationFrameId);
            clearVisualizer();
        };

        state.recorder.start();
        updateStatus(elements.voiceStatus, 'Recording...', 'info');
        elements.stopBtn.disabled = false;
    }

    function stopRecording() {
        if (state.recorder && state.recorder.state === 'recording') {
            state.recorder.stop();
            elements.stopBtn.disabled = true;
            elements.recordBtn.disabled = false;
        }
    }

    function analyzeVoice(audioBlob) {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');
        updateStatus(elements.voiceStatus, 'Analyzing...', 'info');
        
        fetch('/analyze_voice', { method: 'POST', body: formData })
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                state.currentVoiceEmotion = data.voice_emotion;
                elements.analysisCard.style.display = 'block';
                updateStatus(elements.voiceStatus, `Success!`, 'success');
                updateResultDisplay(elements.voiceEmotionResultDisplay, `Vocal Tone: <strong>${data.voice_emotion}</strong>`);
            })
            .catch(error => {
                state.currentVoiceEmotion = null;
                updateStatus(elements.voiceStatus, `Analysis failed.`, 'error');
                updateResultDisplay(elements.voiceEmotionResultDisplay, `Analysis Failed`, true);
            });
    }
    
    function drawVoiceVisualizer() {
        const canvas = elements.voiceVisualizer;
        const canvasCtx = canvas.getContext('2d');
        state.analyser.fftSize = 256;
        const bufferLength = state.analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        
        const draw = () => {
            state.animationFrameId = requestAnimationFrame(draw);
            state.analyser.getByteFrequencyData(dataArray);
            
            canvasCtx.fillStyle = '#f8f9fa';
            canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
            
            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;
            
            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                canvasCtx.fillStyle = `rgba(58, 141, 222, ${barHeight / 100})`;
                canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };
        draw();
    }

    function clearVisualizer() {
        const canvas = elements.voiceVisualizer;
        const canvasCtx = canvas.getContext('2d');
        canvasCtx.fillStyle = '#f8f9fa';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // --- LOGGING & DASHBOARD ---
    function handleLogCheckin() {
        if (!state.currentFaceEmotion || state.currentFaceEmotion.startsWith("Error")) {
            alert("Please complete a successful face analysis before logging.");
            return;
        }
        const payload = {
            emotion: state.currentFaceEmotion, voice_emotion: state.currentVoiceEmotion || "N/A",
            sleep_hours: parseFloat(elements.sleepSlider.value), activity_level: elements.activitySelect.value,
            detector: state.currentDetector, image_path: state.currentImagePath || "N/A"
        };
        elements.logCheckinBtn.textContent = 'Logging...';
        elements.logCheckinBtn.disabled = true;

        fetch('/log_checkin', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                displayFeedback(data.feedback, data.stress_score);
                displayRecommendations(data.stress_score);
                updateDashboard();
            })
            .catch(error => alert(`Error logging check-in: ${error.message}`))
            .finally(() => {
                elements.logCheckinBtn.textContent = 'Complete Check-in & Get Feedback';
                elements.logCheckinBtn.disabled = false;
            });
    }

    function handleClearLogs() {
        if (confirm("Are you sure you want to permanently delete all log data?")) {
            fetch('/clear_logs', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') updateDashboard();
                    else throw new Error(data.message);
                })
                .catch(error => alert(`Error clearing logs: ${error.message}`));
        }
    }

    function updateDashboard() {
        fetch('/get_logs')
            .then(response => response.json())
            .then(logData => {
                Object.values(state.charts).forEach(chart => chart.destroy());
                state.charts = {};
                
                if (logData.data && logData.data.length > 0) {
                    elements.noLogsMessage.style.display = 'none';
                    populateLogTable(logData.data);
                    processLogDataForCharts(logData.data);
                } else {
                    elements.logTableBody.innerHTML = '';
                    elements.noLogsMessage.style.display = 'block';
                }
            })
            .catch(error => console.error("Failed to fetch logs:", error));
    }

    function populateLogTable(data) {
        elements.logTableBody.innerHTML = '';
        data.slice().reverse().forEach(log => {
            const row = document.createElement('tr');
            row.innerHTML = `<td>${new Date(log.timestamp).toLocaleString()}</td> <td>${log.face_emotion}</td> <td>${log.voice_emotion}</td> <td>${log.sleep_hours}h</td> <td>${log.activity_level}</td> <td>${log.stress_score}</td>`;
            elements.logTableBody.appendChild(row);
        });
    }

    function processLogDataForCharts(data) {
        const labels = data.map(d => new Date(d.timestamp).toLocaleDateString());
        renderChart('trends-chart', 'line', {
            labels,
            datasets: [
                { label: 'Stress Score', data: data.map(d => d.stress_score), borderColor: '#D9534F', backgroundColor: 'rgba(217, 83, 79, 0.1)', fill: true, tension: 0.4 },
                { label: 'Sleep Hours', data: data.map(d => d.sleep_hours), borderColor: '#3A8DDE', backgroundColor: 'rgba(58, 141, 222, 0.1)', fill: true, tension: 0.4 }
            ]
        }, 'Well-being Trends');
        
        const faceEmotionCounts = countOccurrences(data.map(d => d.face_emotion).filter(e => e && !e.startsWith('Error')));
        renderChart('emotion-chart', 'doughnut', { labels: Object.keys(faceEmotionCounts), datasets: [{ data: Object.values(faceEmotionCounts) }] }, 'Facial Emotion Spectrum');

        const voiceEmotionCounts = countOccurrences(data.map(d => d.voice_emotion).filter(e => e && !e.startsWith('Error') && e !== 'N/A'));
        renderChart('voice-emotion-chart', 'doughnut', { labels: Object.keys(voiceEmotionCounts), datasets: [{ data: Object.values(voiceEmotionCounts) }] }, 'Vocal Emotion Spectrum');
    }

    // --- UTILITY & DISPLAY FUNCTIONS ---
    function updateStatus(element, message, type) { element.innerHTML = message; element.className = `status-message ${type}`; }
    function updateResultDisplay(element, message, isError = false) { element.innerHTML = message; element.style.color = isError ? 'var(--danger-color)' : 'var(--text-primary)'; element.classList.add('show'); }
    function displayFeedback(feedback, score) {
        elements.feedbackReportDisplay.style.display = 'block';
        elements.stressMetric.innerHTML = `<div class="metric-label">Potential Stress Score</div><div class="metric-value">${score}</div>`;
        elements.feedbackText.innerHTML = feedback.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
    }
    function displayRecommendations(score) {
        let content = '<ul>';
        if (score <= 2) { content += '<li>You seem to be in a great place! Keep up your healthy routines.</li><li>Consider sharing your positivity with someone today.</li>'; }
        else if (score <= 4) { content += '<li>You\'re managing well. A short walk or 5 minutes of quiet time could be beneficial.</li><li>Ensure you are staying hydrated throughout the day.</li>'; }
        else { content += '<li>Your stress score is elevated. Prioritize getting a full night\'s sleep.</li><li>Consider a calming activity like listening to music or a guided meditation.</li><li>Reaching out to a friend, family member, or professional can be very helpful.</li>'; }
        content += '</ul>';
        elements.recommendationsContent.innerHTML = content;
        elements.recommendationsCard.style.display = 'block';
    }
    function renderChart(canvasId, type, chartData, title) {
        const ctx = document.getElementById(canvasId).getContext('2d');
        const chartColors = ['#3A8DDE', '#5FAD56', '#F0AD4E', '#D9534F', '#5BC0DE', '#8E7CC3', '#E56B6F'];
        let options = { responsive: true, maintainAspectRatio: false,
            plugins: {
                title: { display: true, text: title, font: { size: 16, family: "'Poppins', sans-serif" }, padding: { top: 10, bottom: 20 } },
                legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } },
                tooltip: { backgroundColor: 'rgba(0, 0, 0, 0.7)', padding: 10, cornerRadius: 4 }
            }
        };
        
        if (type === 'line') {
            options.scales = { y: { beginAtZero: true, grid: { color: 'rgba(0,0,0,0.05)' } }, x: { grid: { display: false } } };
            options.plugins.legend.labels.usePointStyle = true;
        } else if (type === 'doughnut') {
            chartData.datasets[0].backgroundColor = chartColors;
            chartData.datasets[0].borderColor = 'var(--card-bg)';
            options.cutout = '60%';
        }
        
        state.charts[canvasId] = new Chart(ctx, { type, data: chartData, options });
    }
    function countOccurrences(arr) { return arr.reduce((acc, curr) => (acc[curr] = (acc[curr] || 0) + 1, acc), {}); }
});

function openTab(evt, tabName) {
    document.querySelectorAll(".tab-content").forEach(tc => tc.style.display = "none");
    document.querySelectorAll(".tab-link").forEach(tl => tl.classList.remove("active"));
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.classList.add("active");
}