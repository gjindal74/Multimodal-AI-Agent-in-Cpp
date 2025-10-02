#include "../../include/audio.h"
#include <whisper.h>
#include <portaudio.h>
#include <iostream>
#include <cmath>
#include <algorithm>

AudioModule::AudioModule(const std::string& modelPath)
    : modelPath_(modelPath), ctx_(nullptr), isListening_(false),
      shouldStop_(false), isRecording_(false), vadThreshold_(0.01f), sampleRate_(16000),
      bufferSizeMs_(3000), transcriptCount_(0), currentAudioLevel_(0.0f) {
    audioBuffer_.reserve(sampleRate_ * bufferSizeMs_ / 1000);
}

AudioModule::~AudioModule() {
    stopListening();
    
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
    
    Pa_Terminate();
}

bool AudioModule::init() {
    std::cout << "Initializing Audio Module..." << std::endl;
    
    // Initialize Whisper model
    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true;  // Try to use GPU if available
    
    ctx_ = whisper_init_from_file_with_params(modelPath_.c_str(), cparams);
    if (!ctx_) {
        std::cerr << "Failed to load Whisper model from: " << modelPath_ << std::endl;
        return false;
    }
    
    std::cout << "Whisper model loaded successfully" << std::endl;
    
    // Initialize PortAudio
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << std::endl;
        return false;
    }
    
    std::cout << "PortAudio initialized successfully" << std::endl;
    std::cout << "Audio Module ready (sample rate: " << sampleRate_ << " Hz)" << std::endl;
    
    return true;
}

void AudioModule::startListening() {
    if (isListening_) {
        std::cout << "Already listening" << std::endl;
        return;
    }
    
    shouldStop_ = false;
    isListening_ = true;
    captureThread_ = std::thread(&AudioModule::audioThread, this);
    
    std::cout << "Started listening for voice commands..." << std::endl;
}

void AudioModule::stopListening() {
    if (!isListening_) return;
    
    shouldStop_ = true;
    isListening_ = false;
    
    if (captureThread_.joinable()) {
        captureThread_.join();
    }
    
    std::cout << "Stopped listening" << std::endl;
}

void AudioModule::startRecording() {
    if (!isListening_) {
        std::cout << "Cannot record: audio module not listening" << std::endl;
        return;
    }
    
    if (isRecording_) return;
    
    std::lock_guard<std::mutex> lock(bufferMutex_);
    audioBuffer_.clear();
    isRecording_ = true;
    std::cout << "🔴 Recording started (hold SPACE)..." << std::endl;
}

void AudioModule::stopRecording() {
    if (!isRecording_) return;
    
    isRecording_ = false;
    std::cout << "⏹️  Recording stopped, processing..." << std::endl;
    
    // Copy buffer and process
    std::vector<float> recordedAudio;
    {
        std::lock_guard<std::mutex> lock(bufferMutex_);
        recordedAudio = audioBuffer_;
        audioBuffer_.clear();
    }
    
    if (!recordedAudio.empty()) {
        processAudioBuffer(recordedAudio);
    } else {
        std::cout << "No audio recorded" << std::endl;
    }
}

void AudioModule::audioThread() {
    PaStream* stream = nullptr;
    PaStreamParameters inputParams;
    
    // Configure input parameters
    inputParams.device = Pa_GetDefaultInputDevice();
    if (inputParams.device == paNoDevice) {
        std::cerr << "No default input device found" << std::endl;
        isListening_ = false;
        return;
    }
    
    inputParams.channelCount = 1;  // Mono
    inputParams.sampleFormat = paFloat32;
    inputParams.suggestedLatency = Pa_GetDeviceInfo(inputParams.device)->defaultLowInputLatency;
    inputParams.hostApiSpecificStreamInfo = nullptr;
    
    // Open stream
    PaError err = Pa_OpenStream(&stream, &inputParams, nullptr, sampleRate_,
                                 paFramesPerBufferUnspecified, paClipOff,
                                 nullptr, nullptr);
    
    if (err != paNoError) {
        std::cerr << "Failed to open audio stream: " << Pa_GetErrorText(err) << std::endl;
        isListening_ = false;
        return;
    }
    
    err = Pa_StartStream(stream);
    if (err != paNoError) {
        std::cerr << "Failed to start audio stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(stream);
        isListening_ = false;
        return;
    }
    
    std::cout << "Audio stream started successfully" << std::endl;
    
    // Capture loop - simplified for push-to-talk
    const int framesPerRead = sampleRate_ / 10;  // 100ms chunks
    std::vector<float> tempBuffer(framesPerRead);
    const float audioGain = 100.0f;  // Amplify audio by 100x
    
    while (!shouldStop_) {
        // Read audio data
        err = Pa_ReadStream(stream, tempBuffer.data(), framesPerRead);
        
        if (err != paNoError && err != paInputOverflowed) {
            std::cerr << "Error reading audio stream: " << Pa_GetErrorText(err) << std::endl;
            break;
        }
        
        // Amplify audio signal
        for (float& sample : tempBuffer) {
            sample *= audioGain;
            // Clamp to prevent overflow
            if (sample > 1.0f) sample = 1.0f;
            if (sample < -1.0f) sample = -1.0f;
        }
        
        // Calculate and store current audio level for visualization
        float energy = 0.0f;
        for (float sample : tempBuffer) {
            energy += sample * sample;
        }
        energy = std::sqrt(energy / tempBuffer.size());
        currentAudioLevel_ = energy;
        
        // If recording (spacebar held), accumulate audio
        if (isRecording_) {
            std::lock_guard<std::mutex> lock(bufferMutex_);
            audioBuffer_.insert(audioBuffer_.end(), tempBuffer.begin(), tempBuffer.end());
        }
    }
    
    // Clean up
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    std::cout << "Audio thread stopped" << std::endl;
}

bool AudioModule::detectVoiceActivity(const std::vector<float>& audioData) {
    if (audioData.empty()) return false;
    
    // Simple energy-based VAD
    float energy = 0.0f;
    for (float sample : audioData) {
        energy += sample * sample;
    }
    energy /= audioData.size();
    
    // Debug: Print energy levels occasionally
    static int debugCounter = 0;
    if (++debugCounter % 50 == 0) {  // Print every 5 seconds
        std::cout << "Audio energy: " << energy << " (threshold: " << vadThreshold_ << ")" << std::endl;
    }
    
    return energy > vadThreshold_;
}

void AudioModule::processAudioBuffer(const std::vector<float>& audioData) {
    if (!ctx_ || audioData.empty()) {
        std::cout << "Cannot process: invalid context or empty audio" << std::endl;
        return;
    }
    
    // Require at least 0.5 seconds of audio
    const float minDurationSec = 0.5f;
    const int minSamples = static_cast<int>(sampleRate_ * minDurationSec);
    
    if (audioData.size() < minSamples) {
        std::cout << "Audio too short (" << audioData.size() / sampleRate_ 
                  << "s), need at least " << minDurationSec << "s" << std::endl;
        return;
    }
    
    // Simple noise reduction: remove very quiet parts at beginning/end
    std::vector<float> cleanedAudio = audioData;
    const float noiseThreshold = 0.01f;
    
    // Trim silence from start
    size_t startIdx = 0;
    for (size_t i = 0; i < cleanedAudio.size(); i++) {
        if (std::abs(cleanedAudio[i]) > noiseThreshold) {
            startIdx = i;
            break;
        }
    }
    
    // Trim silence from end
    size_t endIdx = cleanedAudio.size();
    for (size_t i = cleanedAudio.size(); i > 0; i--) {
        if (std::abs(cleanedAudio[i-1]) > noiseThreshold) {
            endIdx = i;
            break;
        }
    }
    
    // Use trimmed audio
    if (endIdx > startIdx) {
        cleanedAudio = std::vector<float>(cleanedAudio.begin() + startIdx, 
                                          cleanedAudio.begin() + endIdx);
    }
    
    std::cout << "Processing audio with Whisper (" 
              << cleanedAudio.size() / sampleRate_ << "s)..." << std::endl;
    
    // Prepare Whisper parameters
    struct whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress = false;
    wparams.print_special = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.translate = false;
    wparams.language = "en";
    wparams.n_threads = 4;
    wparams.suppress_blank = false;  // Don't suppress blanks
    wparams.no_speech_thold = 0.3f;   // Lower threshold (was 0.6 default)
    wparams.entropy_thold = 2.0f;     // Lower entropy threshold
    
    // Run inference on cleaned audio
    int ret = whisper_full(ctx_, wparams, cleanedAudio.data(), cleanedAudio.size());
    
    if (ret != 0) {
        std::cerr << "Whisper inference failed" << std::endl;
        return;
    }
    
    // Get transcription
    const int n_segments = whisper_full_n_segments(ctx_);
    std::string fullTranscript;
    
    for (int i = 0; i < n_segments; i++) {
        const char* text = whisper_full_get_segment_text(ctx_, i);
        if (text) {
            fullTranscript += text;
        }
    }
    
    // Trim whitespace
    fullTranscript.erase(0, fullTranscript.find_first_not_of(" \t\n\r"));
    fullTranscript.erase(fullTranscript.find_last_not_of(" \t\n\r") + 1);
    
    if (!fullTranscript.empty()) {
        std::cout << "Transcript: \"" << fullTranscript << "\"" << std::endl;
        
        // Store transcript
        {
            std::lock_guard<std::mutex> lock(transcriptMutex_);
            transcriptQueue_.push(fullTranscript);
            transcriptCount_++;
        }
        
        // Call callback if set
        if (transcriptCallback_) {
            transcriptCallback_(fullTranscript);
        }
    } else {
        std::cout << "No speech detected in audio" << std::endl;
    }
}

std::string AudioModule::getLatestTranscript() {
    std::lock_guard<std::mutex> lock(transcriptMutex_);
    if (transcriptQueue_.empty()) {
        return "";
    }
    
    std::string transcript = transcriptQueue_.front();
    transcriptQueue_.pop();
    return transcript;
}

bool AudioModule::hasNewTranscript() {
    std::lock_guard<std::mutex> lock(transcriptMutex_);
    return !transcriptQueue_.empty();
}

void AudioModule::setTranscriptCallback(std::function<void(const std::string&)> callback) {
    transcriptCallback_ = callback;
}

float AudioModule::getCurrentAudioLevel() const {
    return currentAudioLevel_;
}