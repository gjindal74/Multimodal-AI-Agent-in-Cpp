#ifndef AUDIO_H
#define AUDIO_H

#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <queue>
#include <functional>

// Forward declare whisper context to avoid including whisper.h here
struct whisper_context;

class AudioModule {
public:
    AudioModule(const std::string& modelPath);
    ~AudioModule();
    
    // Initialize audio capture and whisper model
    bool init();
    
    // Start/stop listening
    void startListening();
    void stopListening();
    bool isListening() const { return isListening_; }
    
    // Get the latest transcribed text
    std::string getLatestTranscript();
    bool hasNewTranscript();
    
    // Set callback for when new transcript is available
    void setTranscriptCallback(std::function<void(const std::string&)> callback);
    
    // Voice Activity Detection threshold (0.0 to 1.0)
    void setVADThreshold(float threshold) { vadThreshold_ = threshold; }
    
private:
    // Audio capture thread function
    void audioThread();
    
    // Process audio buffer with Whisper
    void processAudioBuffer(const std::vector<float>& audioData);
    
    // Check if audio contains speech (simple energy-based VAD)
    bool detectVoiceActivity(const std::vector<float>& audioData);
    
    std::string modelPath_;
    whisper_context* ctx_;
    
    // Threading
    std::atomic<bool> isListening_;
    std::atomic<bool> shouldStop_;
    std::thread captureThread_;
    
    // Transcript management
    std::mutex transcriptMutex_;
    std::queue<std::string> transcriptQueue_;
    std::function<void(const std::string&)> transcriptCallback_;
    
    // Audio buffer
    std::vector<float> audioBuffer_;
    std::mutex bufferMutex_;
    
    // Settings
    float vadThreshold_;
    int sampleRate_;
    int bufferSizeMs_;  // Buffer size in milliseconds
    
    // Stats
    int transcriptCount_;
};

#endif // AUDIO_H