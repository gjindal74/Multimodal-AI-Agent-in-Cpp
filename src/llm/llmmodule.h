#ifndef LLMMODULE_H
#define LLMMODULE_H

#include <string>
#include <vector>
#include <memory>

// Forward declarations to avoid including llama.h in header
struct llama_model;
struct llama_context;
struct llama_sampler;

struct LLMResponse {
    std::string text;
    bool success;
    std::string error;
};

class LLMModule {
public:
    LLMModule(const std::string& modelPath);
    ~LLMModule();
    
    // Initialize the LLM
    bool init();
    
    // Generate response from prompt
    LLMResponse generate(const std::string& prompt, int maxTokens = 256);
    
    // Build structured prompt from vision + audio context
    std::string buildContextPrompt(const std::vector<std::string>& detectedObjects,
                                   const std::string& userCommand);
    
    // Check if model is loaded
    bool isLoaded() const { return model_ != nullptr && ctx_ != nullptr; }
    
private:
    std::string modelPath_;
    llama_model* model_;
    llama_context* ctx_;
    llama_sampler* sampler_;
    
    // Model parameters
    int n_ctx_;        // Context size
    int n_threads_;    // Number of threads
    float temperature_;
    float top_p_;
    int top_k_;
};

#endif // LLMMODULE_H