#include "llmmodule.h"
#include <llama.h>
#include <iostream>
#include <sstream>

LLMModule::LLMModule(const std::string& modelPath)
    : modelPath_(modelPath), model_(nullptr), ctx_(nullptr), sampler_(nullptr),
      n_ctx_(2048), n_threads_(4), temperature_(0.7f), top_p_(0.9f), top_k_(40) {
}

LLMModule::~LLMModule() {
    if (sampler_) {
        llama_sampler_free(sampler_);
        sampler_ = nullptr;
    }
    
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    
    llama_backend_free();
}

bool LLMModule::init() {
    std::cout << "Initializing LLM Module..." << std::endl;
    
    // Initialize llama backend
    llama_backend_init();
    
    // Load model
    struct llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99; // Offload all layers to GPU
    
    model_ = llama_model_load_from_file(modelPath_.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load LLM model from: " << modelPath_ << std::endl;
        return false;
    }
    
    std::cout << "LLM model loaded successfully" << std::endl;
    
    // Create context
    struct llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx_;
    ctx_params.n_threads = n_threads_;
    ctx_params.n_threads_batch = n_threads_;
    
    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "Failed to create LLM context" << std::endl;
        return false;
    }
    
    // Create sampler
    auto sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_k(top_k_));
    llama_sampler_chain_add(sampler_, llama_sampler_init_top_p(top_p_, 1));
    llama_sampler_chain_add(sampler_, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(sampler_, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    
    std::cout << "LLM Module ready (context size: " << n_ctx_ << " tokens)" << std::endl;
    
    return true;
}

LLMResponse LLMModule::generate(const std::string& prompt, int maxTokens) {
    LLMResponse response;
    response.success = false;
    
    if (!isLoaded()) {
        response.error = "LLM not initialized";
        return response;
    }
    
    // Don't recreate context - just use it as-is
    // The model will handle state internally
    
    std::cout << "\n=== LLM Generation ===" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Generating..." << std::endl;
    
    // Get vocab from model
    const struct llama_vocab * vocab = llama_model_get_vocab(model_);
    
    // Tokenize the prompt (don't add BOS since prompt already has it)
    std::vector<llama_token> tokens;
    tokens.resize(prompt.size() + 128);
    
    int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.size(), 
                                   tokens.data(), tokens.size(), false, false);
    
    if (n_tokens < 0) {
        response.error = "Failed to tokenize prompt";
        return response;
    }
    
    tokens.resize(n_tokens);
    
    std::cout << "Prompt tokens: " << n_tokens << std::endl;
    
    // Safety check
    if (n_tokens > n_ctx_ - 256) {
        response.error = "Prompt too long for context window";
        return response;
    }
    
    // Prepare batch with proper size
    llama_batch batch = llama_batch_init(n_ctx_, 0, 1);
    
    // Manually populate batch fields for initial prompt
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = false;
    }
    batch.n_tokens = tokens.size();
    
    // Process the prompt - only need logits for last token
    batch.logits[batch.n_tokens - 1] = true;
    
    if (llama_decode(ctx_, batch) != 0) {
        response.error = "Failed to decode prompt";
        llama_batch_free(batch);
        return response;
    }
    
    // Generate tokens
    std::string generated_text;
    int n_generated = 0;
    int braceCount = 0;
    bool inJson = false;
    
    while (n_generated < maxTokens) {
        // Sample next token
        llama_token new_token = llama_sampler_sample(sampler_, ctx_, -1);
        
        // Check for end of generation
        if (llama_vocab_is_eog(vocab, new_token)) {
            break;
        }
        
        // Convert token to text
        char buf[128];
        int n_chars = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
        
        if (n_chars > 0) {
            std::string tokenText(buf, n_chars);
            generated_text.append(tokenText);
            
            // Track JSON braces to stop when complete
            for (char c : tokenText) {
                if (c == '{') {
                    braceCount++;
                    inJson = true;
                } else if (c == '}') {
                    braceCount--;
                    if (inJson && braceCount == 0) {
                        // Complete JSON object found, stop generation
                        goto generation_complete;
                    }
                }
            }
        }
        
        // Prepare next batch for single token
        llama_batch_free(batch);
        batch = llama_batch_init(n_ctx_, 0, 1);
        
        // Manually set single token
        batch.token[0] = new_token;
        batch.pos[0] = n_tokens + n_generated;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = true;
        batch.n_tokens = 1;

        if (llama_decode(ctx_, batch) != 0) {
            response.error = "Failed to decode during generation";
            llama_batch_free(batch);
            return response;
        }
        
        n_generated++;
    }
    
generation_complete:
    llama_batch_free(batch);
    
    // Clean up the response - extract only JSON if present
    size_t jsonStart = generated_text.find('{');
    size_t jsonEnd = generated_text.rfind('}');
    if (jsonStart != std::string::npos && jsonEnd != std::string::npos && jsonEnd > jsonStart) {
        generated_text = generated_text.substr(jsonStart, jsonEnd - jsonStart + 1);
    }
    
    std::cout << "Generated " << n_generated << " tokens" << std::endl;
    std::cout << "Response: " << generated_text << std::endl;
    std::cout << "===================" << std::endl;
    
    response.text = generated_text;
    response.success = true;
    
    return response;
}

std::string LLMModule::buildContextPrompt(const std::vector<std::string>& detectedObjects,
                                         const std::string& userCommand) {
    std::ostringstream prompt;
    
    // System prompt for Llama 3.2
    prompt << "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n";
    prompt << "You are a computer control assistant. Respond ONLY with valid JSON.\n";
    prompt << "Actions: open_url, notify, none\n";
    prompt << "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n";
    
    // Visual context
    prompt << "Scene: ";
    if (detectedObjects.empty()) {
        prompt << "empty";
    } else {
        for (size_t i = 0; i < detectedObjects.size(); i++) {
            prompt << detectedObjects[i];
            if (i < detectedObjects.size() - 1) prompt << ", ";
        }
    }
    prompt << "\n";
    
    // User command
    prompt << "Command: \"" << userCommand << "\"\n\n";
    prompt << "JSON response:<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    
    return prompt.str();
}