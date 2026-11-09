#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace m5engine {

struct ChatMessage {
    std::string role;
    std::string content;
};

class QwenTokenizer {
public:
    explicit QwenTokenizer(uint32_t vocab_size = 32768);

    // Load vocabulary from HuggingFace tokenizer.json if present
    bool load_from_huggingface_json(const std::string& path);

    // Encode raw text into token IDs
    std::vector<int32_t> encode(const std::string& text) const;

    // Apply Qwen ChatML template (<|im_start|>role\ncontent<|im_end|>\n) and encode
    std::vector<int32_t> apply_chat_template_and_encode(const std::vector<ChatMessage>& messages) const;

    // Decode a single token ID or sequence of token IDs into text
    std::string decode_token(int32_t token_id) const;
    std::string decode(const std::vector<int32_t>& tokens) const;

    int32_t bos_token_id() const { return bos_id_; }
    int32_t eos_token_id() const { return eos_id_; }
    int32_t im_start_id() const { return im_start_id_; }
    int32_t im_end_id() const { return im_end_id_; }
    uint32_t vocab_size() const { return vocab_size_; }

private:
    uint32_t vocab_size_;
    int32_t bos_id_ = 151643;
    int32_t eos_id_ = 151645;
    int32_t im_start_id_ = 151644;
    int32_t im_end_id_ = 151645;

    std::vector<std::string> id_to_piece_;
    std::unordered_map<std::string, int32_t> piece_to_id_;

    void init_default_vocab();
};

} // namespace m5engine
