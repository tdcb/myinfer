#include "tokenizer.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

namespace m5engine {

QwenTokenizer::QwenTokenizer(uint32_t vocab_size) : vocab_size_(vocab_size) {
    init_default_vocab();
}

void QwenTokenizer::init_default_vocab() {
    id_to_piece_.assign(vocab_size_, "");
    piece_to_id_.clear();

    // 0..255: raw byte fallback tokens
    for (uint32_t b = 0; b < 256 && b < vocab_size_; ++b) {
        std::string s(1, static_cast<char>(b));
        id_to_piece_[b] = s;
        piece_to_id_[s] = static_cast<int32_t>(b);
    }

    // Common English words, subwords, coding tokens, and technical terms
    const char* common_tokens[] = {
        " the", " and", " to", " of", " in", " is", " for", " that", " with", " on",
        " as", " are", " this", " by", " be", " at", " or", " from", " an", " have",
        " Apple", " Silicon", " M5", " Pro", " Metal", " Unified", " Memory", " GPU",
        " CPU", " Super", " cores", " bandwidth", " inference", " engine", " model",
        " Dense", " MoE", " Mixture", " Experts", " router", " top", " active",
        " FlashAttention", " KV", " cache", " prefix", " Radix", " speculative",
        " decoding", " Multi", " Token", " Prediction", " MTP", " quantized", " 4-bit",
        " Q4_G64", " SIMD", " matrix", " throughput", " latency", " TTFT", " tokens",
        " second", " thermal", " fan", " zero-copy", " mmap", " OpenAI", " API",
        " Qwen", " 27B", " 3.8", " architecture", " kernel", " threadgroup", " register",
        " SwiGLU", " RoPE", " RMSNorm", " GQA", " prefill", " decode", " batch",
        "import ", "def ", "return ", "class ", "struct ", "const ", "void ", "int ",
        "float ", "half ", "uint ", "std::", "vector", "string", "include", "pragma",
        "```", "python", "cpp", "json", "\n\n", "    ", " -> ", " == ", " != ",
        " because", " reduces", " eliminates", " hardware", " execution", " pipeline",
        " command", " buffer", " dispatch", " semaphore", " sleep", " coalesced",
        " cache-line", " 128-byte", " 16KB", " page-aligned", " dequantization",
        " fused", " gate", " up", " down", " projection", " attention", " heads",
        " layer", " norm", " residual", " stream", " response", " assistant", " user",
        " system", " fast", " efficient", " sustained", " performance", " optimal"
    };

    int32_t next_id = 256;
    for (const char* tok : common_tokens) {
        if (static_cast<uint32_t>(next_id) >= vocab_size_ - 16) break;
        std::string s(tok);
        id_to_piece_[next_id] = s;
        piece_to_id_[s] = next_id;
        next_id++;
    }

    // Fill remaining vocab slots with deterministic readable subword pieces
    for (uint32_t id = static_cast<uint32_t>(next_id); id < vocab_size_ - 8; ++id) {
        id_to_piece_[id] = "t" + std::to_string(id);
    }

    // Special Qwen ChatML control tokens at the top of vocab
    bos_id_      = static_cast<int32_t>(vocab_size_ - 4);
    im_start_id_ = static_cast<int32_t>(vocab_size_ - 3);
    im_end_id_   = static_cast<int32_t>(vocab_size_ - 2);
    eos_id_      = static_cast<int32_t>(vocab_size_ - 2);

    id_to_piece_[bos_id_]      = "<|endoftext|>";
    id_to_piece_[im_start_id_] = "<|im_start|>";
    id_to_piece_[im_end_id_]   = "<|im_end|>";

    piece_to_id_["<|endoftext|>"] = bos_id_;
    piece_to_id_["<|im_start|>"]  = im_start_id_;
    piece_to_id_["<|im_end|>"]    = im_end_id_;
}

// Convert GPT-2 / Qwen byte-level BPE unicode representation (Ġ -> space, Ċ -> \n)
static std::string unescape_bpe_piece(const std::string& raw) {
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(raw[i]);
        if (c == 0xC4 && i + 1 < raw.size()) {
            unsigned char c2 = static_cast<unsigned char>(raw[i + 1]);
            if (c2 == 0xA0) { // 'Ġ' (U+0120) -> ' '
                out.push_back(' ');
                ++i;
                continue;
            } else if (c2 == 0x8A) { // 'Ċ' (U+010A) -> '\n'
                out.push_back('\n');
                ++i;
                continue;
            }
        }
        out.push_back(static_cast<char>(c));
    }
    return out;
}

bool QwenTokenizer::load_from_huggingface_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return false;

    std::stringstream ss;
    ss << f.rdbuf();
    std::string content = ss.str();

    size_t vocab_pos = content.find("\"vocab\"");
    if (vocab_pos == std::string::npos) return false;
    size_t brace_start = content.find('{', vocab_pos);
    if (brace_start == std::string::npos) return false;

    size_t pos = brace_start + 1;
    uint32_t loaded_count = 0;
    while (pos < content.size()) {
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;
        if (pos >= content.size() || content[pos] == '}') break;
        if (content[pos] == ',') { pos++; continue; }
        if (content[pos] != '"') break;

        // Parse JSON string key
        pos++;
        std::string key;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.size()) {
                pos++;
                char esc = content[pos];
                if (esc == 'n') key.push_back('\n');
                else if (esc == 't') key.push_back('\t');
                else if (esc == 'r') key.push_back('\r');
                else if (esc == '"' || esc == '\\' || esc == '/') key.push_back(esc);
                else key.push_back(esc);
            } else {
                key.push_back(content[pos]);
            }
            pos++;
        }
        if (pos < content.size() && content[pos] == '"') pos++;

        size_t colon = content.find(':', pos);
        if (colon == std::string::npos) break;
        pos = colon + 1;
        while (pos < content.size() && std::isspace(static_cast<unsigned char>(content[pos]))) pos++;

        size_t num_end = pos;
        while (num_end < content.size() && (std::isdigit(static_cast<unsigned char>(content[num_end])) || content[num_end] == '-')) {
            num_end++;
        }
        if (num_end == pos) break;
        int32_t id = std::stoi(content.substr(pos, num_end - pos));
        pos = num_end;

        if (id >= 0) {
            if (static_cast<uint32_t>(id) >= id_to_piece_.size()) {
                id_to_piece_.resize(id + 1);
                vocab_size_ = static_cast<uint32_t>(id_to_piece_.size());
            }
            std::string decoded = unescape_bpe_piece(key);
            id_to_piece_[id] = decoded;
            piece_to_id_[decoded] = id;
            if (key == "<|im_start|>") im_start_id_ = id;
            if (key == "<|im_end|>") { im_end_id_ = id; eos_id_ = id; }
            if (key == "<|endoftext|>") bos_id_ = id;
            loaded_count++;
        }
    }
    return loaded_count > 100;
}

std::vector<int32_t> QwenTokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens;
    tokens.reserve(text.size() / 2 + 4);

    size_t i = 0;
    while (i < text.size()) {
        // Greedy longest-match subword lookup (up to 24 bytes)
        size_t max_len = std::min<size_t>(24, text.size() - i);
        bool matched = false;
        for (size_t len = max_len; len >= 2; --len) {
            std::string sub = text.substr(i, len);
            auto it = piece_to_id_.find(sub);
            if (it != piece_to_id_.end()) {
                tokens.push_back(it->second);
                i += len;
                matched = true;
                break;
            }
        }
        if (!matched) {
            std::string single(1, text[i]);
            auto it = piece_to_id_.find(single);
            if (it != piece_to_id_.end()) {
                tokens.push_back(it->second);
            } else {
                tokens.push_back(static_cast<uint8_t>(text[i]));
            }
            i++;
        }
    }
    return tokens;
}

std::vector<int32_t> QwenTokenizer::apply_chat_template_and_encode(const std::vector<ChatMessage>& messages) const {
    std::vector<int32_t> out;
    for (const auto& msg : messages) {
        out.push_back(im_start_id_);
        std::string body = msg.role + "\n" + msg.content;
        auto body_toks = encode(body);
        out.insert(out.end(), body_toks.begin(), body_toks.end());
        out.push_back(im_end_id_);
        auto nl = encode("\n");
        out.insert(out.end(), nl.begin(), nl.end());
    }
    // Prime assistant turn
    out.push_back(im_start_id_);
    auto asst = encode("assistant\n");
    out.insert(out.end(), asst.begin(), asst.end());
    return out;
}

std::string QwenTokenizer::decode_token(int32_t token_id) const {
    if (token_id < 0 || static_cast<size_t>(token_id) >= id_to_piece_.size()) {
        return "";
    }
    if (token_id == im_start_id_ || token_id == im_end_id_ || token_id == bos_id_) {
        return "";
    }
    return id_to_piece_[token_id];
}

std::string QwenTokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string out;
    for (int32_t t : tokens) {
        out += decode_token(t);
    }
    return out;
}

} // namespace m5engine
