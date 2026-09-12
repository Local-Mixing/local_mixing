#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Gate = std::array<int, 3>;
using Mask = std::array<uint64_t, 2>;

static int val(char c) {
  const std::string chars =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
  auto p = chars.find(c);
  if (p == std::string::npos) return -1;
  return static_cast<int>(p);
}

static int hex_val(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
}

static std::vector<Gate> parse(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path);
  std::vector<Gate> gates;
  std::vector<int> w;
  int overflow = 0;
  char ch;
  while (in.get(ch)) {
    if (ch == ';') {
      if (!w.empty()) {
        if (w.size() != 3) throw std::runtime_error("bad gate");
        gates.push_back({w[0], w[1], w[2]});
        w.clear();
      }
      overflow = 0;
    } else if (ch == '~') {
      overflow++;
    } else if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      continue;
    } else {
      int b = val(ch);
      if (b < 0) throw std::runtime_error("bad char");
      w.push_back(b + 83 * overflow);
      overflow = 0;
    }
  }
  if (!w.empty()) throw std::runtime_error("unterminated gate");
  return gates;
}

static uint64_t parse_hex_low64(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  uint64_t out = 0;
  for (int bit = 0; bit < 64; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) out |= 1ULL << bit;
  }
  for (int bit = 64; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) throw std::runtime_error("target too wide");
  }
  return out;
}

static int parse_int(const std::string &s, const char *name) {
  int out = 0;
  std::stringstream ss(s);
  ss >> out;
  if (!ss || out < 0) throw std::runtime_error(std::string("bad ") + name);
  return out;
}

static bool bit_mask(const Mask &m, int bit) {
  return (m[bit >> 6] >> (bit & 63)) & 1ULL;
}

static void set_mask(Mask &m, int bit) { m[bit >> 6] |= 1ULL << (bit & 63); }

static void clear_mask(Mask &m, int bit) {
  m[bit >> 6] &= ~(1ULL << (bit & 63));
}

static void xor_mask(Mask &a, const Mask &b) {
  a[0] ^= b[0];
  a[1] ^= b[1];
}

static uint8_t parity_and(const Mask &a, const Mask &b) {
  return (__builtin_popcountll(a[0] & b[0]) ^
          __builtin_popcountll(a[1] & b[1])) &
         1;
}

static bool zero_mask(const Mask &m) { return !(m[0] || m[1]); }

static Mask low_mask(int cols) {
  if (cols <= 64) return {cols == 64 ? ~0ULL : ((1ULL << cols) - 1), 0};
  return {~0ULL, (1ULL << (cols - 64)) - 1};
}

static int pop64(uint64_t x) { return __builtin_popcountll(x); }

static std::vector<uint8_t> bits_from_x(const Mask &x, int total_n,
                                        int free_bits) {
  std::vector<uint8_t> bits(total_n, 0);
  for (int i = 0; i < free_bits; i++) bits[i] = bit_mask(x, i);
  return bits;
}

static uint64_t low64_eval(const Mask &x, int total_n, int free_bits,
                           const std::vector<Gate> &gates,
                           std::vector<uint8_t> *full_out = nullptr) {
  auto bits = bits_from_x(x, total_n, free_bits);
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) {
    if (bits[i]) out |= 1ULL << i;
  }
  if (full_out) *full_out = bits;
  return out;
}

static void eval_low64_jac(const Mask &x, int total_n, int free_bits,
                           const std::vector<Gate> &gates, uint64_t &out,
                           std::array<Mask, 64> &rows) {
  auto bits = bits_from_x(x, total_n, free_bits);
  std::vector<Mask> deriv(total_n, Mask{0, 0});
  for (int i = 0; i < free_bits; i++) set_mask(deriv[i], i);
  const Mask lane_mask = low_mask(free_bits);

  for (auto [a, b, c] : gates) {
    uint8_t g0 = bits[b] | !bits[c];
    Mask bmask = deriv[b];
    Mask cmask = deriv[c];
    if (bits[b]) xor_mask(bmask, lane_mask);
    if (bits[c]) xor_mask(cmask, lane_mask);
    Mask gmask{bmask[0] | (~cmask[0] & lane_mask[0]),
               bmask[1] | (~cmask[1] & lane_mask[1])};
    if (g0) xor_mask(gmask, lane_mask);
    xor_mask(deriv[a], gmask);
    bits[a] ^= g0;
  }

  out = 0;
  for (int i = 0; i < 64; i++) {
    if (bits[i]) out |= 1ULL << i;
    rows[i] = deriv[i];
  }
}

static bool solve_linear_random(std::array<Mask, 64> rows_in, uint64_t rhs64,
                                int cols, std::mt19937_64 &rng, Mask &delta) {
  std::array<uint8_t, 64> rhs{};
  for (int r = 0; r < 64; r++) rhs[r] = (rhs64 >> r) & 1ULL;
  std::array<int, 64> pivots{};
  pivots.fill(-1);
  int rank = 0;
  for (int c = 0; c < cols && rank < 64; c++) {
    int piv = -1;
    for (int r = rank; r < 64; r++) {
      if (bit_mask(rows_in[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows_in[rank], rows_in[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < 64; r++) {
      if (r != rank && bit_mask(rows_in[r], c)) {
        xor_mask(rows_in[r], rows_in[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivots[rank++] = c;
  }
  for (int r = rank; r < 64; r++) {
    if (zero_mask(rows_in[r]) && rhs[r]) return false;
  }

  Mask pivot_mask{0, 0};
  for (int r = 0; r < rank; r++) set_mask(pivot_mask, pivots[r]);
  Mask cm = low_mask(cols);
  delta = {rng() & cm[0] & ~pivot_mask[0], rng() & cm[1] & ~pivot_mask[1]};
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivots[r];
    Mask without = rows_in[r];
    clear_mask(without, c);
    uint8_t bit = rhs[r] ^ parity_and(without, delta);
    if (bit)
      set_mask(delta, c);
    else
      clear_mask(delta, c);
  }
  return true;
}

static std::string hex_original_from_x(const Mask &x, int original_n,
                                       int free_bits) {
  std::vector<uint8_t> bits(original_n, 0);
  for (int i = 0; i < free_bits && i < original_n; i++) bits[i] = bit_mask(x, i);
  int nibbles = (original_n + 3) / 4;
  std::string out(nibbles, '0');
  for (int nib = 0; nib < nibbles; nib++) {
    int v = 0;
    for (int j = 0; j < 4; j++) {
      int bit = nib * 4 + j;
      if (bit < original_n && bits[bit]) v |= 1 << j;
    }
    out[nibbles - 1 - nib] = "0123456789abcdef"[v];
  }
  return "0x" + out;
}

static std::string hex_bits(const std::vector<uint8_t> &bits, int lo,
                            int count) {
  int nibbles = (count + 3) / 4;
  std::string out(nibbles, '0');
  for (int nib = 0; nib < nibbles; nib++) {
    int v = 0;
    for (int j = 0; j < 4; j++) {
      int bit = nib * 4 + j;
      if (bit < count && bits[lo + bit]) v |= 1 << j;
    }
    out[nibbles - 1 - nib] = "0123456789abcdef"[v];
  }
  return "0x" + out;
}

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "usage: search_forward_low64_orig_lz_gadget F.txt total_wires "
                 "original_wires target_hex leading_zero_bits seconds [threads=4] [seed]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  int total_n = parse_int(argv[2], "total wire count");
  int original_n = parse_int(argv[3], "original wire count");
  uint64_t target = parse_hex_low64(argv[4]);
  int leading_zero_bits = parse_int(argv[5], "leading zero bits");
  int seconds = parse_int(argv[6], "seconds");
  int threads = argc > 7 ? parse_int(argv[7], "threads") : 4;
  uint64_t seed = argc > 8 ? std::stoull(argv[8], nullptr, 0) : 0xbb67ae8584caa73bULL;
  if (total_n <= 0 || total_n > 256 || original_n <= 0 || original_n > total_n ||
      leading_zero_bits > original_n) {
    throw std::runtime_error("bad dimensions");
  }
  int free_bits = original_n - leading_zero_bits;
  if (free_bits <= 0 || free_bits > 128) throw std::runtime_error("bad free bit count");
  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= total_n || b < 0 || b >= total_n || c < 0 || c >= total_n) {
      throw std::runtime_error("wire out of range");
    }
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::mutex mu;
  int global_best = 65;
  Mask global_best_x{0, 0};

  auto report = [&](int sc, const Mask &x) {
    std::lock_guard<std::mutex> lock(mu);
    if (sc < global_best) {
      global_best = sc;
      global_best_x = x;
      std::cerr << "best " << sc << " x "
                << hex_original_from_x(x, original_n, free_bits) << "\n";
    }
  };

  auto try_solution = [&](const Mask &x, uint64_t out) {
    if (out == target) {
      std::lock_guard<std::mutex> lock(mu);
      if (!found.exchange(true)) {
        std::vector<uint8_t> full;
        uint64_t checked = low64_eval(x, total_n, free_bits, gates, &full);
        std::cout << "x_original "
                  << hex_original_from_x(x, original_n, free_bits) << "\n";
        std::cout << "output_low_original " << hex_bits(full, 0, original_n)
                  << "\n";
        std::cout << "z " << hex_bits(full, 64, original_n - 64) << "\n";
        std::cout << "low_output 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << checked << std::dec << "\n";
        std::cout << "target 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << target << std::dec << "\n";
        std::cout << "verified " << (checked == target ? "yes" : "no") << "\n";
      }
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (tid + 1)));
    Mask cm = low_mask(free_bits);
    Mask x{rng() & cm[0], rng() & cm[1]};
    while (!found.load() && std::chrono::steady_clock::now() < deadline) {
      uint64_t out = 0;
      std::array<Mask, 64> rows;
      eval_low64_jac(x, total_n, free_bits, gates, out, rows);
      uint64_t residual = out ^ target;
      int base_score = pop64(residual);
      report(base_score, x);
      if (!base_score) {
        try_solution(x, out);
        return;
      }

      Mask best_x = x;
      int best_score = base_score;
      int attempts = base_score <= 16 ? 4096 : 768;
      bool got = false;
      for (int attempt = 0; attempt < attempts && !found.load(); attempt++) {
        Mask delta{0, 0};
        if (!solve_linear_random(rows, residual, free_bits, rng, delta)) break;
        got = true;
        Mask cand{x[0] ^ delta[0], x[1] ^ delta[1]};
        cand[0] &= cm[0];
        cand[1] &= cm[1];
        uint64_t cand_out = low64_eval(cand, total_n, free_bits, gates);
        int sc = pop64(cand_out ^ target);
        if (!sc) {
          try_solution(cand, cand_out);
          return;
        }
        if (sc < best_score || (sc <= best_score + 1 && (rng() & 31) == 0)) {
          best_score = sc;
          best_x = cand;
        }
      }
      if (got && (best_score <= base_score || (rng() & 15) == 0)) {
        x = best_x;
      } else {
        x = {rng() & cm[0], rng() & cm[1]};
      }
    }
  };

  std::vector<std::thread> workers;
  for (int i = 0; i < threads; i++) workers.emplace_back(worker, i);
  for (auto &t : workers) t.join();
  if (!found.load()) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_x " << hex_original_from_x(global_best_x, original_n, free_bits)
              << "\n";
    return 1;
  }
  return 0;
}
