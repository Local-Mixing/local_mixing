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
using Mask = std::array<uint64_t, 3>;

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
  a[2] ^= b[2];
}

static uint8_t parity_and(const Mask &a, const Mask &b) {
  return (__builtin_popcountll(a[0] & b[0]) ^
          __builtin_popcountll(a[1] & b[1]) ^
          __builtin_popcountll(a[2] & b[2])) &
         1;
}

static bool zero_mask(const Mask &m) { return !(m[0] || m[1] || m[2]); }

static Mask low_mask(int cols) {
  Mask out{0, 0, 0};
  for (int word = 0; word < 3; word++) {
    int remaining = cols - 64 * word;
    if (remaining >= 64) {
      out[word] = ~0ULL;
    } else if (remaining > 0) {
      out[word] = (1ULL << remaining) - 1;
    }
  }
  return out;
}

static std::vector<uint8_t> output_bits(int n, int low_bits, uint64_t low,
                                        const Mask &u) {
  std::vector<uint8_t> bits(n);
  for (int i = 0; i < low_bits; i++) bits[i] = (low >> i) & 1ULL;
  for (int i = low_bits; i < n; i++) {
    int j = i - low_bits;
    bits[i] = (u[j >> 6] >> (j & 63)) & 1ULL;
  }
  return bits;
}

static std::vector<uint8_t> reverse_eval(std::vector<uint8_t> bits,
                                         const std::vector<Gate> &gates) {
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    bits[a] ^= bits[b] | !bits[c];
  }
  return bits;
}

static std::vector<uint8_t> forward_eval(std::vector<uint8_t> bits,
                                         const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  return bits;
}

static std::string hex_bits(const std::vector<uint8_t> &bits, int lo,
                            int count) {
  int nibbles = (count + 3) / 4;
  std::string out(nibbles, '0');
  for (int nib = 0; nib < nibbles; nib++) {
    int value = 0;
    for (int j = 0; j < 4; j++) {
      int bit = nib * 4 + j;
      if (bit < count && bits[lo + bit]) value |= 1 << j;
    }
    out[nibbles - 1 - nib] = "0123456789abcdef"[value];
  }
  return "0x" + out;
}

static int leading_zeros_original(const std::vector<uint8_t> &bits,
                                  int original_n) {
  int out = 0;
  for (int i = original_n - 1; i >= 0; --i) {
    if (bits[i]) break;
    out++;
  }
  return out;
}

static bool smaller_original(const std::vector<uint8_t> &a,
                             const std::vector<uint8_t> &b, int original_n) {
  for (int i = original_n - 1; i >= 0; --i) {
    if (a[i] != b[i]) return a[i] < b[i];
  }
  return false;
}

static void eval_reverse_jac(const Mask &u, int n, int low_bits, uint64_t low,
                             const std::vector<Gate> &gates,
                             std::vector<uint8_t> &bits,
                             std::vector<Mask> &deriv) {
  bits = output_bits(n, low_bits, low, u);
  deriv.assign(n, Mask{0, 0, 0});
  int u_bits = n - low_bits;
  Mask lane_mask = low_mask(u_bits);
  for (int i = low_bits; i < n; i++) set_mask(deriv[i], i - low_bits);

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    uint8_t g0 = bits[b] | !bits[c];
    Mask bmask = deriv[b];
    Mask cmask = deriv[c];
    if (bits[b]) xor_mask(bmask, lane_mask);
    if (bits[c]) xor_mask(cmask, lane_mask);
    Mask gmask{bmask[0] | (~cmask[0] & lane_mask[0]),
               bmask[1] | (~cmask[1] & lane_mask[1]),
               bmask[2] | (~cmask[2] & lane_mask[2])};
    if (g0) xor_mask(gmask, lane_mask);
    xor_mask(deriv[a], gmask);
    bits[a] ^= g0;
  }
}

static bool solve_linear_random(std::vector<Mask> rows, std::vector<uint8_t> rhs,
                                int cols, std::mt19937_64 &rng, Mask &delta) {
  std::vector<int> pivots;
  int rank = 0;
  for (int c = 0; c < cols && rank < static_cast<int>(rows.size()); c++) {
    int piv = -1;
    for (int r = rank; r < static_cast<int>(rows.size()); r++) {
      if (bit_mask(rows[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows[rank], rows[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < static_cast<int>(rows.size()); r++) {
      if (r != rank && bit_mask(rows[r], c)) {
        xor_mask(rows[r], rows[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivots.push_back(c);
    rank++;
  }
  for (int r = rank; r < static_cast<int>(rows.size()); r++) {
    if (zero_mask(rows[r]) && rhs[r]) return false;
  }

  Mask pivot_mask{0, 0, 0};
  for (int c : pivots) set_mask(pivot_mask, c);
  Mask col_mask = low_mask(cols);
  delta = {rng() & col_mask[0] & ~pivot_mask[0],
           rng() & col_mask[1] & ~pivot_mask[1],
           rng() & col_mask[2] & ~pivot_mask[2]};
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivots[r];
    Mask without = rows[r];
    clear_mask(without, c);
    uint8_t bit = rhs[r] ^ parity_and(without, delta);
    if (bit)
      set_mask(delta, c);
    else
      clear_mask(delta, c);
  }
  return true;
}

static Mask random_u(int cols, std::mt19937_64 &rng) {
  Mask u{rng(), rng(), rng()};
  if (cols < 192) {
    for (int bit = cols; bit < 192; bit++) clear_mask(u, bit);
  }
  return u;
}

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "usage: search_reverse_low64_orig_lz_gadget F.txt total_wires "
                 "original_wires target_hex leading_zero_goal seconds [threads=4] [seed]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  int total_n = parse_int(argv[2], "total wire count");
  int original_n = parse_int(argv[3], "original wire count");
  uint64_t target = parse_hex_low64(argv[4]);
  int goal = parse_int(argv[5], "leading zero goal");
  int seconds = parse_int(argv[6], "seconds");
  int threads = argc > 7 ? parse_int(argv[7], "threads") : 4;
  uint64_t seed = argc > 8 ? std::stoull(argv[8], nullptr, 0) : 0x6a09e667f3bcc909ULL;
  const int low_bits = 64;
  if (total_n <= 0 || total_n > 256 || original_n <= 0 || original_n > total_n) {
    throw std::runtime_error("bad dimensions");
  }
  if (goal < 0 || goal > original_n) throw std::runtime_error("bad leading-zero goal");
  int free_bits = total_n - low_bits;
  if (free_bits <= 0 || free_bits > 192) throw std::runtime_error("bad free bit count");
  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= total_n || b < 0 || b >= total_n || c < 0 || c >= total_n) {
      throw std::runtime_error("wire out of range");
    }
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::mutex mu;
  int best_lz = -1;
  std::vector<uint8_t> best_input;
  std::vector<uint8_t> best_output;
  Mask best_u{0, 0, 0};

  auto report = [&](const Mask &u, const std::vector<uint8_t> &x) {
    int lz = leading_zeros_original(x, original_n);
    std::lock_guard<std::mutex> lock(mu);
    if (lz > best_lz ||
        (lz == best_lz &&
         (best_input.empty() || smaller_original(x, best_input, original_n)))) {
      best_lz = lz;
      best_input = x;
      best_u = u;
      auto y = forward_eval(x, gates);
      best_output = y;
      std::cout << "best_lz " << best_lz << " x_original "
                << hex_bits(x, 0, original_n) << " z "
                << hex_bits(y, low_bits, original_n - low_bits) << " low "
                << hex_bits(y, 0, low_bits) << "\n";
      std::cout.flush();
      if (best_lz >= goal) found.store(true);
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0xbf58476d1ce4e5b9ULL * (tid + 1)));
    Mask u = random_u(free_bits, rng);
    if (tid == 0) u = {0, 0, 0};
    std::vector<uint8_t> bits;
    std::vector<Mask> deriv;
    while (!found.load() && std::chrono::steady_clock::now() < deadline) {
      eval_reverse_jac(u, total_n, low_bits, target, gates, bits, deriv);
      report(u, bits);

      std::vector<Mask> rows;
      std::vector<uint8_t> rhs;
      rows.reserve(goal);
      rhs.reserve(goal);
      for (int i = original_n - goal; i < original_n; i++) {
        rows.push_back(deriv[i]);
        rhs.push_back(bits[i]);
      }

      Mask best_local_u = u;
      auto best_local_bits = bits;
      int best_local_lz = leading_zeros_original(bits, original_n);
      int attempts = best_local_lz + 8 >= goal ? 8192 : 1536;
      bool got = false;
      for (int a = 0; a < attempts && !found.load(); a++) {
        Mask delta{0, 0, 0};
        if (!solve_linear_random(rows, rhs, free_bits, rng, delta)) break;
        got = true;
        Mask cand_u{u[0] ^ delta[0], u[1] ^ delta[1], u[2] ^ delta[2]};
        for (int bit = free_bits; bit < 192; bit++) clear_mask(cand_u, bit);
        auto cand =
            reverse_eval(output_bits(total_n, low_bits, target, cand_u), gates);
        int cand_lz = leading_zeros_original(cand, original_n);
        if (cand_lz > best_local_lz ||
            (cand_lz == best_local_lz &&
             smaller_original(cand, best_local_bits, original_n))) {
          best_local_lz = cand_lz;
          best_local_u = cand_u;
          best_local_bits = cand;
          report(cand_u, cand);
        }
      }
      if (got && (best_local_lz >= leading_zeros_original(bits, original_n) ||
                  (rng() & 7) == 0)) {
        u = best_local_u;
      } else {
        u = random_u(free_bits, rng);
      }
    }
  };

  std::vector<std::thread> workers;
  for (int t = 0; t < threads; t++) workers.emplace_back(worker, t);
  for (auto &worker_thread : workers) worker_thread.join();

  std::lock_guard<std::mutex> lock(mu);
  if (best_lz >= 0) {
    auto y = forward_eval(best_input, gates);
    std::cout << "final_lz " << best_lz << "\n";
    std::cout << "x_original " << hex_bits(best_input, 0, original_n) << "\n";
    if (total_n > original_n) {
      std::cout << "aux_input "
                << hex_bits(best_input, original_n, total_n - original_n)
                << "\n";
    }
    std::cout << "input_full " << hex_bits(best_input, 0, total_n) << "\n";
    std::cout << "output_low_original " << hex_bits(y, 0, original_n) << "\n";
    std::cout << "z " << hex_bits(y, low_bits, original_n - low_bits) << "\n";
    std::cout << "low_output " << hex_bits(y, 0, low_bits) << "\n";
    std::cout << "target 0x" << std::hex << std::setw(16) << std::setfill('0')
              << target << std::dec << "\n";
    uint64_t low = 0;
    for (int i = 0; i < low_bits; i++) {
      if (y[i]) low |= 1ULL << i;
    }
    std::cout << "verified "
              << ((best_lz >= goal && low == target) ? "yes" : "no") << "\n";
    std::cout << "u_free_output " << hex_bits(output_bits(total_n, low_bits, target, best_u),
                                            low_bits, free_bits)
              << "\n";
    return (best_lz >= goal && low == target) ? 0 : 1;
  }
  std::cout << "not_found\n";
  return 1;
}
