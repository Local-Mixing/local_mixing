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

static std::vector<uint8_t> output_bits(int n, int low_bits, uint64_t low,
                                        uint64_t u) {
  std::vector<uint8_t> bits(n);
  for (int i = 0; i < low_bits; i++) bits[i] = (low >> i) & 1ULL;
  for (int i = low_bits; i < n; i++) bits[i] = (u >> (i - low_bits)) & 1ULL;
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

static int leading_zeros(const std::vector<uint8_t> &bits) {
  int out = 0;
  for (int i = static_cast<int>(bits.size()) - 1; i >= 0; --i) {
    if (bits[i]) break;
    out++;
  }
  return out;
}

static bool smaller_bits(const std::vector<uint8_t> &a,
                         const std::vector<uint8_t> &b) {
  for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
    if (a[i] != b[i]) return a[i] < b[i];
  }
  return false;
}

static void eval_reverse_jac(uint64_t u, int n, int low_bits, uint64_t low,
                             const std::vector<Gate> &gates,
                             std::vector<uint8_t> &bits,
                             std::vector<uint64_t> &deriv) {
  bits = output_bits(n, low_bits, low, u);
  deriv.assign(n, 0);
  int u_bits = n - low_bits;
  uint64_t lane_mask = u_bits == 64 ? ~0ULL : ((1ULL << u_bits) - 1);
  for (int i = low_bits; i < n; i++) deriv[i] = 1ULL << (i - low_bits);

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    uint8_t g0 = bits[b] | !bits[c];
    uint64_t bmask = deriv[b];
    uint64_t cmask = deriv[c];
    if (bits[b]) bmask ^= lane_mask;
    if (bits[c]) cmask ^= lane_mask;
    uint64_t gmask = bmask | (~cmask & lane_mask);
    if (g0) gmask ^= lane_mask;
    deriv[a] ^= gmask;
    bits[a] ^= g0;
  }
}

static bool solve_linear_random(std::vector<uint64_t> rows,
                                std::vector<uint8_t> rhs, int cols,
                                std::mt19937_64 &rng, uint64_t &delta) {
  std::vector<int> pivots;
  int rank = 0;
  for (int c = 0; c < cols && rank < static_cast<int>(rows.size()); c++) {
    int piv = -1;
    for (int r = rank; r < static_cast<int>(rows.size()); r++) {
      if ((rows[r] >> c) & 1ULL) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows[rank], rows[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < static_cast<int>(rows.size()); r++) {
      if (r != rank && ((rows[r] >> c) & 1ULL)) {
        rows[r] ^= rows[rank];
        rhs[r] ^= rhs[rank];
      }
    }
    pivots.push_back(c);
    rank++;
  }
  for (int r = rank; r < static_cast<int>(rows.size()); r++) {
    if (!rows[r] && rhs[r]) return false;
  }

  uint64_t pivot_mask = 0;
  for (int c : pivots) pivot_mask |= 1ULL << c;
  uint64_t col_mask = cols == 64 ? ~0ULL : ((1ULL << cols) - 1);
  delta = rng() & col_mask & ~pivot_mask;
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivots[r];
    uint64_t without = rows[r] & ~(1ULL << c);
    uint8_t bit = rhs[r] ^ (__builtin_popcountll(without & delta) & 1);
    if (bit)
      delta |= 1ULL << c;
    else
      delta &= ~(1ULL << c);
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    std::cerr << "usage: search_reverse_lowtarget_prefix_generic F.txt n target_hex "
                 "target_bits leading_zero_goal seconds [threads=4] [seed]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  int n = parse_int(argv[2], "wire count");
  uint64_t target = parse_hex_low64(argv[3]);
  int low_bits = parse_int(argv[4], "target bits");
  int goal = parse_int(argv[5], "leading zero goal");
  int seconds = parse_int(argv[6], "seconds");
  int threads = argc > 7 ? parse_int(argv[7], "threads") : 4;
  uint64_t seed = argc > 8 ? std::stoull(argv[8], nullptr, 0) : 0x9e3779b97f4a7c15ULL;
  if (n <= 0 || n > 128) throw std::runtime_error("bad n");
  if (low_bits != 64 || n - low_bits <= 0 || n - low_bits > 64) {
    throw std::runtime_error("this helper expects 64 target bits and <=64 free bits");
  }
  if (goal < 0 || goal > n) throw std::runtime_error("bad leading-zero goal");
  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) {
      throw std::runtime_error("wire out of range");
    }
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found_goal(false);
  std::mutex mu;
  int best_lz = -1;
  std::vector<uint8_t> best_bits;
  uint64_t best_u = 0;

  auto report = [&](uint64_t u, const std::vector<uint8_t> &x) {
    int lz = leading_zeros(x);
    std::lock_guard<std::mutex> lock(mu);
    if (lz > best_lz || (lz == best_lz && (best_bits.empty() || smaller_bits(x, best_bits)))) {
      best_lz = lz;
      best_bits = x;
      best_u = u;
      auto y = forward_eval(x, gates);
      std::cout << "best_lz " << best_lz << " x " << hex_bits(x, 0, n)
                << " z " << hex_bits(y, low_bits, n - low_bits)
                << " low " << hex_bits(y, 0, low_bits) << "\n";
      std::cout.flush();
      if (best_lz >= goal) found_goal.store(true);
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0xbf58476d1ce4e5b9ULL * (tid + 1)));
    int u_bits = n - low_bits;
    uint64_t u_mask = u_bits == 64 ? ~0ULL : ((1ULL << u_bits) - 1);
    uint64_t u = rng() & u_mask;
    if (tid == 0) u = 0;
    std::vector<uint8_t> bits;
    std::vector<uint64_t> deriv;
    while (!found_goal.load() && std::chrono::steady_clock::now() < deadline) {
      eval_reverse_jac(u, n, low_bits, target, gates, bits, deriv);
      report(u, bits);

      int rows_count = std::min(goal, n);
      std::vector<uint64_t> rows;
      std::vector<uint8_t> rhs;
      rows.reserve(rows_count);
      rhs.reserve(rows_count);
      for (int i = n - rows_count; i < n; i++) {
        rows.push_back(deriv[i]);
        rhs.push_back(bits[i]);
      }

      uint64_t best_local_u = u;
      auto best_local_bits = bits;
      int best_local_lz = leading_zeros(bits);
      int attempts = best_local_lz + 8 >= goal ? 4096 : 768;
      bool got = false;
      for (int a = 0; a < attempts && !found_goal.load(); a++) {
        uint64_t delta = 0;
        if (!solve_linear_random(rows, rhs, u_bits, rng, delta)) break;
        got = true;
        uint64_t cand_u = (u ^ delta) & u_mask;
        auto cand = reverse_eval(output_bits(n, low_bits, target, cand_u), gates);
        int cand_lz = leading_zeros(cand);
        if (cand_lz > best_local_lz ||
            (cand_lz == best_local_lz && smaller_bits(cand, best_local_bits))) {
          best_local_lz = cand_lz;
          best_local_u = cand_u;
          best_local_bits = cand;
          report(cand_u, cand);
        }
      }
      if (got && (best_local_lz >= leading_zeros(bits) || (rng() & 7) == 0)) {
        u = best_local_u;
      } else {
        u = rng() & u_mask;
      }
      for (int j = 0; j < 16 && !found_goal.load(); j++) {
        uint64_t cand_u = (u ^ (1ULL << (rng() % u_bits))) & u_mask;
        auto cand = reverse_eval(output_bits(n, low_bits, target, cand_u), gates);
        if (leading_zeros(cand) >= best_local_lz || (rng() & 15) == 0) {
          u = cand_u;
          report(cand_u, cand);
          break;
        }
      }
    }
  };

  std::vector<std::thread> workers;
  for (int t = 0; t < threads; t++) workers.emplace_back(worker, t);
  for (auto &worker_thread : workers) worker_thread.join();

  std::lock_guard<std::mutex> lock(mu);
  if (best_lz >= 0) {
    auto y = forward_eval(best_bits, gates);
    std::cout << "final_lz " << best_lz << "\n";
    std::cout << "x " << hex_bits(best_bits, 0, n) << "\n";
    std::cout << "output " << hex_bits(y, 0, n) << "\n";
    std::cout << "z " << hex_bits(y, low_bits, n - low_bits) << "\n";
    std::cout << "low_output " << hex_bits(y, 0, low_bits) << "\n";
    std::cout << "target 0x" << std::hex << std::setw(16) << std::setfill('0') << target
              << std::dec << "\n";
    std::cout << "verified "
              << ((hex_bits(y, 0, low_bits).substr(2) ==
                   (static_cast<std::ostringstream &&>(std::ostringstream() << std::hex
                                                         << std::setw(16) << std::setfill('0')
                                                         << target))
                       .str())
                      ? "yes"
                      : "check-low-output")
              << "\n";
    std::cout << "u 0x" << std::hex << std::setw(16) << std::setfill('0') << best_u
              << std::dec << "\n";
  }
  return 0;
}
