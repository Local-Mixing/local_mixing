#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
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
using U128 = std::array<uint64_t, 2>;

static int val(char c) {
  const std::string chars =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
  auto p = chars.find(c);
  if (p == std::string::npos) return -1;
  return static_cast<int>(p);
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

static int hex_val(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
}

static U128 parse_low128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  U128 out{0, 0};
  for (int bit = 0; bit < 128; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) out[bit >> 6] |= 1ULL << (bit & 63);
  }
  return out;
}

static bool parse_hex_u(std::string s, U128 &u) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) return false;
  while (s.size() < 32) s = "0" + s;
  uint64_t hi = 0, lo = 0;
  std::stringstream hs(s.substr(0, 16));
  std::stringstream ls(s.substr(16, 16));
  hs >> std::hex >> hi;
  ls >> std::hex >> lo;
  if (!hs || !ls) return false;
  u = {lo, hi};
  return true;
}

static std::string hex_u(const U128 &u) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(16) << u[1]
      << std::setw(16) << u[0];
  return out.str();
}

static std::string hex_bits(const std::array<uint8_t, 256> &bits) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int chunk = 7; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++) {
      if (bits[chunk * 32 + i]) v |= 1u << i;
    }
    out << std::setw(8) << v;
  }
  return out.str();
}

static bool bit_u(const U128 &u, int bit) {
  return (u[bit >> 6] >> (bit & 63)) & 1ULL;
}

static void set_u(U128 &u, int bit) { u[bit >> 6] |= 1ULL << (bit & 63); }
static void xor_u(U128 &a, const U128 &b) {
  a[0] ^= b[0];
  a[1] ^= b[1];
}

static int pop_u(const U128 &u) {
  return __builtin_popcountll(u[0]) + __builtin_popcountll(u[1]);
}

static uint8_t parity_and(const U128 &a, const U128 &b) {
  return (__builtin_popcountll(a[0] & b[0]) ^
          __builtin_popcountll(a[1] & b[1])) &
         1;
}

static std::array<uint8_t, 256> reverse_eval(const U128 &u, const U128 &low,
                                             const std::vector<Gate> &gates) {
  std::array<uint8_t, 256> bits{};
  for (int i = 0; i < 128; i++) bits[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
  for (int i = 0; i < 128; i++) bits[128 + i] = bit_u(u, i);
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    bits[a] ^= bits[b] | !bits[c];
  }
  return bits;
}

static std::array<uint8_t, 256> forward_eval(std::array<uint8_t, 256> bits,
                                             const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  return bits;
}

static int score_x(const std::array<uint8_t, 256> &bits) {
  int out = 0;
  for (int i = 156; i < 256; i++) out += bits[i] != 0;
  return out;
}

static void eval_zero_jac(const U128 &u, const U128 &low,
                          const std::vector<Gate> &gates,
                          std::array<uint8_t, 100> &rhs,
                          std::array<U128, 100> &rows) {
  std::array<uint8_t, 256> bits{};
  std::array<U128, 256> deriv{};
  for (int i = 0; i < 128; i++) bits[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
  for (int i = 0; i < 128; i++) {
    bits[128 + i] = bit_u(u, i);
    set_u(deriv[128 + i], i);
  }
  const U128 lane_mask{~0ULL, ~0ULL};
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    uint8_t g0 = bits[b] | !bits[c];
    U128 bmask = deriv[b], cmask = deriv[c];
    if (bits[b]) {
      bmask[0] ^= lane_mask[0];
      bmask[1] ^= lane_mask[1];
    }
    if (bits[c]) {
      cmask[0] ^= lane_mask[0];
      cmask[1] ^= lane_mask[1];
    }
    U128 gmask{bmask[0] | (~cmask[0] & lane_mask[0]),
               bmask[1] | (~cmask[1] & lane_mask[1])};
    if (g0) {
      gmask[0] ^= lane_mask[0];
      gmask[1] ^= lane_mask[1];
    }
    xor_u(deriv[a], gmask);
    bits[a] ^= g0;
  }
  for (int i = 0; i < 100; i++) {
    rhs[i] = bits[156 + i];
    rows[i] = deriv[156 + i];
  }
}

static bool affine_space(std::array<U128, 100> rows_in,
                         std::array<uint8_t, 100> rhs, U128 &particular,
                         std::vector<U128> &basis) {
  std::array<int, 100> pivot_col{};
  pivot_col.fill(-1);
  int rank = 0;
  for (int c = 0; c < 128 && rank < 100; c++) {
    int piv = -1;
    for (int r = rank; r < 100; r++) {
      if (bit_u(rows_in[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows_in[rank], rows_in[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < 100; r++) {
      if (r != rank && bit_u(rows_in[r], c)) {
        xor_u(rows_in[r], rows_in[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivot_col[rank++] = c;
  }
  for (int r = rank; r < 100; r++) {
    if (!(rows_in[r][0] || rows_in[r][1]) && rhs[r]) return false;
  }
  U128 is_pivot{0, 0};
  for (int r = 0; r < rank; r++) set_u(is_pivot, pivot_col[r]);

  particular = {0, 0};
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivot_col[r];
    U128 without = rows_in[r];
    without[c >> 6] &= ~(1ULL << (c & 63));
    if (rhs[r] ^ parity_and(without, particular)) set_u(particular, c);
  }

  basis.clear();
  for (int c = 0; c < 128; c++) {
    if (bit_u(is_pivot, c)) continue;
    U128 v{0, 0};
    set_u(v, c);
    for (int r = rank - 1; r >= 0; r--) {
      if (bit_u(rows_in[r], c)) set_u(v, pivot_col[r]);
    }
    basis.push_back(v);
  }
  return true;
}

struct BatchBest {
  int score = 999;
  U128 u{0, 0};
};

static void eval_batch(const std::vector<U128> &us, int start, int count,
                       const U128 &low, const std::vector<Gate> &gates,
                       BatchBest &best) {
  std::array<uint64_t, 256> bits{};
  const uint64_t lane_mask = count == 64 ? ~0ULL : ((1ULL << count) - 1);
  for (int i = 0; i < 128; i++) {
    bits[i] = ((low[i >> 6] >> (i & 63)) & 1ULL) ? lane_mask : 0ULL;
  }
  for (int lane = 0; lane < count; lane++) {
    const U128 &u = us[start + lane];
    for (int i = 0; i < 128; i++) {
      if (bit_u(u, i)) bits[128 + i] |= 1ULL << lane;
    }
  }
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    bits[a] ^= bits[b] | (~bits[c] & lane_mask);
  }
  for (int lane = 0; lane < count; lane++) {
    int sc = 0;
    for (int i = 156; i < 256; i++) sc += (bits[i] >> lane) & 1ULL;
    if (sc < best.score) {
      best.score = sc;
      best.u = us[start + lane];
    }
  }
}

static void verify_print(const U128 &u, const U128 &low,
                         const std::vector<Gate> &gates,
                         std::atomic<bool> &found, std::mutex &mu) {
  auto x = reverse_eval(u, low, gates);
  if (score_x(x)) return;
  auto y = forward_eval(x, gates);
  for (int i = 0; i < 128; i++) {
    if (y[i] != ((low[i >> 6] >> (i & 63)) & 1ULL)) return;
  }
  std::lock_guard<std::mutex> lock(mu);
  if (!found.exchange(true)) {
    std::cout << "x " << hex_bits(x) << "\n";
    std::cout << "U " << hex_u(u) << "\n";
    std::cout << "output " << hex_bits(y) << "\n";
    std::cout << "verified yes\n";
  }
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "usage: search_reverse_lowtarget_leading0_linear_space_batch F.txt target128 center_U threads seconds\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  auto low = parse_low128(argv[2]);
  U128 center{0, 0};
  if (!parse_hex_u(argv[3], center)) throw std::runtime_error("bad center U");
  int threads = std::stoi(argv[4]);
  int seconds = std::stoi(argv[5]);
  std::array<uint8_t, 100> rhs;
  std::array<U128, 100> rows;
  eval_zero_jac(center, low, gates, rhs, rows);
  U128 particular{0, 0};
  std::vector<U128> basis;
  if (!affine_space(rows, rhs, particular, basis)) {
    std::cerr << "linear system inconsistent\n";
    return 1;
  }
  if (basis.size() >= 63) throw std::runtime_error("basis too large");
  uint64_t total = 1ULL << basis.size();
  std::cerr << "basis_dim " << basis.size() << " total " << total
            << " particular_weight " << pop_u(particular) << "\n";

  std::atomic<bool> found(false);
  std::atomic<uint64_t> done(0);
  std::mutex mu;
  BatchBest global_best;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

  auto worker = [&](int tid) {
    uint64_t per = (total + threads - 1) / threads;
    uint64_t begin = std::min<uint64_t>(total, per * tid);
    uint64_t end = std::min<uint64_t>(total, begin + per);
    std::vector<U128> batch;
    batch.reserve(4096);
    auto push_idx = [&](uint64_t idx) {
      uint64_t gray = idx ^ (idx >> 1);
      U128 delta = particular;
      for (int j = 0; j < static_cast<int>(basis.size()); j++) {
        if ((gray >> j) & 1ULL) xor_u(delta, basis[j]);
      }
      U128 u{center[0] ^ delta[0], center[1] ^ delta[1]};
      batch.push_back(u);
    };
    auto flush = [&]() {
      for (int i = 0; i < static_cast<int>(batch.size()); i += 64) {
        BatchBest br;
        int count = std::min(64, static_cast<int>(batch.size()) - i);
        eval_batch(batch, i, count, low, gates, br);
        done += count;
        if (br.score < global_best.score) {
          std::lock_guard<std::mutex> lock(mu);
          if (br.score < global_best.score) {
            global_best = br;
            std::cerr << "best " << br.score << " U " << hex_u(br.u)
                      << " done " << done.load() << "\n";
          }
        }
        if (br.score == 0) verify_print(br.u, low, gates, found, mu);
        if (found) return;
      }
      batch.clear();
    };
    for (uint64_t idx = begin; idx < end && !found; idx++) {
      if (std::chrono::steady_clock::now() >= deadline) break;
      push_idx(idx);
      if (batch.size() >= 4096) flush();
    }
    if (!batch.empty() && !found) flush();
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best.score << "\n";
    std::cout << "best_U " << hex_u(global_best.u) << "\n";
    std::cout << "done " << done.load() << "\n";
    return 1;
  }
  return 0;
}
