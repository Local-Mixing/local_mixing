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

static std::array<uint64_t, 2> parse_low128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  std::array<uint64_t, 2> out{0, 0};
  for (int bit = 0; bit < 128; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) out[bit >> 6] |= 1ULL << (bit & 63);
  }
  for (int bit = 128; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) throw std::runtime_error("target too wide");
  }
  return out;
}

static int pop2(std::array<uint64_t, 2> x) {
  return __builtin_popcountll(x[0]) + __builtin_popcountll(x[1]);
}

static int pop3(const Mask &x) {
  return __builtin_popcountll(x[0]) + __builtin_popcountll(x[1]) +
         __builtin_popcountll(x[2]);
}

static bool bit_mask(const Mask &m, int bit) {
  return (m[bit >> 6] >> (bit & 63)) & 1ULL;
}

static void set_mask(Mask &m, int bit) { m[bit >> 6] |= 1ULL << (bit & 63); }

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

static std::array<uint8_t, 256> bits_from_x(const Mask &x) {
  std::array<uint8_t, 256> bits{};
  for (int i = 0; i < 156; i++) bits[i] = bit_mask(x, i);
  return bits;
}

static std::array<uint64_t, 2> low128_eval(const Mask &x,
                                           const std::vector<Gate> &gates) {
  auto bits = bits_from_x(x);
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  std::array<uint64_t, 2> out{0, 0};
  for (int i = 0; i < 128; i++) {
    if (bits[i]) out[i >> 6] |= 1ULL << (i & 63);
  }
  return out;
}

static void eval_low128_jac(const Mask &x, const std::vector<Gate> &gates,
                            std::array<uint64_t, 2> &out,
                            std::array<Mask, 128> &rows) {
  std::array<uint8_t, 256> bits = bits_from_x(x);
  std::array<Mask, 256> deriv{};
  for (int i = 0; i < 156; i++) set_mask(deriv[i], i);
  const Mask lane_mask{~0ULL, ~0ULL, (1ULL << 28) - 1};

  for (auto [a, b, c] : gates) {
    uint8_t g0 = bits[b] | !bits[c];
    Mask bmask = deriv[b];
    Mask cmask = deriv[c];
    if (bits[b]) {
      bmask[0] ^= lane_mask[0];
      bmask[1] ^= lane_mask[1];
      bmask[2] ^= lane_mask[2];
    }
    if (bits[c]) {
      cmask[0] ^= lane_mask[0];
      cmask[1] ^= lane_mask[1];
      cmask[2] ^= lane_mask[2];
    }
    Mask gmask{bmask[0] | (~cmask[0] & lane_mask[0]),
               bmask[1] | (~cmask[1] & lane_mask[1]),
               bmask[2] | (~cmask[2] & lane_mask[2])};
    if (g0) {
      gmask[0] ^= lane_mask[0];
      gmask[1] ^= lane_mask[1];
      gmask[2] ^= lane_mask[2];
    }
    xor_mask(deriv[a], gmask);
    bits[a] ^= g0;
  }

  out = {0, 0};
  for (int i = 0; i < 128; i++) {
    if (bits[i]) out[i >> 6] |= 1ULL << (i & 63);
    rows[i] = deriv[i];
  }
}

static bool solve_linear_random(std::array<Mask, 128> rows_in,
                                std::array<uint64_t, 2> rhs_in,
                                std::mt19937_64 &rng, Mask &delta) {
  std::array<uint8_t, 128> rhs{};
  for (int r = 0; r < 128; r++) rhs[r] = (rhs_in[r >> 6] >> (r & 63)) & 1ULL;

  std::array<int, 128> pivot_col{};
  pivot_col.fill(-1);
  int rank = 0;
  for (int c = 0; c < 156 && rank < 128; c++) {
    int piv = -1;
    for (int r = rank; r < 128; r++) {
      if (bit_mask(rows_in[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows_in[rank], rows_in[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < 128; r++) {
      if (r != rank && bit_mask(rows_in[r], c)) {
        xor_mask(rows_in[r], rows_in[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivot_col[rank++] = c;
  }
  for (int r = rank; r < 128; r++) {
    if (zero_mask(rows_in[r]) && rhs[r]) return false;
  }

  Mask is_pivot{0, 0, 0};
  for (int r = 0; r < rank; r++) set_mask(is_pivot, pivot_col[r]);
  delta = {rng(), rng(), rng() & ((1ULL << 28) - 1)};
  delta[0] &= ~is_pivot[0];
  delta[1] &= ~is_pivot[1];
  delta[2] &= ~is_pivot[2];
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivot_col[r];
    Mask without = rows_in[r];
    without[c >> 6] &= ~(1ULL << (c & 63));
    uint8_t bit = rhs[r] ^ parity_and(without, delta);
    if (bit)
      set_mask(delta, c);
    else
      delta[c >> 6] &= ~(1ULL << (c & 63));
  }
  return true;
}

static std::string hex256_from_x(const Mask &x) {
  auto bits = bits_from_x(x);
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

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_forward_lowtarget_leading0_newton_wide F.txt target128 seconds [threads=4] [seed]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  auto target = parse_low128(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  uint64_t seed = argc > 5 ? std::stoull(argv[5], nullptr, 0) : 0x243f6a8885a308d3ULL;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

  std::atomic<bool> found(false);
  std::mutex mu;
  int global_best = 999;
  Mask global_best_x{0, 0, 0};

  auto report = [&](int sc, const Mask &x) {
    std::lock_guard<std::mutex> lock(mu);
    if (sc < global_best) {
      global_best = sc;
      global_best_x = x;
      std::cerr << "best " << sc << " x " << hex256_from_x(x) << "\n";
    }
  };

  auto try_solution = [&](const Mask &x, std::array<uint64_t, 2> out) {
    if (out[0] == target[0] && out[1] == target[1]) {
      std::lock_guard<std::mutex> lock(mu);
      if (!found.exchange(true)) {
        std::cout << "x " << hex256_from_x(x) << "\n";
        std::cout << "low128_Fx 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << out[1] << std::setw(16) << out[0]
                  << std::dec << "\n";
        std::cout << "verified yes\n";
      }
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1)));
    Mask x{rng(), rng(), rng() & ((1ULL << 28) - 1)};
    while (!found && std::chrono::steady_clock::now() < deadline) {
      std::array<uint64_t, 2> out;
      std::array<Mask, 128> rows;
      eval_low128_jac(x, gates, out, rows);
      std::array<uint64_t, 2> residual{out[0] ^ target[0], out[1] ^ target[1]};
      int base_score = pop2(residual);
      report(base_score, x);
      if (!base_score) {
        try_solution(x, out);
        return;
      }

      Mask best_x = x;
      int best_score = base_score;
      bool got_linear = false;
      int attempts = base_score <= 24 ? 4096 : 512;
      for (int attempt = 0; attempt < attempts; attempt++) {
        Mask delta{0, 0, 0};
        if (!solve_linear_random(rows, residual, rng, delta)) break;
        got_linear = true;
        Mask cand{x[0] ^ delta[0], x[1] ^ delta[1], x[2] ^ delta[2]};
        auto cand_out = low128_eval(cand, gates);
        int sc = pop2({cand_out[0] ^ target[0], cand_out[1] ^ target[1]});
        if (!sc) {
          try_solution(cand, cand_out);
          return;
        }
        if (sc < best_score || (sc == best_score && pop3(delta) < 50) ||
            (sc <= best_score + 2 && (rng() & 63) == 0)) {
          best_score = sc;
          best_x = cand;
        }
      }

      if (got_linear && (best_score <= base_score || (rng() & 15) == 0)) {
        x = best_x;
      } else {
        x = {rng(), rng(), rng() & ((1ULL << 28) - 1)};
      }

      for (int j = 0; j < 24 && !found; j++) {
        int bit = rng() % 156;
        Mask cand = x;
        cand[bit >> 6] ^= 1ULL << (bit & 63);
        auto cand_out = low128_eval(cand, gates);
        auto cur_out = low128_eval(x, gates);
        int sc = pop2({cand_out[0] ^ target[0], cand_out[1] ^ target[1]});
        int cur = pop2({cur_out[0] ^ target[0], cur_out[1] ^ target[1]});
        if (!sc) {
          try_solution(cand, cand_out);
          return;
        }
        if (sc <= cur || (rng() & 127) == 0) x = cand;
      }
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_x " << hex256_from_x(global_best_x) << "\n";
    return 1;
  }
  return 0;
}
