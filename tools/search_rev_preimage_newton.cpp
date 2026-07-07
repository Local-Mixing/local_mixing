#include <array>
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

static uint64_t parse_u64(const std::string &s) {
  uint64_t out = 0;
  std::stringstream ss(s);
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0)
    ss >> std::hex >> out;
  else
    ss >> out;
  if (!ss) throw std::runtime_error("bad target");
  return out;
}

static std::array<uint8_t, 128> reverse_eval(uint64_t hi, uint64_t lo,
                                             const std::vector<Gate> &gates) {
  std::array<uint8_t, 128> bits{};
  for (int i = 0; i < 64; i++) bits[i] = (lo >> i) & 1ULL;
  for (int i = 0; i < 64; i++) bits[64 + i] = (hi >> i) & 1ULL;
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    bits[a] ^= bits[b] | !bits[c];
  }
  return bits;
}

static std::array<uint8_t, 128> forward_eval(std::array<uint8_t, 128> bits,
                                             const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  return bits;
}

static uint64_t top50_mask(uint64_t hi, uint64_t target,
                           const std::vector<Gate> &gates) {
  auto bits = reverse_eval(hi, target, gates);
  uint64_t out = 0;
  for (int i = 0; i < 50; i++) {
    if (bits[78 + i]) out |= 1ULL << i;
  }
  return out;
}

static int popcount64(uint64_t x) { return __builtin_popcountll(x); }

static uint64_t low64(const std::array<uint8_t, 128> &bits) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) {
    if (bits[i]) out |= 1ULL << i;
  }
  return out;
}

static std::string hex128(const std::array<uint8_t, 128> &bits) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int chunk = 3; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++) {
      if (bits[chunk * 32 + i]) v |= 1u << i;
    }
    out << std::setw(8) << v;
  }
  return out.str();
}

static bool solve_linear_random(const uint64_t cols[64], uint64_t rhs,
                                std::mt19937_64 &rng, uint64_t &delta) {
  uint64_t row_coeff[50]{};
  uint8_t row_rhs[50]{};
  for (int r = 0; r < 50; r++) row_rhs[r] = (rhs >> r) & 1ULL;
  for (int c = 0; c < 64; c++) {
    uint64_t col = cols[c];
    for (int r = 0; r < 50; r++) {
      if ((col >> r) & 1ULL) row_coeff[r] |= 1ULL << c;
    }
  }

  int pivot_col[50];
  for (int i = 0; i < 50; i++) pivot_col[i] = -1;
  int rank = 0;
  for (int c = 0; c < 64 && rank < 50; c++) {
    int piv = -1;
    for (int r = rank; r < 50; r++) {
      if ((row_coeff[r] >> c) & 1ULL) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(row_coeff[rank], row_coeff[piv]);
    std::swap(row_rhs[rank], row_rhs[piv]);
    for (int r = 0; r < 50; r++) {
      if (r != rank && ((row_coeff[r] >> c) & 1ULL)) {
        row_coeff[r] ^= row_coeff[rank];
        row_rhs[r] ^= row_rhs[rank];
      }
    }
    pivot_col[rank] = c;
    rank++;
  }
  for (int r = rank; r < 50; r++) {
    if (!row_coeff[r] && row_rhs[r]) return false;
  }

  uint64_t is_pivot = 0;
  for (int r = 0; r < rank; r++) is_pivot |= 1ULL << pivot_col[r];
  delta = rng() & ~is_pivot;
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivot_col[r];
    uint8_t bit = row_rhs[r] ^ (popcount64(row_coeff[r] & delta) & 1);
    if (bit)
      delta |= 1ULL << c;
    else
      delta &= ~(1ULL << c);
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_rev_preimage_newton F.txt target64 seconds [threads=4] [seed_hi]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  uint64_t seed_hi = argc > 5 ? parse_u64(argv[5]) : 0;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

  bool found = false;
  std::mutex mu;
  int global_best = 999;
  uint64_t global_best_hi = 0;

  auto report = [&](int score, uint64_t hi) {
    std::lock_guard<std::mutex> lock(mu);
    if (score < global_best) {
      global_best = score;
      global_best_hi = hi;
      std::cerr << "best " << score << " hi 0x" << std::hex << std::setw(16)
                << std::setfill('0') << hi << std::dec << "\n";
    }
  };

  auto try_solution = [&](uint64_t hi) {
    auto x = reverse_eval(hi, target, gates);
    auto y = forward_eval(x, gates);
    if (low64(y) == target && top50_mask(hi, target, gates) == 0) {
      std::lock_guard<std::mutex> lock(mu);
      if (!found) {
        found = true;
        std::cout << "x " << hex128(x) << "\n";
        std::cout << "z 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << hi << std::dec << "\n";
        std::cout << "low64_Fx 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << low64(y) << std::dec << "\n";
        std::cout << "verified yes\n";
      }
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(0x6a09e667f3bcc909ULL ^ (uint64_t)tid);
    uint64_t hi = seed_hi && tid == 0 ? seed_hi : rng();
    while (!found && std::chrono::steady_clock::now() < deadline) {
      uint64_t base = top50_mask(hi, target, gates);
      int base_score = popcount64(base);
      report(base_score, hi);
      if (!base) {
        try_solution(hi);
        break;
      }

      uint64_t cols[64];
      for (int c = 0; c < 64; c++)
        cols[c] = top50_mask(hi ^ (1ULL << c), target, gates) ^ base;

      uint64_t best_hi = hi;
      int best_score = base_score;
      for (int attempt = 0; attempt < 8192; attempt++) {
        uint64_t delta = 0;
        if (!solve_linear_random(cols, base, rng, delta)) break;
        uint64_t cand = hi ^ delta;
        int sc = popcount64(top50_mask(cand, target, gates));
        if (sc < best_score || (sc == best_score && (rng() & 15) == 0)) {
          best_score = sc;
          best_hi = cand;
        }
        if (!sc) {
          try_solution(cand);
          return;
        }
      }

      if (best_score <= base_score || (rng() & 7) == 0) {
        hi = best_hi;
      } else {
        hi ^= rng();
      }

      // A short noisy local step.
      for (int j = 0; j < 32 && !found; j++) {
        int bit = rng() & 63;
        uint64_t cand = hi ^ (1ULL << bit);
        int sc = popcount64(top50_mask(cand, target, gates));
        int cur = popcount64(top50_mask(hi, target, gates));
        if (sc <= cur || (rng() & 31) == 0) hi = cand;
      }
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_hi 0x" << std::hex << std::setw(16)
              << std::setfill('0') << global_best_hi << std::dec << "\n";
    return 1;
  }
  return 0;
}
