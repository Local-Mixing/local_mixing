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

static std::array<uint8_t, 128> bits_from_x(uint64_t lo64, uint16_t hi14) {
  std::array<uint8_t, 128> bits{};
  for (int i = 0; i < 64; i++) bits[i] = (lo64 >> i) & 1ULL;
  for (int i = 0; i < 14; i++) bits[64 + i] = (hi14 >> i) & 1U;
  return bits;
}

static void eval(std::array<uint8_t, 128> &bits, const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
}

static uint64_t low64_eval(uint64_t lo64, uint16_t hi14,
                           const std::vector<Gate> &gates) {
  auto bits = bits_from_x(lo64, hi14);
  eval(bits, gates);
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) {
    if (bits[i]) out |= 1ULL << i;
  }
  return out;
}

static int score(uint64_t lo64, uint16_t hi14, uint64_t target,
                 const std::vector<Gate> &gates) {
  return __builtin_popcountll(low64_eval(lo64, hi14, gates) ^ target);
}

static std::string hex128(uint64_t lo64, uint16_t hi14) {
  std::array<uint8_t, 128> bits = bits_from_x(lo64, hi14);
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

static bool parse_hex128_seed(std::string s, uint64_t &lo64, uint16_t &hi14) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) return false;
  while (s.size() < 32) s = "0" + s;
  uint64_t high = 0;
  uint64_t low = 0;
  std::stringstream hs(s.substr(0, 16));
  std::stringstream ls(s.substr(16, 16));
  hs >> std::hex >> high;
  ls >> std::hex >> low;
  if (!hs || !ls) return false;
  lo64 = low;
  hi14 = high & ((1u << 14) - 1);
  return (high >> 14) == 0;
}

static int parity128(unsigned __int128 x) {
  uint64_t lo = static_cast<uint64_t>(x);
  uint64_t hi = static_cast<uint64_t>(x >> 64);
  return (__builtin_popcountll(lo) ^ __builtin_popcountll(hi)) & 1;
}

static bool solve_linear_random(const uint64_t cols[78], uint64_t rhs,
                                std::mt19937_64 &rng,
                                unsigned __int128 &delta) {
  unsigned __int128 rows[64]{};
  for (int r = 0; r < 64; r++) rows[r] = ((static_cast<unsigned __int128>((rhs >> r) & 1ULL)) << 78);
  for (int c = 0; c < 78; c++) {
    uint64_t col = cols[c];
    for (int r = 0; r < 64; r++) {
      if ((col >> r) & 1ULL) rows[r] |= static_cast<unsigned __int128>(1) << c;
    }
  }

  int pivot_col[64];
  for (int i = 0; i < 64; i++) pivot_col[i] = -1;
  int rank = 0;
  for (int c = 0; c < 78 && rank < 64; c++) {
    int piv = -1;
    for (int r = rank; r < 64; r++) {
      if ((rows[r] >> c) & 1U) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows[rank], rows[piv]);
    for (int r = 0; r < 64; r++) {
      if (r != rank && ((rows[r] >> c) & 1U)) rows[r] ^= rows[rank];
    }
    pivot_col[rank++] = c;
  }

  const unsigned __int128 coeff_mask = (static_cast<unsigned __int128>(1) << 78) - 1;
  for (int r = rank; r < 64; r++) {
    if ((rows[r] & coeff_mask) == 0 && ((rows[r] >> 78) & 1U)) return false;
  }

  unsigned __int128 is_pivot = 0;
  for (int r = 0; r < rank; r++) is_pivot |= static_cast<unsigned __int128>(1) << pivot_col[r];
  unsigned __int128 random_bits =
      static_cast<unsigned __int128>(rng()) |
      (static_cast<unsigned __int128>(rng() & ((1ULL << 14) - 1)) << 64);
  delta = random_bits & ~is_pivot & coeff_mask;
  for (int r = rank - 1; r >= 0; r--) {
    int c = pivot_col[r];
    unsigned __int128 without_pivot = rows[r] & coeff_mask &
                                      ~(static_cast<unsigned __int128>(1) << c);
    int bit = static_cast<int>((rows[r] >> 78) & 1U) ^ parity128(without_pivot & delta);
    if (bit)
      delta |= static_cast<unsigned __int128>(1) << c;
    else
      delta &= ~(static_cast<unsigned __int128>(1) << c);
  }
  return true;
}

static void apply_delta(uint64_t &lo64, uint16_t &hi14, unsigned __int128 delta) {
  lo64 ^= static_cast<uint64_t>(delta);
  hi14 ^= static_cast<uint16_t>((delta >> 64) & ((1u << 14) - 1));
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_forward_low64_newton F.txt target64 seconds [threads=4] [seed_x]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  bool have_seed = false;
  uint64_t seed_lo = 0;
  uint16_t seed_hi = 0;
  if (argc > 5) have_seed = parse_hex128_seed(argv[5], seed_lo, seed_hi);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::mutex mu;
  int global_best = 999;
  uint64_t best_lo = 0;
  uint16_t best_hi = 0;

  auto report = [&](int sc, uint64_t lo64, uint16_t hi14) {
    std::lock_guard<std::mutex> lock(mu);
    if (sc < global_best) {
      global_best = sc;
      best_lo = lo64;
      best_hi = hi14;
      std::cerr << "best " << sc << " x " << hex128(lo64, hi14) << "\n";
    }
  };

  auto try_solution = [&](uint64_t lo64, uint16_t hi14) {
    uint64_t out = low64_eval(lo64, hi14, gates);
    if (out == target) {
      std::lock_guard<std::mutex> lock(mu);
      if (!found) {
        found = true;
        std::cout << "x " << hex128(lo64, hi14) << "\n";
        std::cout << "low64_Fx 0x" << std::hex << std::setw(16)
                  << std::setfill('0') << out << std::dec << "\n";
        std::cout << "verified yes\n";
      }
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(0xbb67ae8584caa73bULL ^ (uint64_t)tid);
    uint64_t lo64 = (have_seed && tid == 0) ? seed_lo : rng();
    uint16_t hi14 = (have_seed && tid == 0) ? seed_hi : (rng() & ((1u << 14) - 1));
    while (!found && std::chrono::steady_clock::now() < deadline) {
      uint64_t base_out = low64_eval(lo64, hi14, gates);
      uint64_t residual = base_out ^ target;
      int base_score = __builtin_popcountll(residual);
      report(base_score, lo64, hi14);
      if (!residual) {
        try_solution(lo64, hi14);
        break;
      }

      uint64_t cols[78];
      for (int c = 0; c < 78; c++) {
        uint64_t clo = lo64;
        uint16_t chi = hi14;
        if (c < 64)
          clo ^= 1ULL << c;
        else
          chi ^= 1U << (c - 64);
        cols[c] = low64_eval(clo, chi, gates) ^ base_out;
      }

      uint64_t next_lo = lo64;
      uint16_t next_hi = hi14;
      int next_score = base_score;
      bool got_linear = false;
      for (int attempt = 0; attempt < 8192; attempt++) {
        unsigned __int128 delta = 0;
        if (!solve_linear_random(cols, residual, rng, delta)) break;
        got_linear = true;
        uint64_t cand_lo = lo64;
        uint16_t cand_hi = hi14;
        apply_delta(cand_lo, cand_hi, delta);
        int sc = score(cand_lo, cand_hi, target, gates);
        if (sc < next_score || (sc == next_score && (rng() & 31) == 0)) {
          next_score = sc;
          next_lo = cand_lo;
          next_hi = cand_hi;
        }
        if (!sc) {
          try_solution(cand_lo, cand_hi);
          return;
        }
      }

      if (got_linear && (next_score <= base_score || (rng() & 7) == 0)) {
        lo64 = next_lo;
        hi14 = next_hi;
      } else {
        lo64 = rng();
        hi14 = rng() & ((1u << 14) - 1);
      }
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_x " << hex128(best_lo, best_hi) << "\n";
    return 1;
  }
  return 0;
}
