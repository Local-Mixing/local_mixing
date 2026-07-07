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

static std::array<uint8_t, 128> bits_from_lo78(uint64_t lo64, uint16_t hi14) {
  std::array<uint8_t, 128> bits{};
  for (int i = 0; i < 64; i++) bits[i] = (lo64 >> i) & 1ULL;
  for (int i = 0; i < 14; i++) bits[64 + i] = (hi14 >> i) & 1U;
  return bits;
}

static void eval(std::array<uint8_t, 128> &bits, const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
}

static uint64_t low64(const std::array<uint8_t, 128> &bits) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) if (bits[i]) out |= 1ULL << i;
  return out;
}

static std::string hex128_input(uint64_t lo64, uint16_t hi14) {
  std::array<uint8_t, 128> bits = bits_from_lo78(lo64, hi14);
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int chunk = 3; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++)
      if (bits[chunk * 32 + i]) v |= 1u << i;
    out << std::setw(8) << v;
  }
  return out.str();
}

static int score(uint64_t lo64, uint16_t hi14, uint64_t target,
                 const std::vector<Gate> &gates) {
  auto bits = bits_from_lo78(lo64, hi14);
  eval(bits, gates);
  return __builtin_popcountll(low64(bits) ^ target);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_forward_low64_anneal F.txt target64 seconds [threads=4]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::atomic<uint64_t> evals(0);
  std::mutex mu;
  int global_best = 999;
  uint64_t best_lo = 0;
  uint16_t best_hi = 0;

  auto worker = [&](int tid) {
    std::mt19937_64 rng(0x243f6a8885a308d3ULL ^ (uint64_t)tid);
    uint64_t lo = rng();
    uint16_t hi = rng() & ((1u << 14) - 1);
    int cur = score(lo, hi, target, gates);
    while (!found && std::chrono::steady_clock::now() < deadline) {
      int best = cur;
      uint64_t nlo = lo;
      uint16_t nhi = hi;
      int start = rng() % 78;
      for (int j = 0; j < 78; j++) {
        int bit = (start + j) % 78;
        uint64_t clo = lo;
        uint16_t chi = hi;
        if (bit < 64) clo ^= 1ULL << bit;
        else chi ^= 1U << (bit - 64);
        int sc = score(clo, chi, target, gates);
        evals++;
        if (sc < best || (sc == best && (rng() & 31) == 0)) {
          best = sc;
          nlo = clo;
          nhi = chi;
        }
      }
      if (best <= cur || (rng() % 1000) < 10) {
        lo = nlo;
        hi = nhi;
        cur = best;
      } else {
        lo ^= rng();
        hi ^= rng() & ((1u << 14) - 1);
        cur = score(lo, hi, target, gates);
      }
      if (cur < global_best) {
        std::lock_guard<std::mutex> lock(mu);
        if (cur < global_best) {
          global_best = cur;
          best_lo = lo;
          best_hi = hi;
          std::cerr << "best " << global_best << " x " << hex128_input(best_lo, best_hi)
                    << " evals " << evals.load() << "\n";
        }
      }
      if (!cur) {
        auto bits = bits_from_lo78(lo, hi);
        auto in = bits;
        eval(bits, gates);
        if (low64(bits) == target) {
          std::lock_guard<std::mutex> lock(mu);
          found = true;
          std::cout << "x " << hex128_input(lo, hi) << "\n";
          std::cout << "low64_Fx 0x" << std::hex << std::setw(16)
                    << std::setfill('0') << low64(bits) << std::dec << "\n";
          std::cout << "verified yes\n";
        }
      }
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_x " << hex128_input(best_lo, best_hi) << "\n";
    return 1;
  }
  return 0;
}
