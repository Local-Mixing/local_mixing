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
  return gates;
}

static uint64_t parse_u64(const std::string &s) {
  uint64_t out = 0;
  std::stringstream ss(s);
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) ss >> std::hex >> out;
  else ss >> out;
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

static int score(const std::array<uint8_t, 128> &bits) {
  int bad = 0;
  for (int i = 78; i < 128; i++) bad += bits[i] != 0;
  return bad;
}

static uint64_t low64(const std::array<uint8_t, 128> &bits) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++) if (bits[i]) out |= 1ULL << i;
  return out;
}

static std::string hex128(const std::array<uint8_t, 128> &bits) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0');
  for (int chunk = 3; chunk >= 0; chunk--) {
    uint32_t v = 0;
    for (int i = 0; i < 32; i++) if (bits[chunk * 32 + i]) v |= 1u << i;
    out << std::setw(8) << v;
  }
  return out.str();
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_rev_preimage_anneal F.txt target64 seconds [threads=4]\n";
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
  uint64_t global_best_hi = 0;

  auto worker = [&](int tid) {
    std::mt19937_64 rng(0x9e3779b97f4a7c15ULL ^ (uint64_t)tid);
    uint64_t hi = rng();
    int cur = score(reverse_eval(hi, target, gates));
    while (!found && std::chrono::steady_clock::now() < deadline) {
      bool moved = false;
      int best = cur;
      uint64_t best_hi = hi;
      int start = rng() & 63;
      for (int j = 0; j < 64; j++) {
        int bit = (start + j) & 63;
        uint64_t cand = hi ^ (1ULL << bit);
        int sc = score(reverse_eval(cand, target, gates));
        evals++;
        if (sc < best || (sc == best && (rng() & 15) == 0)) {
          best = sc;
          best_hi = cand;
        }
      }
      if (best <= cur || (rng() % 1000) < 7) {
        hi = best_hi;
        cur = best;
        moved = true;
      }
      if (!moved || (rng() & 1023) == 0) {
        hi ^= rng();
        cur = score(reverse_eval(hi, target, gates));
        evals++;
      }
      if (cur < global_best) {
        std::lock_guard<std::mutex> lock(mu);
        if (cur < global_best) {
          global_best = cur;
          global_best_hi = hi;
          std::cerr << "best " << global_best << " hi 0x" << std::hex << std::setw(16)
                    << std::setfill('0') << global_best_hi << std::dec
                    << " evals " << evals.load() << "\n";
        }
      }
      if (cur == 0) {
        auto x = reverse_eval(hi, target, gates);
        auto y = forward_eval(x, gates);
        if (low64(y) == target && score(x) == 0) {
          std::lock_guard<std::mutex> lock(mu);
          found = true;
          std::cout << "x " << hex128(x) << "\n";
          std::cout << "hi 0x" << std::hex << std::setw(16) << std::setfill('0') << hi
                    << std::dec << "\n";
          std::cout << "evals " << evals.load() << "\n";
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
    std::cout << "best_hi 0x" << std::hex << std::setw(16) << std::setfill('0')
              << global_best_hi << std::dec << "\n";
    return 1;
  }
  return 0;
}
