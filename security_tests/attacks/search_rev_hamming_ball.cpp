#include <array>
#include <atomic>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
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
  if (!ss) throw std::runtime_error("bad u64");
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
  for (int i = 0; i < 50; i++)
    if (bits[78 + i]) out |= 1ULL << i;
  return out;
}

static uint64_t low64(const std::array<uint8_t, 128> &bits) {
  uint64_t out = 0;
  for (int i = 0; i < 64; i++)
    if (bits[i]) out |= 1ULL << i;
  return out;
}

static std::string hex128(const std::array<uint8_t, 128> &bits) {
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

static void visit_combinations(int start, int left, uint64_t flips,
                               const std::vector<Gate> &gates, uint64_t target,
                               uint64_t center, std::atomic<bool> &found,
                               std::atomic<uint64_t> &checked,
                               std::mutex &out_mu, int &best,
                               uint64_t &best_hi) {
  if (found) return;
  if (!left) {
    uint64_t hi = center ^ flips;
    uint64_t mask = top50_mask(hi, target, gates);
    int score = __builtin_popcountll(mask);
    checked++;
    if (score < best) {
      std::lock_guard<std::mutex> lock(out_mu);
      if (score < best) {
        best = score;
        best_hi = hi;
        std::cerr << "best " << best << " hi 0x" << std::hex
                  << std::setw(16) << std::setfill('0') << best_hi
                  << std::dec << " checked " << checked.load() << "\n";
      }
    }
    if (!mask) {
      auto x = reverse_eval(hi, target, gates);
      auto y = forward_eval(x, gates);
      if (low64(y) == target) {
        std::lock_guard<std::mutex> lock(out_mu);
        if (!found.exchange(true)) {
          std::cout << "x " << hex128(x) << "\n";
          std::cout << "z 0x" << std::hex << std::setw(16)
                    << std::setfill('0') << hi << std::dec << "\n";
          std::cout << "low64_Fx 0x" << std::hex << std::setw(16)
                    << std::setfill('0') << low64(y) << std::dec << "\n";
          std::cout << "verified yes\n";
        }
      }
    }
    return;
  }
  for (int bit = start; bit <= 64 - left; bit++) {
    visit_combinations(bit + 1, left - 1, flips ^ (1ULL << bit), gates,
                       target, center, found, checked, out_mu, best, best_hi);
    if (found) return;
  }
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: search_rev_hamming_ball F.txt target64 center_hi radius threads\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[2]);
  uint64_t center = parse_u64(argv[3]);
  int radius = std::stoi(argv[4]);
  int threads = std::stoi(argv[5]);
  if (radius < 0 || radius > 64 || threads < 1)
    throw std::runtime_error("bad radius/threads");

  std::atomic<bool> found(false);
  std::atomic<uint64_t> checked(0);
  std::mutex out_mu;
  int best = __builtin_popcountll(top50_mask(center, target, gates));
  uint64_t best_hi = center;
  std::cerr << "center_score " << best << "\n";

  for (int r = 0; r <= radius && !found; r++) {
    std::cerr << "radius " << r << "\n";
    if (r == 0) {
      visit_combinations(0, 0, 0, gates, target, center, found, checked,
                         out_mu, best, best_hi);
      continue;
    }
    std::vector<std::thread> ts;
    for (int first = 0; first <= 64 - r; first++) {
      if ((int)ts.size() == threads) {
        for (auto &t : ts) t.join();
        ts.clear();
      }
      ts.emplace_back([&, first, r] {
        visit_combinations(first + 1, r - 1, 1ULL << first, gates, target,
                           center, found, checked, out_mu, best, best_hi);
      });
    }
    for (auto &t : ts) t.join();
  }

  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << best << "\n";
    std::cout << "best_hi 0x" << std::hex << std::setw(16)
              << std::setfill('0') << best_hi << std::dec << "\n";
    std::cout << "checked " << checked.load() << "\n";
    return 1;
  }
  return 0;
}
