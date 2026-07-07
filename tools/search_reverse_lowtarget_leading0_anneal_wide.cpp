#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
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

static std::array<uint8_t, 256> y_bits(const U128 &u, const U128 &low) {
  std::array<uint8_t, 256> bits{};
  for (int i = 0; i < 128; i++) bits[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
  for (int i = 0; i < 128; i++) bits[128 + i] = (u[i >> 6] >> (i & 63)) & 1ULL;
  return bits;
}

static std::array<uint8_t, 256> reverse_eval(const U128 &u, const U128 &low,
                                             const std::vector<Gate> &gates) {
  auto bits = y_bits(u, low);
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

static int score_u(const U128 &u, const U128 &low,
                   const std::vector<Gate> &gates) {
  return score_x(reverse_eval(u, low, gates));
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

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_reverse_lowtarget_leading0_anneal_wide F.txt target128 seconds [threads=4] [seed] [seed_U]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  auto low = parse_low128(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  uint64_t seed = argc > 5 ? std::stoull(argv[5], nullptr, 0) : 0x3243f6a8885a308dULL;
  bool have_seed_u = false;
  U128 seed_u{0, 0};
  if (argc > 6) have_seed_u = parse_hex_u(argv[6], seed_u);

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::atomic<uint64_t> evals(0);
  std::mutex mu;
  int global_best = 999;
  U128 global_best_u{0, 0};

  auto report = [&](int sc, const U128 &u) {
    std::lock_guard<std::mutex> lock(mu);
    if (sc < global_best) {
      global_best = sc;
      global_best_u = u;
      std::cerr << "best " << sc << " U " << hex_u(u)
                << " evals " << evals.load() << "\n";
    }
  };

  auto try_solution = [&](const U128 &u) {
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
      std::cout << "verified yes\n";
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1)));
    U128 u = (have_seed_u && tid == 0) ? seed_u : U128{rng(), rng()};
    int cur = score_u(u, low, gates);
    evals++;
    int steps_since_restart = 0;
    while (!found && std::chrono::steady_clock::now() < deadline) {
      report(cur, u);
      if (!cur) {
        try_solution(u);
        return;
      }
      double progress = 1.0 - std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count() / seconds;
      double temp = 7.5 * (1.0 - progress) + 0.75;
      U128 cand = u;
      int flips;
      uint64_t r = rng() % 100;
      if (r < 55)
        flips = 1;
      else if (r < 78)
        flips = 2;
      else if (r < 91)
        flips = 3 + (rng() & 1);
      else
        flips = 5 + (rng() % 8);
      for (int i = 0; i < flips; i++) {
        int bit = rng() & 127;
        cand[bit >> 6] ^= 1ULL << (bit & 63);
      }
      int sc = score_u(cand, low, gates);
      evals++;
      int delta = sc - cur;
      bool accept = delta <= 0;
      if (!accept) {
        double p = std::exp(-static_cast<double>(delta) / temp);
        double q = (rng() >> 11) * (1.0 / 9007199254740992.0);
        accept = q < p;
      }
      if (accept) {
        u = cand;
        cur = sc;
      }
      steps_since_restart++;
      if (steps_since_restart > 20000 || (cur > 45 && steps_since_restart > 2000)) {
        if (have_seed_u && (rng() & 3) == 0) {
          u = seed_u;
          for (int i = 0; i < 6; i++) {
            int bit = rng() & 127;
            u[bit >> 6] ^= 1ULL << (bit & 63);
          }
        } else {
          u = {rng(), rng()};
        }
        cur = score_u(u, low, gates);
        evals++;
        steps_since_restart = 0;
      }
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best << "\n";
    std::cout << "best_U " << hex_u(global_best_u) << "\n";
    return 1;
  }
  return 0;
}
