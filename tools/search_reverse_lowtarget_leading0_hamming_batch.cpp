#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
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
  for (int bit = 128; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if ((hv >> (bit & 3)) & 1) throw std::runtime_error("target too wide");
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

static std::array<uint8_t, 256> reverse_eval(const U128 &u, const U128 &low,
                                             const std::vector<Gate> &gates) {
  std::array<uint8_t, 256> bits{};
  for (int i = 0; i < 128; i++) bits[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
  for (int i = 0; i < 128; i++) bits[128 + i] = (u[i >> 6] >> (i & 63)) & 1ULL;
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

static U128 flip_bits(U128 u, const std::vector<int> &bits) {
  for (int bit : bits) u[bit >> 6] ^= 1ULL << (bit & 63);
  return u;
}

struct BatchResult {
  int score = 999;
  U128 u{0, 0};
};

static void eval_batch(const std::vector<U128> &us, int start, int count,
                       const U128 &low, const std::vector<Gate> &gates,
                       BatchResult &best) {
  std::array<uint64_t, 256> bits{};
  const uint64_t lane_mask = count == 64 ? ~0ULL : ((1ULL << count) - 1);
  for (int i = 0; i < 128; i++) {
    bits[i] = ((low[i >> 6] >> (i & 63)) & 1ULL) ? lane_mask : 0ULL;
  }
  for (int lane = 0; lane < count; lane++) {
    const U128 &u = us[start + lane];
    for (int i = 0; i < 128; i++) {
      if ((u[i >> 6] >> (i & 63)) & 1ULL) bits[128 + i] |= 1ULL << lane;
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

static void verify_and_print_solution(const U128 &u, const U128 &low,
                                      const std::vector<Gate> &gates,
                                      std::atomic<bool> &found,
                                      std::mutex &out_mu) {
  auto x = reverse_eval(u, low, gates);
  if (score_x(x) != 0) return;
  auto y = forward_eval(x, gates);
  for (int i = 0; i < 128; i++) {
    if (y[i] != ((low[i >> 6] >> (i & 63)) & 1ULL)) return;
  }
  std::lock_guard<std::mutex> lock(out_mu);
  if (!found.exchange(true)) {
    std::cout << "x " << hex_bits(x) << "\n";
    std::cout << "U " << hex_u(u) << "\n";
    std::cout << "output " << hex_bits(y) << "\n";
    std::cout << "verified yes\n";
  }
}

static uint64_t choose_u64(int n, int k) {
  if (k < 0 || k > n) return 0;
  if (k == 0 || k == n) return 1;
  if (k > n - k) k = n - k;
  __uint128_t out = 1;
  for (int i = 1; i <= k; i++) {
    out = (out * (n - k + i)) / i;
  }
  return static_cast<uint64_t>(out);
}

static void worker_fixed_first(int tid, int threads, int radius, U128 center,
                               const U128 &low, const std::vector<Gate> &gates,
                               std::atomic<bool> &found,
                               std::atomic<uint64_t> &done,
                               std::mutex &out_mu, BatchResult &global_best) {
  BatchResult local_best;
  std::vector<U128> batch;
  batch.reserve(64);
  auto flush = [&]() {
    for (int i = 0; i < static_cast<int>(batch.size()); i += 64) {
      BatchResult br;
      int count = std::min(64, static_cast<int>(batch.size()) - i);
      eval_batch(batch, i, count, low, gates, br);
      done += count;
      if (br.score < local_best.score) {
        local_best = br;
        std::lock_guard<std::mutex> lock(out_mu);
        if (br.score < global_best.score) {
          global_best = br;
          std::cerr << "best " << br.score << " U " << hex_u(br.u) << "\n";
        }
      }
      if (br.score == 0) {
        verify_and_print_solution(br.u, low, gates, found, out_mu);
      }
      if (found) return;
    }
    batch.clear();
  };

  if (radius == 0) {
    if (tid == 0) {
      batch.push_back(center);
      flush();
    }
    return;
  }

  for (int first = 0; first <= 128 - radius && !found; first++) {
    if (first % threads != tid) continue;
    std::vector<int> comb(radius);
    comb[0] = first;
    std::function<void(int, int)> rec = [&](int depth, int next) {
      if (found) return;
      if (depth == radius) {
        batch.push_back(flip_bits(center, comb));
        if (batch.size() >= 4096) flush();
        return;
      }
      for (int b = next; b <= 128 - (radius - depth); b++) {
        comb[depth] = b;
        rec(depth + 1, b + 1);
        if (found) return;
      }
    };
    rec(1, first + 1);
  }
  if (!batch.empty() && !found) flush();
}

int main(int argc, char **argv) {
  if (argc < 6) {
    std::cerr << "usage: search_reverse_lowtarget_leading0_hamming_batch F.txt target128 center_U max_radius threads\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  auto low = parse_low128(argv[2]);
  U128 center{0, 0};
  if (!parse_hex_u(argv[3], center)) throw std::runtime_error("bad center U");
  int max_radius = std::stoi(argv[4]);
  int threads = std::max(1, std::stoi(argv[5]));

  std::atomic<bool> found(false);
  std::atomic<uint64_t> done(0);
  std::mutex out_mu;
  BatchResult global_best;
  auto started = std::chrono::steady_clock::now();

  for (int r = 0; r <= max_radius && !found; r++) {
    done = 0;
    {
      std::lock_guard<std::mutex> lock(out_mu);
      std::cerr << "radius " << r << " candidates " << choose_u64(128, r)
                << "\n";
    }
    std::vector<std::thread> ts;
    for (int tid = 0; tid < threads; tid++) {
      ts.emplace_back(worker_fixed_first, tid, threads, r, center,
                      std::cref(low), std::cref(gates), std::ref(found),
                      std::ref(done), std::ref(out_mu),
                      std::ref(global_best));
    }
    while (!found) {
      bool all_done = true;
      for (auto &t : ts) {
        if (t.joinable()) {
          all_done = false;
          break;
        }
      }
      if (!all_done) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::lock_guard<std::mutex> lock(out_mu);
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                           std::chrono::steady_clock::now() - started)
                           .count();
        std::cerr << "progress radius " << r << " done " << done.load()
                  << " best " << global_best.score << " U "
                  << hex_u(global_best.u) << " elapsed_s " << elapsed
                  << "\n";
        break;
      }
    }
    for (auto &t : ts) t.join();
  }

  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_score " << global_best.score << "\n";
    std::cout << "best_U " << hex_u(global_best.u) << "\n";
    return 1;
  }
  return 0;
}
