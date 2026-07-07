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
using Mask = std::array<uint64_t, 2>;

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

static bool bit_mask(const Mask &m, int bit) {
  return (m[bit >> 6] >> (bit & 63)) & 1ULL;
}

static void set_mask(Mask &m, int bit) { m[bit >> 6] |= 1ULL << (bit & 63); }

static void xor_mask(Mask &a, const Mask &b) {
  a[0] ^= b[0];
  a[1] ^= b[1];
}

static uint8_t parity_and(const Mask &a, const Mask &b) {
  return (__builtin_popcountll(a[0] & b[0]) ^
          __builtin_popcountll(a[1] & b[1])) &
         1;
}

static bool zero_mask(const Mask &m) { return !(m[0] || m[1]); }

static int pop_mask(const Mask &m) {
  return __builtin_popcountll(m[0]) + __builtin_popcountll(m[1]);
}

static int pop100(const std::array<uint8_t, 256> &bits) {
  int out = 0;
  for (int i = 156; i < 256; i++) out += bits[i] != 0;
  return out;
}

static int pop_mask100(const std::array<uint8_t, 100> &rhs) {
  int out = 0;
  for (uint8_t bit : rhs) out += bit != 0;
  return out;
}

static std::array<uint8_t, 256> y_bits(const Mask &u,
                                       std::array<uint64_t, 2> low) {
  std::array<uint8_t, 256> bits{};
  for (int i = 0; i < 128; i++) bits[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
  for (int i = 0; i < 128; i++) bits[128 + i] = bit_mask(u, i);
  return bits;
}

static std::array<uint8_t, 256> reverse_eval(const Mask &u,
                                             std::array<uint64_t, 2> low,
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

static void eval_zero_jac(const Mask &u, std::array<uint64_t, 2> low,
                          const std::vector<Gate> &gates,
                          std::array<uint8_t, 100> &rhs,
                          std::array<Mask, 100> &rows,
                          std::array<uint8_t, 256> *x_out = nullptr) {
  std::array<uint8_t, 256> bits = y_bits(u, low);
  std::array<Mask, 256> deriv{};
  for (int i = 0; i < 128; i++) set_mask(deriv[128 + i], i);
  const Mask lane_mask{~0ULL, ~0ULL};

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    uint8_t g0 = bits[b] | !bits[c];
    Mask bmask = deriv[b];
    Mask cmask = deriv[c];
    if (bits[b]) {
      bmask[0] ^= lane_mask[0];
      bmask[1] ^= lane_mask[1];
    }
    if (bits[c]) {
      cmask[0] ^= lane_mask[0];
      cmask[1] ^= lane_mask[1];
    }
    Mask gmask{bmask[0] | (~cmask[0] & lane_mask[0]),
               bmask[1] | (~cmask[1] & lane_mask[1])};
    if (g0) {
      gmask[0] ^= lane_mask[0];
      gmask[1] ^= lane_mask[1];
    }
    xor_mask(deriv[a], gmask);
    bits[a] ^= g0;
  }

  if (x_out) *x_out = bits;
  for (int i = 0; i < 100; i++) {
    rhs[i] = bits[156 + i];
    rows[i] = deriv[156 + i];
  }
}

static bool solve_linear_random(std::array<Mask, 100> rows_in,
                                std::array<uint8_t, 100> rhs,
                                std::mt19937_64 &rng, Mask &delta) {
  std::array<int, 100> pivot_col{};
  pivot_col.fill(-1);
  int rank = 0;
  for (int c = 0; c < 128 && rank < 100; c++) {
    int piv = -1;
    for (int r = rank; r < 100; r++) {
      if (bit_mask(rows_in[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows_in[rank], rows_in[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < 100; r++) {
      if (r != rank && bit_mask(rows_in[r], c)) {
        xor_mask(rows_in[r], rows_in[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivot_col[rank++] = c;
  }
  for (int r = rank; r < 100; r++) {
    if (zero_mask(rows_in[r]) && rhs[r]) return false;
  }

  Mask is_pivot{0, 0};
  for (int r = 0; r < rank; r++) set_mask(is_pivot, pivot_col[r]);
  delta = {rng() & ~is_pivot[0], rng() & ~is_pivot[1]};
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

static bool solve_linear_low_weight(std::array<Mask, 100> rows_in,
                                    std::array<uint8_t, 100> rhs,
                                    std::mt19937_64 &rng, Mask &delta) {
  std::array<int, 100> pivot_col{};
  pivot_col.fill(-1);
  int rank = 0;
  for (int c = 0; c < 128 && rank < 100; c++) {
    int piv = -1;
    for (int r = rank; r < 100; r++) {
      if (bit_mask(rows_in[r], c)) {
        piv = r;
        break;
      }
    }
    if (piv < 0) continue;
    std::swap(rows_in[rank], rows_in[piv]);
    std::swap(rhs[rank], rhs[piv]);
    for (int r = 0; r < 100; r++) {
      if (r != rank && bit_mask(rows_in[r], c)) {
        xor_mask(rows_in[r], rows_in[rank]);
        rhs[r] ^= rhs[rank];
      }
    }
    pivot_col[rank++] = c;
  }
  for (int r = rank; r < 100; r++) {
    if (zero_mask(rows_in[r]) && rhs[r]) return false;
  }

  Mask is_pivot{0, 0};
  for (int r = 0; r < rank; r++) set_mask(is_pivot, pivot_col[r]);

  Mask particular{0, 0};
  for (int r = rank - 1; r >= 0; r--) {
    if (rhs[r]) set_mask(particular, pivot_col[r]);
  }

  std::vector<Mask> basis;
  basis.reserve(128 - rank);
  for (int c = 0; c < 128; c++) {
    if (bit_mask(is_pivot, c)) continue;
    Mask v{0, 0};
    set_mask(v, c);
    for (int r = rank - 1; r >= 0; r--) {
      if (bit_mask(rows_in[r], c)) set_mask(v, pivot_col[r]);
    }
    basis.push_back(v);
  }

  auto improve = [&](Mask cur) {
    bool changed = true;
    while (changed) {
      changed = false;
      int cur_w = pop_mask(cur);
      int best_i = -1;
      Mask best = cur;
      int best_w = cur_w;
      for (int i = 0; i < static_cast<int>(basis.size()); i++) {
        Mask cand{cur[0] ^ basis[i][0], cur[1] ^ basis[i][1]};
        int w = pop_mask(cand);
        if (w < best_w) {
          best_w = w;
          best_i = i;
          best = cand;
        }
      }
      if (best_i >= 0) {
        cur = best;
        changed = true;
      }
    }
    return cur;
  };

  delta = improve(particular);
  int best_w = pop_mask(delta);
  int restarts = basis.size() <= 30 ? 384 : 128;
  for (int restart = 0; restart < restarts; restart++) {
    Mask cur = particular;
    for (const auto &b : basis) {
      if (rng() & 1) {
        cur[0] ^= b[0];
        cur[1] ^= b[1];
      }
    }
    cur = improve(cur);
    int w = pop_mask(cur);
    if (w < best_w || (w == best_w && (rng() & 7) == 0)) {
      best_w = w;
      delta = cur;
    }
  }
  return true;
}

static std::string hex_bits(const std::array<uint8_t, 256> &bits, int lo,
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

static std::string hex_u(const Mask &u) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setfill('0') << std::setw(16) << u[1]
      << std::setw(16) << u[0];
  return out.str();
}

static bool parse_hex_u(std::string s, Mask &u) {
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

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "usage: search_reverse_lowtarget_leading0_newton_wide F.txt target128 seconds [threads=4] [seed] [seed_U]\n";
    return 2;
  }
  auto gates = parse(argv[1]);
  auto low = parse_low128(argv[2]);
  int seconds = std::stoi(argv[3]);
  int threads = argc > 4 ? std::stoi(argv[4]) : 4;
  uint64_t seed = argc > 5 ? std::stoull(argv[5], nullptr, 0) : 0x13198a2e03707344ULL;
  bool have_seed_u = false;
  Mask seed_u{0, 0};
  if (argc > 6) have_seed_u = parse_hex_u(argv[6], seed_u);

  if (seconds == 0 && have_seed_u) {
    std::mt19937_64 rng(seed);
    std::array<uint8_t, 100> rhs;
    std::array<Mask, 100> rows;
    std::array<uint8_t, 256> x;
    eval_zero_jac(seed_u, low, gates, rhs, rows, &x);
    int base_score = pop_mask100(rhs);
    std::cout << "center_U " << hex_u(seed_u) << "\n";
    std::cout << "center_score " << base_score << "\n";
    Mask low_weight_delta{0, 0};
    if (solve_linear_low_weight(rows, rhs, rng, low_weight_delta)) {
      Mask cand{seed_u[0] ^ low_weight_delta[0],
                seed_u[1] ^ low_weight_delta[1]};
      auto cand_x = reverse_eval(cand, low, gates);
      std::cout << "low_weight_delta " << hex_u(low_weight_delta) << "\n";
      std::cout << "low_weight_delta_weight " << pop_mask(low_weight_delta)
                << "\n";
      std::cout << "low_weight_candidate_U " << hex_u(cand) << "\n";
      std::cout << "low_weight_candidate_score " << pop100(cand_x) << "\n";
    }
    int best_random_score = 999;
    Mask best_random_delta{0, 0};
    for (int i = 0; i < 4096; i++) {
      Mask delta{0, 0};
      if (!solve_linear_random(rows, rhs, rng, delta)) break;
      Mask cand{seed_u[0] ^ delta[0], seed_u[1] ^ delta[1]};
      int sc = pop100(reverse_eval(cand, low, gates));
      if (sc < best_random_score) {
        best_random_score = sc;
        best_random_delta = delta;
      }
    }
    std::cout << "best_random_delta " << hex_u(best_random_delta) << "\n";
    std::cout << "best_random_delta_weight " << pop_mask(best_random_delta)
              << "\n";
    std::cout << "best_random_candidate_score " << best_random_score << "\n";

    Mask residual{0, 0};
    std::array<Mask, 128> cols{};
    for (int r = 0; r < 100; r++) {
      if (rhs[r]) set_mask(residual, r);
      for (int bit = 0; bit < 128; bit++) {
        if (bit_mask(rows[r], bit)) set_mask(cols[bit], r);
      }
    }
    struct PairMask {
      Mask mask;
      int a;
      int b;
    };
    std::vector<PairMask> pairs;
    pairs.reserve(8128);
    for (int a = 0; a < 128; a++) {
      for (int b = a + 1; b < 128; b++) {
        pairs.push_back({Mask{cols[a][0] ^ cols[b][0],
                              cols[a][1] ^ cols[b][1]},
                         a,
                         b});
      }
    }
    struct FourMove {
      int pred;
      int a;
      int b;
      int c;
      int d;
    };
    std::array<FourMove, 64> fours;
    for (auto &f : fours) f = {999, -1, -1, -1, -1};
    auto maybe_add_four = [&](int pred, int a, int b, int c, int d) {
      int worst = 0;
      for (int i = 1; i < static_cast<int>(fours.size()); i++) {
        if (fours[i].pred > fours[worst].pred) worst = i;
      }
      if (pred < fours[worst].pred) fours[worst] = {pred, a, b, c, d};
    };
    for (int i = 0; i < static_cast<int>(pairs.size()); i++) {
      const auto &p = pairs[i];
      for (int j = i + 1; j < static_cast<int>(pairs.size()); j++) {
        const auto &q = pairs[j];
        if (p.a == q.a || p.a == q.b || p.b == q.a || p.b == q.b) continue;
        Mask pred_mask{residual[0] ^ p.mask[0] ^ q.mask[0],
                       residual[1] ^ p.mask[1] ^ q.mask[1]};
        maybe_add_four(pop_mask(pred_mask), p.a, p.b, q.a, q.b);
      }
    }
    int best_four_score = base_score;
    Mask best_four_u = seed_u;
    int best_four_pred = 999;
    for (const auto &f : fours) {
      if (f.a < 0) continue;
      Mask cand = seed_u;
      cand[f.a >> 6] ^= 1ULL << (f.a & 63);
      cand[f.b >> 6] ^= 1ULL << (f.b & 63);
      cand[f.c >> 6] ^= 1ULL << (f.c & 63);
      cand[f.d >> 6] ^= 1ULL << (f.d & 63);
      int sc = pop100(reverse_eval(cand, low, gates));
      if (sc < best_four_score) {
        best_four_score = sc;
        best_four_u = cand;
        best_four_pred = f.pred;
      }
    }
    std::cout << "best_four_candidate_U " << hex_u(best_four_u) << "\n";
    std::cout << "best_four_candidate_score " << best_four_score << "\n";
    std::cout << "best_four_predicted_score " << best_four_pred << "\n";
    return 0;
  }

  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

  std::atomic<bool> found(false);
  std::mutex mu;
  int global_best = 999;
  Mask global_best_u{0, 0};

  auto report = [&](int sc, const Mask &u) {
    std::lock_guard<std::mutex> lock(mu);
    if (sc < global_best) {
      global_best = sc;
      global_best_u = u;
      std::cerr << "best " << sc << " U " << hex_u(u) << "\n";
    }
  };

  auto try_solution = [&](const Mask &u, const std::array<uint8_t, 256> &x) {
    if (pop100(x) != 0) return;
    auto y = forward_eval(x, gates);
    for (int i = 0; i < 128; i++) {
      if (y[i] != ((low[i >> 6] >> (i & 63)) & 1ULL)) return;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (!found.exchange(true)) {
      std::cout << "x " << hex_bits(x, 0, 256) << "\n";
      std::cout << "U " << hex_u(u) << "\n";
      std::cout << "output " << hex_bits(y, 0, 256) << "\n";
      std::cout << "low128_Fx 0x" << std::hex << std::setfill('0')
                << std::setw(16) << low[1] << std::setw(16) << low[0]
                << std::dec << "\n";
      std::cout << "verified yes\n";
    }
  };

  auto worker = [&](int tid) {
    std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1)));
    Mask u = (have_seed_u && tid == 0) ? seed_u : Mask{rng(), rng()};
    while (!found && std::chrono::steady_clock::now() < deadline) {
      std::array<uint8_t, 100> rhs;
      std::array<Mask, 100> rows;
      std::array<uint8_t, 256> x;
      eval_zero_jac(u, low, gates, rhs, rows, &x);
      int base_score = pop_mask100(rhs);
      report(base_score, u);
      if (!base_score) {
        try_solution(u, x);
        return;
      }

      int best_single_score = base_score;
      int best_single_bit = -1;
      std::array<int, 128> single_scores{};
      for (int bit = 0; bit < 128; bit++) {
        int sc = 0;
        for (int r = 0; r < 100; r++) sc += rhs[r] ^ bit_mask(rows[r], bit);
        single_scores[bit] = sc;
        if (sc < best_single_score) {
          best_single_score = sc;
          best_single_bit = bit;
        }
      }
      if (best_single_bit >= 0) {
        u[best_single_bit >> 6] ^= 1ULL << (best_single_bit & 63);
        continue;
      }

      if (base_score <= 32) {
        struct PairMove {
          int score;
          int a;
          int b;
        };
        std::array<PairMove, 16> pairs;
        for (auto &p : pairs) p = {999, -1, -1};
        auto maybe_add_pair = [&](int sc, int a, int b) {
          int worst = 0;
          for (int i = 1; i < static_cast<int>(pairs.size()); i++) {
            if (pairs[i].score > pairs[worst].score) worst = i;
          }
          if (sc < pairs[worst].score) pairs[worst] = {sc, a, b};
        };
        for (int a = 0; a < 128; a++) {
          for (int b = a + 1; b < 128; b++) {
            int sc = 0;
            for (int r = 0; r < 100; r++) {
              sc += rhs[r] ^ bit_mask(rows[r], a) ^ bit_mask(rows[r], b);
            }
            maybe_add_pair(sc, a, b);
          }
        }
        int actual_best = base_score;
        Mask actual_best_u = u;
        for (const auto &p : pairs) {
          if (p.a < 0) continue;
          Mask cand = u;
          cand[p.a >> 6] ^= 1ULL << (p.a & 63);
          cand[p.b >> 6] ^= 1ULL << (p.b & 63);
          auto cand_x = reverse_eval(cand, low, gates);
          int sc = pop100(cand_x);
          if (!sc) {
            try_solution(cand, cand_x);
            return;
          }
          if (sc < actual_best) {
            actual_best = sc;
            actual_best_u = cand;
          }
        }
        if (actual_best < base_score) {
          u = actual_best_u;
          continue;
        }
      }

      if (base_score <= 24) {
        std::array<int, 40> top_bits;
        top_bits.fill(-1);
        for (int bit = 0; bit < 128; bit++) {
          int slot = -1;
          for (int i = 0; i < static_cast<int>(top_bits.size()); i++) {
            if (top_bits[i] < 0 ||
                single_scores[bit] < single_scores[top_bits[i]]) {
              slot = i;
              break;
            }
          }
          if (slot >= 0) {
            for (int i = static_cast<int>(top_bits.size()) - 1; i > slot; i--) {
              top_bits[i] = top_bits[i - 1];
            }
            top_bits[slot] = bit;
          }
        }
        struct TripleMove {
          int score;
          int a;
          int b;
          int c;
        };
        std::array<TripleMove, 16> triples;
        for (auto &t : triples) t = {999, -1, -1, -1};
        auto maybe_add_triple = [&](int sc, int a, int b, int c) {
          int worst = 0;
          for (int i = 1; i < static_cast<int>(triples.size()); i++) {
            if (triples[i].score > triples[worst].score) worst = i;
          }
          if (sc < triples[worst].score) triples[worst] = {sc, a, b, c};
        };
        for (int ia = 0; ia < static_cast<int>(top_bits.size()); ia++) {
          int a = top_bits[ia];
          if (a < 0) continue;
          for (int ib = ia + 1; ib < static_cast<int>(top_bits.size()); ib++) {
            int b = top_bits[ib];
            if (b < 0) continue;
            for (int ic = ib + 1; ic < static_cast<int>(top_bits.size()); ic++) {
              int c = top_bits[ic];
              if (c < 0) continue;
              int sc = 0;
              for (int r = 0; r < 100; r++) {
                sc += rhs[r] ^ bit_mask(rows[r], a) ^ bit_mask(rows[r], b) ^
                      bit_mask(rows[r], c);
              }
              maybe_add_triple(sc, a, b, c);
            }
          }
        }
        int actual_best = base_score;
        Mask actual_best_u = u;
        for (const auto &t : triples) {
          if (t.a < 0) continue;
          Mask cand = u;
          cand[t.a >> 6] ^= 1ULL << (t.a & 63);
          cand[t.b >> 6] ^= 1ULL << (t.b & 63);
          cand[t.c >> 6] ^= 1ULL << (t.c & 63);
          auto cand_x = reverse_eval(cand, low, gates);
          int sc = pop100(cand_x);
          if (!sc) {
            try_solution(cand, cand_x);
            return;
          }
          if (sc < actual_best) {
            actual_best = sc;
            actual_best_u = cand;
          }
        }
        if (actual_best < base_score) {
          u = actual_best_u;
          continue;
        }
      }

      if (base_score <= 22 && (rng() & 1)) {
        std::array<int, 32> top_walk_bits;
        top_walk_bits.fill(-1);
        for (int bit = 0; bit < 128; bit++) {
          int slot = -1;
          for (int i = 0; i < static_cast<int>(top_walk_bits.size()); i++) {
            if (top_walk_bits[i] < 0 ||
                single_scores[bit] < single_scores[top_walk_bits[i]]) {
              slot = i;
              break;
            }
          }
          if (slot >= 0) {
            for (int i = static_cast<int>(top_walk_bits.size()) - 1; i > slot; i--) {
              top_walk_bits[i] = top_walk_bits[i - 1];
            }
            top_walk_bits[slot] = bit;
          }
        }
        int choices = 0;
        while (choices < static_cast<int>(top_walk_bits.size()) &&
               top_walk_bits[choices] >= 0) {
          choices++;
        }
        if (choices > 0) {
          int flips = 1 + (rng() % (base_score <= 18 ? 6 : 3));
          for (int i = 0; i < flips; i++) {
            int bit = top_walk_bits[rng() % choices];
            u[bit >> 6] ^= 1ULL << (bit & 63);
          }
          continue;
        }
      }

      Mask best_u = u;
      int best_score = base_score;
      bool got_linear = false;
      int attempts = base_score <= 16 ? 8192 : 1024;
      Mask low_weight_delta{0, 0};
      if (solve_linear_low_weight(rows, rhs, rng, low_weight_delta)) {
        got_linear = true;
        Mask cand{u[0] ^ low_weight_delta[0], u[1] ^ low_weight_delta[1]};
        auto cand_x = reverse_eval(cand, low, gates);
        int sc = pop100(cand_x);
        if (!sc) {
          try_solution(cand, cand_x);
          return;
        }
        if (sc < best_score || sc <= base_score + 1) {
          best_score = sc;
          best_u = cand;
        }
      }
      for (int attempt = 0; attempt < attempts; attempt++) {
        Mask delta{0, 0};
        if (!solve_linear_random(rows, rhs, rng, delta)) break;
        got_linear = true;
        Mask cand{u[0] ^ delta[0], u[1] ^ delta[1]};
        auto cand_x = reverse_eval(cand, low, gates);
        int sc = pop100(cand_x);
        if (!sc) {
          try_solution(cand, cand_x);
          return;
        }
        if (sc < best_score || (sc == best_score && (rng() & 31) == 0) ||
            (sc <= best_score + 2 && (rng() & 255) == 0)) {
          best_score = sc;
          best_u = cand;
        }
      }

      if (got_linear && (best_score <= base_score || (rng() & 15) == 0)) {
        u = best_u;
      } else {
        u = {rng(), rng()};
      }

      for (int j = 0; j < 32 && !found; j++) {
        int bit = rng() & 127;
        Mask cand = u;
        cand[bit >> 6] ^= 1ULL << (bit & 63);
        auto cand_x = reverse_eval(cand, low, gates);
        auto cur_x = reverse_eval(u, low, gates);
        int sc = pop100(cand_x);
        int cur = pop100(cur_x);
        if (!sc) {
          try_solution(cand, cand_x);
          return;
        }
        if (sc <= cur || (rng() & 127) == 0) u = cand;
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
