#include <algorithm>
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
using U128 = std::array<uint64_t, 2>;

static int val(char c) {
  const std::string chars =
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
  auto p = chars.find(c);
  if (p == std::string::npos) return -1;
  return static_cast<int>(p);
}

static std::vector<Gate> parse_circuit(const std::string &path) {
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

struct Cnf {
  int vars = 0;
  std::vector<std::vector<int>> clauses;
  std::vector<std::vector<int>> occ;
  std::vector<int8_t> unit_value;
};

static Cnf parse_cnf(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path);
  Cnf cnf;
  std::string tok;
  while (in >> tok) {
    if (tok == "c") {
      std::string rest;
      std::getline(in, rest);
    } else if (tok == "p") {
      std::string kind;
      int clauses = 0;
      in >> kind >> cnf.vars >> clauses;
      cnf.clauses.reserve(clauses);
    } else {
      std::vector<int> clause;
      int lit = std::stoi(tok);
      while (lit != 0) {
        clause.push_back(lit);
        in >> lit;
      }
      cnf.clauses.push_back(std::move(clause));
    }
  }
  cnf.occ.assign(cnf.vars + 1, {});
  cnf.unit_value.assign(cnf.vars + 1, -1);
  for (int ci = 0; ci < static_cast<int>(cnf.clauses.size()); ci++) {
    if (cnf.clauses[ci].size() == 1) {
      int lit = cnf.clauses[ci][0];
      cnf.unit_value[std::abs(lit)] = lit > 0 ? 1 : 0;
    }
    for (int lit : cnf.clauses[ci]) cnf.occ[std::abs(lit)].push_back(ci);
  }
  return cnf;
}

static std::vector<uint8_t> initial_assignment_from_u(
    int vars, int n, const std::vector<Gate> &gates, const U128 &low,
    const U128 &u, const std::vector<int8_t> &unit_value, uint64_t seed,
    bool noisy) {
  std::vector<uint8_t> asn(vars + 1, 0);
  std::vector<int> state(n);
  std::vector<uint8_t> state_val(n, 0);
  std::mt19937_64 rng(seed);
  U128 uu = u;
  if (noisy) {
    int u_flips = 1 + (rng() % 24);
    for (int i = 0; i < u_flips; i++) {
      int bit = rng() & 127;
      uu[bit >> 6] ^= 1ULL << (bit & 63);
    }
  }
  for (int i = 0; i < n; i++) state[i] = i + 1;
  for (int i = 0; i < 128; i++) {
    state_val[i] = (low[i >> 6] >> (i & 63)) & 1ULL;
    asn[i + 1] = state_val[i];
  }
  for (int i = 0; i < 128; i++) {
    state_val[128 + i] = (uu[i >> 6] >> (i & 63)) & 1ULL;
    asn[129 + i] = state_val[128 + i];
  }
  int next_var = n + 1;
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    uint8_t new_val = state_val[a] ^ (state_val[b] | !state_val[c]);
    if (next_var > vars) throw std::runtime_error("CNF has too few vars");
    asn[next_var] = new_val;
    state[a] = next_var++;
    state_val[a] = new_val;
  }
  for (int v = 1; v <= vars; v++) {
    if (unit_value[v] >= 0) asn[v] = unit_value[v];
  }
  if (noisy) {
    int flips = 1 + (rng() % 64);
    for (int i = 0; i < flips; i++) {
      int v = 257 + (rng() % std::max(1, vars - 256));
      if (unit_value[v] >= 0) continue;
      asn[v] ^= 1;
    }
  }
  return asn;
}

struct Solver {
  const Cnf &cnf;
  std::vector<uint8_t> asn;
  std::vector<uint8_t> best_asn;
  const std::vector<int8_t> &unit_value;
  int best_seen = 1 << 20;
  std::vector<int> sat_count;
  std::vector<int> unsat;
  std::vector<int> unsat_pos;
  std::mt19937_64 rng;

  Solver(const Cnf &c, std::vector<uint8_t> a, uint64_t seed)
      : cnf(c), asn(std::move(a)), unit_value(c.unit_value),
        sat_count(c.clauses.size(), 0),
        unsat_pos(c.clauses.size(), -1), rng(seed) {
    for (int ci = 0; ci < static_cast<int>(cnf.clauses.size()); ci++) {
      int cnt = 0;
      for (int lit : cnf.clauses[ci]) cnt += lit_sat(lit);
      sat_count[ci] = cnt;
      if (cnt == 0) add_unsat(ci);
    }
    best_seen = static_cast<int>(unsat.size());
    best_asn = asn;
  }

  bool lit_sat(int lit) const {
    int v = std::abs(lit);
    return lit > 0 ? asn[v] : !asn[v];
  }

  void add_unsat(int ci) {
    if (unsat_pos[ci] >= 0) return;
    unsat_pos[ci] = static_cast<int>(unsat.size());
    unsat.push_back(ci);
  }

  void remove_unsat(int ci) {
    int pos = unsat_pos[ci];
    if (pos < 0) return;
    int last = unsat.back();
    unsat[pos] = last;
    unsat_pos[last] = pos;
    unsat.pop_back();
    unsat_pos[ci] = -1;
  }

  int flip_delta(int v) const {
    if (unit_value[v] >= 0) return 1 << 20;
    int delta = 0;
    for (int ci : cnf.occ[v]) {
      int before = sat_count[ci];
      bool was = false;
      for (int lit : cnf.clauses[ci]) {
        if (std::abs(lit) == v) {
          was = lit_sat(lit);
          break;
        }
      }
      int after = before + (was ? -1 : 1);
      if (before == 0 && after > 0) delta--;
      if (before > 0 && after == 0) delta++;
    }
    return delta;
  }

  void flip(int v) {
    if (unit_value[v] >= 0) return;
    for (int ci : cnf.occ[v]) {
      int before = sat_count[ci];
      bool was = false;
      for (int lit : cnf.clauses[ci]) {
        if (std::abs(lit) == v) {
          was = lit_sat(lit);
          break;
        }
      }
      int after = before + (was ? -1 : 1);
      sat_count[ci] = after;
      if (before == 0 && after > 0) remove_unsat(ci);
      if (before > 0 && after == 0) add_unsat(ci);
    }
    asn[v] ^= 1;
  }

  int pick_var(int ci, int noise_percent) {
    const auto &cl = cnf.clauses[ci];
    if (cl.empty()) return -1;
    if ((int)(rng() % 100) < noise_percent) {
      std::vector<int> allowed;
      for (int lit : cl) {
        int v = std::abs(lit);
        if (unit_value[v] < 0) allowed.push_back(v);
      }
      if (allowed.empty()) return -1;
      return allowed[rng() % allowed.size()];
    }
    int best_delta = 1 << 20;
    std::vector<int> best;
    for (int lit : cl) {
      int v = std::abs(lit);
      if (unit_value[v] >= 0) continue;
      int d = flip_delta(v);
      if (d < best_delta) {
        best_delta = d;
        best.clear();
        best.push_back(v);
      } else if (d == best_delta) {
        best.push_back(v);
      }
    }
    if (best.empty()) return -1;
    return best[rng() % best.size()];
  }

  bool run(uint64_t max_flips, int noise_percent, int &best_unsat) {
    best_unsat = std::min(best_unsat, static_cast<int>(unsat.size()));
    for (uint64_t step = 0; step < max_flips; step++) {
      if (unsat.empty()) return true;
      if (static_cast<int>(unsat.size()) < best_unsat) {
        best_unsat = static_cast<int>(unsat.size());
      }
      if (static_cast<int>(unsat.size()) < best_seen) {
        best_seen = static_cast<int>(unsat.size());
        best_asn = asn;
      }
      int ci = unsat[rng() % unsat.size()];
      int v = pick_var(ci, noise_percent);
      if (v <= 0) return false;
      flip(v);
    }
    return unsat.empty();
  }
};

static U128 u_from_assignment(const std::vector<uint8_t> &asn) {
  U128 u{0, 0};
  for (int i = 0; i < 128; i++) {
    if (asn[129 + i]) u[i >> 6] |= 1ULL << (i & 63);
  }
  return u;
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

static int score_x(const std::array<uint8_t, 256> &bits) {
  int out = 0;
  for (int i = 156; i < 256; i++) out += bits[i] != 0;
  return out;
}

static void print_model(const std::vector<uint8_t> &asn) {
  std::cout << "s SATISFIABLE\n";
  for (int v = 1; v < static_cast<int>(asn.size()); v++) {
    if ((v - 1) % 16 == 0) std::cout << "v";
    std::cout << ' ' << (asn[v] ? v : -v);
    if ((v - 1) % 16 == 15 || v + 1 == static_cast<int>(asn.size()))
      std::cout << " 0\n";
  }
}

int main(int argc, char **argv) {
  if (argc < 8) {
    std::cerr << "usage: sls_circuit_cnf_from_u F.txt cnf target128 center_U seconds threads seed\n";
    return 2;
  }
  auto gates = parse_circuit(argv[1]);
  auto cnf = parse_cnf(argv[2]);
  auto low = parse_low128(argv[3]);
  U128 center{0, 0};
  if (!parse_hex_u(argv[4], center)) throw std::runtime_error("bad center U");
  int seconds = std::stoi(argv[5]);
  int threads = std::stoi(argv[6]);
  uint64_t seed = std::stoull(argv[7], nullptr, 0);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
  std::atomic<bool> found(false);
  std::atomic<int> global_best(1 << 20);
  std::mutex out_mu;

  auto worker = [&](int tid) {
    uint64_t s = seed ^ (0x9e3779b97f4a7c15ULL * (uint64_t)(tid + 1));
    int restart = 0;
    while (!found && std::chrono::steady_clock::now() < deadline) {
      bool noisy = restart != 0;
      auto asn = initial_assignment_from_u(cnf.vars, 256, gates, low, center,
                                           cnf.unit_value,
                                           s + restart * 0x100000001b3ULL,
                                           noisy);
      Solver solver(cnf, std::move(asn), s ^ (uint64_t)restart);
      int best_unsat = solver.unsat.size();
      global_best.store(std::min(global_best.load(), best_unsat));
      bool ok = solver.run(2000000, 23 + (restart % 17), best_unsat);
      if (!solver.best_asn.empty()) {
        U128 implied = u_from_assignment(solver.best_asn);
        int real_score = score_x(reverse_eval(implied, low, gates));
        if (real_score <= 20 || solver.best_seen <= 12) {
          std::lock_guard<std::mutex> lock(out_mu);
          std::cerr << "best_cnf " << solver.best_seen << " real_score "
                    << real_score << " implied_U " << hex_u(implied)
                    << " tid " << tid << " restart " << restart << "\n";
        }
      }
      int old = global_best.load();
      while (best_unsat < old && !global_best.compare_exchange_weak(old, best_unsat)) {}
      if (best_unsat <= old) {
        std::lock_guard<std::mutex> lock(out_mu);
        std::cerr << "best_unsat " << best_unsat << " tid " << tid
                  << " restart " << restart << " U " << hex_u(center)
                  << "\n";
      }
      if (ok) {
        std::lock_guard<std::mutex> lock(out_mu);
        if (!found.exchange(true)) print_model(solver.asn);
        return;
      }
      restart++;
    }
  };

  std::vector<std::thread> ts;
  for (int i = 0; i < threads; i++) ts.emplace_back(worker, i);
  for (auto &t : ts) t.join();
  if (!found) {
    std::cout << "not_found\n";
    std::cout << "best_unsat " << global_best.load() << "\n";
    return 1;
  }
  return 0;
}
