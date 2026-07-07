#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
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
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) {
    ss >> std::hex >> out;
  } else {
    ss >> out;
  }
  if (!ss) throw std::runtime_error("bad target");
  return out;
}

static int parse_int(const std::string &s, const char *name) {
  int out = 0;
  std::stringstream ss(s);
  ss >> out;
  if (!ss || out < 0) throw std::runtime_error(std::string("bad ") + name);
  return out;
}

static void add_update_clauses(std::vector<std::vector<int>> &clauses, int old_a,
                               int b, int c, int new_a) {
  clauses.push_back({-old_a, -b, -new_a});
  clauses.push_back({-old_a, c, -new_a});
  clauses.push_back({old_a, -b, new_a});
  clauses.push_back({old_a, c, new_a});
  clauses.push_back({-old_a, b, -c, new_a});
  clauses.push_back({old_a, b, -c, -new_a});
}

int main(int argc, char **argv) {
  if (argc != 6 && argc != 7) {
    std::cerr << "usage: circuit_to_cnf_lowtarget_leading0_generic F.txt out.cnf n target64 leading_zero_bits [target_bits]\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  int n = parse_int(argv[3], "wire count");
  uint64_t target = parse_u64(argv[4]);
  int leading_zero_bits = parse_int(argv[5], "leading zero bit count");
  if (n <= 0) throw std::runtime_error("wire count must be positive");
  if (leading_zero_bits > n) throw std::runtime_error("too many leading zero bits");

  std::vector<int> state(n);
  for (int i = 0; i < n; i++) state[i] = i + 1;
  int next_var = n + 1;
  std::vector<std::vector<int>> clauses;
  int low_bits = n < 64 ? n : 64;
  if (argc == 7) low_bits = parse_int(argv[6], "target bit count");
  if (low_bits < 0 || low_bits > n || low_bits > 64) {
    throw std::runtime_error("bad target bit count");
  }
  clauses.reserve(6 * gates.size() + low_bits + leading_zero_bits);

  for (int i = 0; i < low_bits; i++) {
    clauses.push_back({((target >> i) & 1ULL) ? state[i] : -state[i]});
  }

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    int a = (*it)[0], b = (*it)[1], c = (*it)[2];
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(clauses, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  for (int i = n - leading_zero_bits; i < n; i++) clauses.push_back({-state[i]});

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << (next_var - 1) << ' ' << clauses.size() << "\n";
  for (const auto &clause : clauses) {
    for (int lit : clause) out << lit << ' ';
    out << "0\n";
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "wires " << n << "\n";
  std::cerr << "target_bits " << low_bits << "\n";
  std::cerr << "leading_zero_bits " << leading_zero_bits << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clauses.size() << "\n";
  return 0;
}
