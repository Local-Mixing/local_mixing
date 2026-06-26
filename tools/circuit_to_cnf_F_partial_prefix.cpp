#include <array>
#include <fstream>
#include <iostream>
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
  if (argc != 3) {
    std::cerr << "usage: circuit_to_cnf_F_partial_prefix F.txt out.cnf\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  std::vector<int> state(128);
  for (int i = 0; i < 128; i++) state[i] = i + 1;
  int next_var = 129;
  std::vector<std::vector<int>> clauses;
  clauses.reserve(6 * gates.size() + 114);

  // Output has 64 leading zeros; low 64 output bits remain free.
  for (int i = 64; i < 128; i++) clauses.push_back({-state[i]});

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    int a = (*it)[0], b = (*it)[1], c = (*it)[2];
    if (a < 0 || a >= 128 || b < 0 || b >= 128 || c < 0 || c >= 128) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(clauses, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  // Input has 50 leading zero bits.
  for (int i = 78; i < 128; i++) clauses.push_back({-state[i]});

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << (next_var - 1) << ' ' << clauses.size() << "\n";
  for (const auto &clause : clauses) {
    for (int lit : clause) out << lit << ' ';
    out << "0\n";
  }
  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clauses.size() << "\n";
  return 0;
}
