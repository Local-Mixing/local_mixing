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

static void write_clause(std::ofstream &out, std::initializer_list<int> lits) {
  for (int lit : lits) out << lit << ' ';
  out << "0\n";
}

static void add_update_clauses(std::ofstream &out, int old_a, int b, int c, int new_a) {
  write_clause(out, {-old_a, -b, -new_a});
  write_clause(out, {-old_a, c, -new_a});
  write_clause(out, {old_a, -b, new_a});
  write_clause(out, {old_a, c, new_a});
  write_clause(out, {-old_a, b, -c, new_a});
  write_clause(out, {old_a, b, -c, -new_a});
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: circuit_to_cnf_rev_lowtarget_leading0_stream F.txt out.cnf target64\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  uint64_t target = parse_u64(argv[3]);

  const int n = 128;
  std::vector<int> state(n);
  for (int i = 0; i < n; i++) state[i] = i + 1;

  const long long vars = n + static_cast<long long>(gates.size());
  const long long clauses = 64 + 6LL * static_cast<long long>(gates.size()) + 50;

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << vars << ' ' << clauses << "\n";

  for (int i = 0; i < 64; i++) {
    write_clause(out, {((target >> i) & 1ULL) ? state[i] : -state[i]});
  }

  int next_var = n + 1;
  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    int a = (*it)[0], b = (*it)[1], c = (*it)[2];
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(out, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  for (int i = 78; i < 128; i++) write_clause(out, {-state[i]});

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "vars " << vars << "\n";
  std::cerr << "clauses " << clauses << "\n";
  return 0;
}
