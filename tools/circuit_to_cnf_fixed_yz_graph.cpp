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
      if (b < 0) throw std::runtime_error("bad char in circuit");
      w.push_back(b + 83 * overflow);
      overflow = 0;
    }
  }
  if (!w.empty()) throw std::runtime_error("unterminated gate");
  return gates;
}

static unsigned char hex_nibble(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  throw std::runtime_error("bad hex digit");
}

static std::array<int, 128> parse_hex128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) throw std::runtime_error("value too wide");
  while (s.size() < 32) s = "0" + s;

  std::array<int, 128> bits{};
  for (int hex_pos = 0; hex_pos < 32; hex_pos++) {
    unsigned char nib = hex_nibble(s[31 - hex_pos]);
    for (int j = 0; j < 4; j++) bits[4 * hex_pos + j] = (nib >> j) & 1;
  }
  return bits;
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

static void add_equiv(std::vector<std::vector<int>> &clauses, int a, int b) {
  clauses.push_back({-a, b});
  clauses.push_back({a, -b});
}

static void add_neq(std::vector<std::vector<int>> &clauses, int a, int b) {
  clauses.push_back({a, b});
  clauses.push_back({-a, -b});
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "usage: circuit_to_cnf_fixed_yz_graph circuit out.cnf y128 z128\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  auto y_value = parse_hex128(argv[3]);
  auto z_value = parse_hex128(argv[4]);

  std::vector<int> state(384);
  for (int i = 0; i < 384; i++) state[i] = i + 1;
  int next_var = 385;
  std::vector<std::vector<int>> clauses;
  clauses.reserve(6 * gates.size() + 512);

  for (int i = 0; i < 128; i++) {
    clauses.push_back({y_value[i] ? state[128 + i] : -state[128 + i]});
    clauses.push_back({z_value[i] ? state[256 + i] : -state[256 + i]});
  }

  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= 384 || b < 0 || b >= 384 || c < 0 || c >= 384) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(clauses, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  const int output_base = next_var;
  next_var += 128;
  for (int i = 0; i < 128; i++) {
    int o = output_base + i;
    if (y_value[i]) {
      add_neq(clauses, o, state[128 + i]);
    } else {
      add_equiv(clauses, o, state[128 + i]);
    }
  }

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "c x variables: 1..128\n";
  out << "c output variables: " << output_base << ".." << (output_base + 127)
      << "\n";
  out << "p cnf " << (next_var - 1) << ' ' << clauses.size() << "\n";
  for (const auto &clause : clauses) {
    for (int lit : clause) out << lit << ' ';
    out << "0\n";
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clauses.size() << "\n";
  std::cerr << "outputs " << output_base << ".." << (output_base + 127) << "\n";
  return 0;
}
