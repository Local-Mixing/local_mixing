#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
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

static unsigned char hex_nibble(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  throw std::runtime_error("bad hex digit");
}

static std::array<int, 128> parse_hex128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) throw std::runtime_error("target too wide");
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

static void add_xor_const(std::vector<std::vector<int>> &clauses, int x, int y,
                          bool value) {
  if (value) {
    clauses.push_back({x, y});
    clauses.push_back({-x, -y});
  } else {
    clauses.push_back({x, -y});
    clauses.push_back({-x, y});
  }
}

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: circuit_to_cnf_feistel_zero_z F.txt out.cnf target128\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  auto target = parse_hex128(argv[3]);

  std::vector<int> state(384);
  for (int i = 0; i < 384; i++) state[i] = i + 1;
  int next_var = 385;
  std::vector<std::vector<int>> clauses;
  clauses.reserve(6 * gates.size() + 384);

  // Input block Z/C, wires 256..383, is fixed to 0^128.
  for (int i = 256; i < 384; i++) clauses.push_back({-state[i]});

  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= 384 || b < 0 || b >= 384 || c < 0 || c >= 384) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(clauses, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  // Challenge condition: output_E xor input_B equals target.
  for (int i = 0; i < 128; i++) {
    add_xor_const(clauses, state[128 + i], i + 129, target[i]);
  }

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
