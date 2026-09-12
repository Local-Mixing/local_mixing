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
  bool comment = false;
  bool line_start = true;
  char ch;
  while (in.get(ch)) {
    if (comment) {
      if (ch == '\n' || ch == '\r') {
        comment = false;
        line_start = true;
      }
      continue;
    }
    if (line_start && ch == '#') {
      comment = true;
    } else if (ch == ';') {
      if (!w.empty()) {
        if (w.size() != 3) throw std::runtime_error("bad gate");
        gates.push_back({w[0], w[1], w[2]});
        w.clear();
      }
      overflow = 0;
    } else if (ch == '~') {
      overflow++;
      line_start = false;
    } else if (ch == '\n' || ch == '\r' || ch == '\t' || ch == ' ') {
      if (ch == '\n' || ch == '\r') line_start = true;
      continue;
    } else {
      int b = val(ch);
      if (b < 0) throw std::runtime_error("bad char in circuit");
      w.push_back(b + 83 * overflow);
      overflow = 0;
      line_start = false;
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

int main(int argc, char **argv) {
  if (argc != 6) {
    std::cerr << "usage: circuit_to_cnf_inverse_middle_to_yz F.txt out.cnf target_middle y128 z128\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  auto target_middle = parse_hex128(argv[3]);
  auto y_value = parse_hex128(argv[4]);
  auto z_value = parse_hex128(argv[5]);

  std::vector<int> state(384);
  for (int i = 0; i < 384; i++) state[i] = i + 1;
  int next_var = 385;
  std::vector<std::vector<int>> clauses;
  clauses.reserve(6 * gates.size() + 384);

  // Input to the inverse is an arbitrary original output (L, target_middle, R).
  for (int i = 0; i < 128; i++) {
    clauses.push_back({target_middle[i] ? state[128 + i] : -state[128 + i]});
  }

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    auto [a, b, c] = *it;
    if (a < 0 || a >= 384 || b < 0 || b >= 384 || c < 0 || c >= 384) {
      throw std::runtime_error("wire out of range");
    }
    int old_a = state[a];
    int new_a = next_var++;
    add_update_clauses(clauses, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  // Output of the inverse is the original input; require its Y,Z blocks.
  for (int i = 0; i < 128; i++) {
    clauses.push_back({y_value[i] ? state[128 + i] : -state[128 + i]});
    clauses.push_back({z_value[i] ? state[256 + i] : -state[256 + i]});
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
