#include <array>
#include <cstdint>
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

static std::array<int8_t, 128> parse_hex128(std::string s) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.size() > 32) throw std::runtime_error("value too wide");
  while (s.size() < 32) s = "0" + s;

  std::array<int8_t, 128> bits{};
  for (int hex_pos = 0; hex_pos < 32; hex_pos++) {
    unsigned char nib = hex_nibble(s[31 - hex_pos]);
    for (int j = 0; j < 4; j++) bits[4 * hex_pos + j] = (nib >> j) & 1;
  }
  return bits;
}

static int8_t term_const(int8_t b, int8_t c) {
  if (b == 1 || c == 0) return 1;
  if (b == 0 && c == 1) return 0;
  return -1;
}

static int8_t xor_const(int8_t a, int8_t b) {
  if (a < 0 || b < 0) return -1;
  return a ^ b;
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
    std::cerr << "usage: circuit_to_cnf_fixed_yz_target_e_sliced F.txt out.cnf y128 z128 target128\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  auto y_value = parse_hex128(argv[3]);
  auto z_value = parse_hex128(argv[4]);
  auto target = parse_hex128(argv[5]);

  std::vector<std::array<int8_t, 3>> operand_consts(gates.size());
  std::array<int8_t, 384> cstate{};
  cstate.fill(-1);
  for (int i = 0; i < 128; i++) {
    cstate[128 + i] = y_value[i];
    cstate[256 + i] = z_value[i];
  }

  for (size_t i = 0; i < gates.size(); i++) {
    auto [a, b, c] = gates[i];
    if (a < 0 || a >= 384 || b < 0 || b >= 384 || c < 0 || c >= 384) {
      throw std::runtime_error("wire out of range");
    }
    operand_consts[i] = {cstate[a], cstate[b], cstate[c]};
    cstate[a] = xor_const(cstate[a], term_const(cstate[b], cstate[c]));
  }

  std::vector<uint8_t> include(gates.size(), 0);
  std::array<uint8_t, 384> needed{};
  needed.fill(0);
  for (int i = 128; i < 256; i++) needed[i] = 1;

  size_t no_op_live = 0;
  for (size_t ri = gates.size(); ri-- > 0;) {
    auto [a, b, c] = gates[ri];
    if (!needed[a]) continue;

    int8_t cb = operand_consts[ri][1];
    int8_t cc = operand_consts[ri][2];
    int8_t t = term_const(cb, cc);
    needed[a] = 1;

    if (t == 0) {
      no_op_live++;
      continue;
    }

    include[ri] = 1;
    if (!(cb == 1 || cc == 0)) {
      if (cb != 0) needed[b] = 1;
      if (cc != 1) needed[c] = 1;
    }
  }

  int next_var = 1;
  std::array<int, 384> state{};
  for (int i = 0; i < 128; i++) state[i] = next_var++;
  int const0 = next_var++;
  int const1 = next_var++;
  for (int i = 0; i < 128; i++) {
    state[128 + i] = y_value[i] ? const1 : const0;
    state[256 + i] = z_value[i] ? const1 : const0;
  }

  std::vector<std::vector<int>> clauses;
  clauses.reserve(6 * gates.size() / 2 + 512);
  clauses.push_back({-const0});
  clauses.push_back({const1});

  size_t kept = 0;
  for (size_t i = 0; i < gates.size(); i++) {
    auto [a, b, c] = gates[i];
    if (!include[i]) continue;
    kept++;
    int new_a = next_var++;
    add_update_clauses(clauses, state[a], state[b], state[c], new_a);
    state[a] = new_a;
  }

  for (int i = 0; i < 128; i++) {
    clauses.push_back({target[i] ? state[128 + i] : -state[128 + i]});
  }

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "c x variables: 1..128\n";
  out << "c const0 variable: " << const0 << "\n";
  out << "c const1 variable: " << const1 << "\n";
  out << "c kept gates: " << kept << "\n";
  out << "p cnf " << (next_var - 1) << ' ' << clauses.size() << "\n";
  for (const auto &clause : clauses) {
    for (int lit : clause) out << lit << ' ';
    out << "0\n";
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "kept_gates " << kept << "\n";
  std::cerr << "live_noop_gates " << no_op_live << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clauses.size() << "\n";
  return 0;
}
