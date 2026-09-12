#include <array>
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

static int hex_val(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
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

static int parse_int(const std::string &s, const char *name) {
  int out = 0;
  std::stringstream ss(s);
  ss >> out;
  if (!ss || out < 0) throw std::runtime_error(std::string("bad ") + name);
  return out;
}

static std::vector<int> parse_hex_bits(std::string s, int bits) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.empty()) throw std::runtime_error("empty target");
  std::vector<int> out(bits, 0);
  for (int bit = 0; bit < bits; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    out[bit] = (hv >> (bit % 4)) & 1;
  }
  for (int bit = bits; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad target hex digit");
    if (((hv >> (bit % 4)) & 1) != 0) {
      throw std::runtime_error("target has non-zero bits above target_bits");
    }
  }
  return out;
}

static void write_update_clauses(std::ofstream &out, int old_a, int b, int c,
                                 int new_a) {
  out << -old_a << ' ' << -b << ' ' << -new_a << " 0\n";
  out << -old_a << ' ' << c << ' ' << -new_a << " 0\n";
  out << old_a << ' ' << -b << ' ' << new_a << " 0\n";
  out << old_a << ' ' << c << ' ' << new_a << " 0\n";
  out << -old_a << ' ' << b << ' ' << -c << ' ' << new_a << " 0\n";
  out << old_a << ' ' << b << ' ' << -c << ' ' << -new_a << " 0\n";
}

int main(int argc, char **argv) {
  if (argc != 8 && argc != 9) {
    std::cerr << "usage: circuit_to_cnf_rev_lowtarget_orig_lz_gadget "
                 "F.txt out.cnf total_wires original_wires target_hex "
                 "leading_zero_bits target_bits [aux_zero]\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  int total_wires = parse_int(argv[3], "total wire count");
  int original_wires = parse_int(argv[4], "original wire count");
  int leading_zero_bits = parse_int(argv[6], "leading zero bit count");
  int target_bits = parse_int(argv[7], "target bit count");
  bool aux_zero = false;
  if (argc == 9) {
    std::string flag = argv[8];
    aux_zero = flag == "1" || flag == "true" || flag == "--aux-zero";
  }
  if (total_wires <= 0 || original_wires <= 0 || original_wires > total_wires ||
      leading_zero_bits > original_wires || target_bits < 0 ||
      target_bits > original_wires) {
    throw std::runtime_error("bad dimensions");
  }
  auto target = parse_hex_bits(argv[5], target_bits);
  for (auto [a, b, c] : gates) {
    if (a < 0 || a >= total_wires || b < 0 || b >= total_wires || c < 0 ||
        c >= total_wires) {
      throw std::runtime_error("wire out of range");
    }
  }

  std::vector<int> state(total_wires);
  for (int i = 0; i < total_wires; i++) state[i] = i + 1;
  int next_var = total_wires + 1;
  const long long aux_clauses = aux_zero ? (total_wires - original_wires) : 0;
  const long long clause_count =
      6LL * gates.size() + target_bits + leading_zero_bits + aux_clauses;

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << (total_wires + static_cast<int>(gates.size())) << ' '
      << clause_count << "\n";

  for (int i = 0; i < target_bits; i++) {
    out << (target[i] ? state[i] : -state[i]) << " 0\n";
  }

  for (auto it = gates.rbegin(); it != gates.rend(); ++it) {
    int a = (*it)[0], b = (*it)[1], c = (*it)[2];
    int old_a = state[a];
    int new_a = next_var++;
    write_update_clauses(out, old_a, state[b], state[c], new_a);
    state[a] = new_a;
  }

  for (int i = original_wires - leading_zero_bits; i < original_wires; i++) {
    out << -state[i] << " 0\n";
  }
  if (aux_zero) {
    for (int i = original_wires; i < total_wires; i++) out << -state[i] << " 0\n";
  }

  std::cerr << "gates " << gates.size() << "\n";
  std::cerr << "total_wires " << total_wires << "\n";
  std::cerr << "original_wires " << original_wires << "\n";
  std::cerr << "target_bits " << target_bits << "\n";
  std::cerr << "leading_zero_bits " << leading_zero_bits << "\n";
  std::cerr << "aux_zero " << (aux_zero ? "yes" : "no") << "\n";
  std::cerr << "vars " << (next_var - 1) << "\n";
  std::cerr << "clauses " << clause_count << "\n";
  return 0;
}
