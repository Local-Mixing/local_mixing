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

static std::vector<int> apply_forward(std::vector<int> bits,
                                      const std::vector<Gate> &gates) {
  for (auto [a, b, c] : gates) bits[a] ^= bits[b] | !bits[c];
  return bits;
}

static std::string hex_bits(const std::vector<int> &bits, int lo, int count) {
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

int main(int argc, char **argv) {
  if (argc != 8) {
    std::cerr << "usage: decode_forward_lowtarget_orig_lz_gadget_model "
                 "F.txt solver.out total_wires original_wires target_hex "
                 "leading_zero_bits target_bits\n";
    return 2;
  }

  auto gates = parse(argv[1]);
  int total_wires = parse_int(argv[3], "total wire count");
  int original_wires = parse_int(argv[4], "original wire count");
  int leading_zero_bits = parse_int(argv[6], "leading zero bit count");
  int low_bits = parse_int(argv[7], "target bit count");
  if (total_wires <= 0 || original_wires <= 0 || original_wires > total_wires) {
    throw std::runtime_error("bad wire dimensions");
  }
  if (leading_zero_bits > original_wires) {
    throw std::runtime_error("too many leading zero bits");
  }
  if (low_bits < 0 || low_bits > original_wires) {
    throw std::runtime_error("bad target bit count");
  }
  auto target = parse_hex_bits(argv[5], low_bits);

  std::ifstream in(argv[2]);
  if (!in) throw std::runtime_error("failed to open solver output");

  std::vector<int> input_bits(total_wires, 0);
  std::string line;
  bool sat = false;
  while (std::getline(in, line)) {
    if (line.rfind("s SATISFIABLE", 0) == 0) sat = true;
    if (line.empty() || line[0] != 'v') continue;
    std::istringstream ss(line.substr(1));
    long long lit = 0;
    while (ss >> lit) {
      if (lit == 0) break;
      long long v = lit < 0 ? -lit : lit;
      if (v >= 1 && v <= total_wires) {
        input_bits[static_cast<size_t>(v - 1)] = lit > 0;
      }
    }
  }
  if (!sat) {
    std::cout << "no SAT marker\n";
    return 1;
  }

  auto output_bits = apply_forward(input_bits, gates);

  int bad_leading = 0;
  for (int i = original_wires - leading_zero_bits; i < original_wires; i++) {
    bad_leading += input_bits[i] != 0;
  }

  int bad_target = 0;
  for (int i = 0; i < low_bits; i++) bad_target += output_bits[i] != target[i];

  std::cout << "x_original " << hex_bits(input_bits, 0, original_wires) << "\n";
  if (total_wires > original_wires) {
    std::cout << "aux_input "
              << hex_bits(input_bits, original_wires, total_wires - original_wires)
              << "\n";
  }
  std::cout << "input_full " << hex_bits(input_bits, 0, total_wires) << "\n";
  std::cout << "output_low_original "
            << hex_bits(output_bits, 0, original_wires) << "\n";
  if (original_wires > low_bits) {
    std::cout << "z_high_part "
              << hex_bits(output_bits, low_bits, original_wires - low_bits)
              << "\n";
  }
  std::cout << "low_output " << hex_bits(output_bits, 0, low_bits) << "\n";
  std::cout << "target " << hex_bits(target, 0, low_bits) << "\n";
  std::cout << "bad_leading_input_bits " << bad_leading << "\n";
  std::cout << "bad_target_bits " << bad_target << "\n";
  std::cout << "verified " << ((bad_leading == 0 && bad_target == 0) ? "yes" : "no")
            << "\n";
  return (bad_leading == 0 && bad_target == 0) ? 0 : 1;
}
