#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Lit {
  int wire = 0;
  bool positive = true;
};

struct Gate {
  int target = 0;
  bool comp = false;
  std::vector<Lit> controls;
};

struct Circuit {
  int wires = 0;
  std::vector<Gate> gates;
};

static int hex_val(char c) {
  if ('0' <= c && c <= '9') return c - '0';
  if ('a' <= c && c <= 'f') return 10 + c - 'a';
  if ('A' <= c && c <= 'F') return 10 + c - 'A';
  return -1;
}

static int parse_int_token(const std::string &s, const char *name) {
  int out = 0;
  std::stringstream ss(s);
  ss >> out;
  if (!ss || out < 0) throw std::runtime_error(std::string("bad ") + name);
  return out;
}

static int next_int(std::istringstream &ss, const std::string &line) {
  int out = 0;
  if (!(ss >> out)) throw std::runtime_error("bad mpmct gate line: " + line);
  return out;
}

static Circuit parse_mpmct(const std::string &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("failed to open " + path);

  std::string line;
  if (!std::getline(in, line)) throw std::runtime_error("empty mpmct file");
  std::istringstream hs(line);
  std::string magic;
  int wires = 0, expected_gates = 0;
  if (!(hs >> magic >> wires >> expected_gates) || magic != "mpmct1" ||
      wires <= 0 || expected_gates < 0) {
    throw std::runtime_error("bad mpmct1 header");
  }

  Circuit circuit;
  circuit.wires = wires;
  circuit.gates.reserve(expected_gates);

  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    std::istringstream ss(line);
    Gate g;
    int comp = 0, k = 0;
    g.target = next_int(ss, line);
    comp = next_int(ss, line);
    k = next_int(ss, line);
    if (g.target < 0 || g.target >= wires) throw std::runtime_error("target out of range");
    if (comp != 0 && comp != 1) throw std::runtime_error("bad comp bit");
    if (k < 0) throw std::runtime_error("bad control count");
    g.comp = comp != 0;
    g.controls.reserve(k);
    for (int i = 0; i < k; i++) {
      int w = next_int(ss, line);
      int p = next_int(ss, line);
      if (w < 0 || w >= wires) throw std::runtime_error("control out of range");
      if (w == g.target) throw std::runtime_error("control on gate target");
      if (p != 0 && p != 1) throw std::runtime_error("bad polarity bit");
      g.controls.push_back({w, p != 0});
    }
    int extra = 0;
    if (ss >> extra) throw std::runtime_error("extra token in mpmct gate line: " + line);
    circuit.gates.push_back(std::move(g));
  }

  if (static_cast<int>(circuit.gates.size()) != expected_gates) {
    throw std::runtime_error("mpmct gate count mismatch");
  }
  return circuit;
}

static std::vector<int> parse_hex_bits(std::string s, int bits) {
  if (s.rfind("0x", 0) == 0 || s.rfind("0X", 0) == 0) s = s.substr(2);
  if (s.empty()) throw std::runtime_error("empty hex value");
  std::vector<int> out(bits, 0);
  for (int bit = 0; bit < bits; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    if (hex_pos < 0) break;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad hex digit");
    out[bit] = (hv >> (bit % 4)) & 1;
  }
  for (int bit = bits; bit < static_cast<int>(s.size()) * 4; bit++) {
    int hex_pos = static_cast<int>(s.size()) - 1 - bit / 4;
    int hv = hex_val(s[hex_pos]);
    if (hv < 0) throw std::runtime_error("bad hex digit");
    if (((hv >> (bit % 4)) & 1) != 0) {
      throw std::runtime_error("hex value has non-zero bits above requested width");
    }
  }
  return out;
}

static std::vector<int> apply_forward(std::vector<int> bits,
                                      const std::vector<Gate> &gates) {
  for (const auto &g : gates) {
    int fire = 1;
    for (const auto &lit : g.controls) {
      int v = bits[lit.wire];
      fire &= lit.positive ? v : !v;
    }
    if (g.comp) fire = !fire;
    bits[g.target] ^= fire;
  }
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

static std::vector<int> xor_block(const std::vector<int> &a, int a_lo,
                                  const std::vector<int> &b, int b_lo,
                                  int count) {
  std::vector<int> out(count, 0);
  for (int i = 0; i < count; i++) out[i] = a[a_lo + i] ^ b[b_lo + i];
  return out;
}

int main(int argc, char **argv) {
  if (argc != 9) {
    std::cerr
        << "usage: decode_xor_io_preimage_mpmct_model F.mpmct solver.out "
           "total_wires original_wires input_start xor_bits output_start target_hex\n";
    return 2;
  }

  auto circuit = parse_mpmct(argv[1]);
  int total_wires = parse_int_token(argv[3], "total wire count");
  int original_wires = parse_int_token(argv[4], "original wire count");
  int input_start = parse_int_token(argv[5], "input start");
  int xor_bits = parse_int_token(argv[6], "xor bit count");
  int output_start = parse_int_token(argv[7], "output start");
  if (total_wires <= 0 || original_wires <= 0 || original_wires > total_wires) {
    throw std::runtime_error("bad wire dimensions");
  }
  if (circuit.wires != total_wires) throw std::runtime_error("wire count mismatch");
  if (input_start + xor_bits > total_wires || output_start + xor_bits > total_wires) {
    throw std::runtime_error("input or output block outside wire range");
  }
  auto target = parse_hex_bits(argv[8], xor_bits);

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
      if (v >= 1 && v <= total_wires) input_bits[static_cast<size_t>(v - 1)] = lit > 0;
    }
  }
  if (!sat) {
    std::cout << "sat no\n";
    return 1;
  }

  auto output_bits_vec = apply_forward(input_bits, circuit.gates);
  auto relation_bits = xor_block(output_bits_vec, output_start, input_bits,
                                 input_start, xor_bits);

  int bad_relation = 0;
  for (int i = 0; i < xor_bits; i++) bad_relation += relation_bits[i] != target[i];

  std::cout << "sat yes\n";
  if (original_wires >= 128) {
    std::cout << "x_block0 " << hex_bits(input_bits, 0, 128) << "\n";
  }
  std::cout << "input_original " << hex_bits(input_bits, 0, original_wires) << "\n";
  std::cout << "input_relation_block "
            << hex_bits(input_bits, input_start, xor_bits) << "\n";
  if (total_wires > original_wires) {
    std::cout << "aux_input "
              << hex_bits(input_bits, original_wires, total_wires - original_wires)
              << "\n";
  }
  std::cout << "output_original "
            << hex_bits(output_bits_vec, 0, original_wires) << "\n";
  std::cout << "output_relation_block "
            << hex_bits(output_bits_vec, output_start, xor_bits) << "\n";
  std::cout << "output_xor_input_block " << hex_bits(relation_bits, 0, xor_bits)
            << "\n";
  std::cout << "target " << hex_bits(target, 0, xor_bits) << "\n";
  std::cout << "bad_relation_bits " << bad_relation << "\n";
  std::cout << "verified " << (bad_relation == 0 ? "yes" : "no") << "\n";
  return bad_relation == 0 ? 0 : 1;
}
