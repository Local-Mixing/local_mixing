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

static void write_xor_lit(std::ofstream &out, int x, int y_lit, int z) {
  out << x << ' ' << y_lit << ' ' << -z << " 0\n";
  out << x << ' ' << -y_lit << ' ' << z << " 0\n";
  out << -x << ' ' << y_lit << ' ' << z << " 0\n";
  out << -x << ' ' << -y_lit << ' ' << -z << " 0\n";
}

static void write_not_equals(std::ofstream &out, int x, int z) {
  out << x << ' ' << z << " 0\n";
  out << -x << ' ' << -z << " 0\n";
}

static void write_xor_equals(std::ofstream &out, int a, int b, int value) {
  if (value) {
    out << a << ' ' << b << " 0\n";
    out << -a << ' ' << -b << " 0\n";
  } else {
    out << -a << ' ' << b << " 0\n";
    out << a << ' ' << -b << " 0\n";
  }
}

// CryptoMiniSat extended DIMACS: an `x` line asserts that the XOR of its
// (possibly signed) literals is true. Negating one literal flips the parity.
static void write_native_xor(std::ofstream &out,
                             const std::vector<int> &lits) {
  out << 'x';
  for (int lit : lits) out << lit << ' ';
  out << "0\n";
}

static void write_native_xor_equals(std::ofstream &out, int a, int b,
                                    int value) {
  write_native_xor(out, {value ? a : -a, b});
}

int main(int argc, char **argv) {
  if (argc < 9) {
    std::cerr
        << "usage: circuit_to_cnf_xor_io_preimage_mpmct F.mpmct out.cnf "
           "total_wires original_wires input_start xor_bits output_start "
           "target_hex [--aux-zero] [--aux-first-half-zero] "
           "[--aux-second-half-zero] [--native-xor] [--group-target-runs] "
           "[--native-xor-break-every=N] [--direct-output] "
           "[--native-xor-target-range=START:COUNT] "
           "[--fix-input-hex=HEX]\n";
    return 2;
  }

  auto circuit = parse_mpmct(argv[1]);
  int total_wires = parse_int_token(argv[3], "total wire count");
  int original_wires = parse_int_token(argv[4], "original wire count");
  int input_start = parse_int_token(argv[5], "input start");
  int xor_bits = parse_int_token(argv[6], "xor bit count");
  int output_start = parse_int_token(argv[7], "output start");
  bool aux_zero = false;
  bool aux_first_half_zero = false;
  bool aux_second_half_zero = false;
  bool native_xor = false;
  bool group_target_runs = false;
  int native_xor_break_every = 0;
  bool native_xor_target_range = false;
  int native_xor_target_start = 0;
  int native_xor_target_count = 0;
  bool direct_output = false;
  bool fix_input = false;
  std::vector<int> fixed_input;
  for (int i = 9; i < argc; i++) {
    std::string flag = argv[i];
    if (flag == "1" || flag == "true" || flag == "--aux-zero") {
      aux_zero = true;
    } else if (flag == "--aux-first-half-zero") {
      aux_first_half_zero = true;
    } else if (flag == "--aux-second-half-zero") {
      aux_second_half_zero = true;
    } else if (flag == "--native-xor") {
      native_xor = true;
    } else if (flag == "--group-target-runs") {
      group_target_runs = true;
    } else if (flag.rfind("--native-xor-break-every=", 0) == 0) {
      native_xor_break_every =
          parse_int_token(flag.substr(25), "native XOR break interval");
      if (native_xor_break_every == 0) {
        throw std::runtime_error("native XOR break interval must be positive");
      }
    } else if (flag.rfind("--native-xor-target-range=", 0) == 0) {
      if (native_xor_target_range) {
        throw std::runtime_error("duplicate --native-xor-target-range");
      }
      const std::string value = flag.substr(26);
      const size_t colon = value.find(':');
      if (colon == std::string::npos || value.find(':', colon + 1) != std::string::npos) {
        throw std::runtime_error(
            "--native-xor-target-range must be START:COUNT");
      }
      native_xor_target_start =
          parse_int_token(value.substr(0, colon), "native XOR target start");
      native_xor_target_count =
          parse_int_token(value.substr(colon + 1), "native XOR target count");
      if (native_xor_target_count == 0) {
        throw std::runtime_error("native XOR target count must be positive");
      }
      native_xor_target_range = true;
    } else if (flag == "--direct-output") {
      direct_output = true;
    } else if (flag.rfind("--fix-input-hex=", 0) == 0) {
      if (fix_input) throw std::runtime_error("duplicate --fix-input-hex");
      fixed_input = parse_hex_bits(flag.substr(16), xor_bits);
      fix_input = true;
    } else {
      throw std::runtime_error("unknown option: " + flag);
    }
  }
  if (group_target_runs && !native_xor) {
    throw std::runtime_error("--group-target-runs requires --native-xor");
  }
  if (native_xor_break_every && !native_xor) {
    throw std::runtime_error("--native-xor-break-every requires --native-xor");
  }
  if (native_xor_target_range && !native_xor) {
    throw std::runtime_error("--native-xor-target-range requires --native-xor");
  }
  if (native_xor_break_every && group_target_runs) {
    throw std::runtime_error(
        "--native-xor-break-every is not implemented with --group-target-runs");
  }
  if (direct_output && !fix_input) {
    throw std::runtime_error("--direct-output requires --fix-input-hex");
  }
  if (native_xor_target_range && group_target_runs) {
    throw std::runtime_error(
        "--native-xor-target-range is not implemented with --group-target-runs");
  }
  if (aux_zero && (aux_first_half_zero || aux_second_half_zero)) {
    throw std::runtime_error("--aux-zero cannot be combined with half-zero options");
  }

  if (total_wires <= 0 || original_wires <= 0 || original_wires > total_wires) {
    throw std::runtime_error("bad wire dimensions");
  }
  if (circuit.wires != total_wires) throw std::runtime_error("wire count mismatch");
  const int aux_wires = total_wires - original_wires;
  if ((aux_first_half_zero || aux_second_half_zero) && aux_wires % 2 != 0) {
    throw std::runtime_error("auxiliary wire count must be even for half-zero options");
  }
  if (input_start + xor_bits > total_wires || output_start + xor_bits > total_wires) {
    throw std::runtime_error("input or output block outside wire range");
  }
  if (native_xor_target_range &&
      native_xor_target_start + native_xor_target_count > total_wires) {
    throw std::runtime_error("native XOR target range outside wire range");
  }

  const auto native_transition = [&](int target_wire) {
    return native_xor &&
           (!native_xor_target_range ||
            (target_wire >= native_xor_target_start &&
             target_wire < native_xor_target_start + native_xor_target_count));
  };

  auto target = parse_hex_bits(argv[8], xor_bits);

  const int aux_half = aux_wires / 2;
  long long aux_clauses = aux_zero ? aux_wires : 0;
  if (aux_first_half_zero) aux_clauses += aux_half;
  if (aux_second_half_zero) aux_clauses += aux_half;
  long long gate_clauses = 0;
  long long gate_aux_vars = 0;
  long long state_vars = 0;
  if (group_target_runs) {
    for (size_t i = 0; i < circuit.gates.size();) {
      const int target_wire = circuit.gates[i].target;
      bool has_fire_lit = false;
      bool parity = false;
      size_t j = i;
      for (; j < circuit.gates.size() &&
             circuit.gates[j].target == target_wire; j++) {
        const auto &g = circuit.gates[j];
        if (g.controls.empty()) {
          if (!g.comp) parity = !parity;
        } else {
          gate_clauses += static_cast<long long>(g.controls.size()) + 1;
          gate_aux_vars++;
          has_fire_lit = true;
        }
      }
      if (has_fire_lit || parity) {
        gate_clauses++;
        state_vars++;
      }
      i = j;
    }
  } else {
    std::vector<int> target_updates(total_wires, 0);
    for (const auto &g : circuit.gates) {
      if (g.controls.empty()) {
        if (!g.comp) {
          target_updates[g.target]++;
          const bool xor_bridge =
              native_transition(g.target) && native_xor_break_every &&
              target_updates[g.target] % native_xor_break_every == 0;
          gate_clauses += native_transition(g.target) && !xor_bridge ? 1 : 2;
          state_vars++;
        }
        continue;
      }
      target_updates[g.target]++;
      const bool xor_bridge =
          native_transition(g.target) && native_xor_break_every &&
          target_updates[g.target] % native_xor_break_every == 0;
      gate_clauses += static_cast<long long>(g.controls.size()) + 1 +
                      (native_transition(g.target) && !xor_bridge ? 1 : 4);
      gate_aux_vars++;
      state_vars++;
    }
  }
  const long long output_clauses =
      direct_output ? xor_bits : (native_xor ? 1LL : 2LL) * xor_bits;
  long long clause_count = gate_clauses +
                           output_clauses + aux_clauses +
                           (fix_input ? xor_bits : 0);
  long long var_count = total_wires + gate_aux_vars + state_vars;

  std::ofstream out(argv[2]);
  if (!out) throw std::runtime_error("failed to create output cnf");
  out << "p cnf " << var_count << ' ' << clause_count << "\n";

  std::vector<int> state(total_wires);
  for (int i = 0; i < total_wires; i++) state[i] = i + 1;
  int next_var = total_wires + 1;

  if (aux_zero) {
    for (int i = original_wires; i < total_wires; i++) out << -state[i] << " 0\n";
  } else {
    if (aux_first_half_zero) {
      for (int i = original_wires; i < original_wires + aux_half; i++) {
        out << -state[i] << " 0\n";
      }
    }
    if (aux_second_half_zero) {
      for (int i = original_wires + aux_half; i < total_wires; i++) {
        out << -state[i] << " 0\n";
      }
    }
  }
  if (fix_input) {
    for (int i = 0; i < xor_bits; i++) {
      out << (fixed_input[i] ? state[input_start + i] : -state[input_start + i])
          << " 0\n";
    }
  }

  long long total_controls = 0;
  if (group_target_runs) {
    for (size_t i = 0; i < circuit.gates.size();) {
      const int target_wire = circuit.gates[i].target;
      const int old_t = state[target_wire];
      bool parity = false;
      std::vector<int> fire_lits;
      size_t j = i;
      for (; j < circuit.gates.size() &&
             circuit.gates[j].target == target_wire; j++) {
        const auto &g = circuit.gates[j];
        if (g.controls.empty()) {
          if (!g.comp) parity = !parity;
          continue;
        }
        int conj = next_var++;
        for (const auto &lit : g.controls) {
          int l = state[lit.wire];
          if (!lit.positive) l = -l;
          out << -conj << ' ' << l << " 0\n";
        }
        out << conj;
        for (const auto &lit : g.controls) {
          int l = state[lit.wire];
          if (!lit.positive) l = -l;
          out << ' ' << -l;
        }
        out << " 0\n";
        fire_lits.push_back(g.comp ? -conj : conj);
        total_controls += g.controls.size();
      }
      if (!fire_lits.empty() || parity) {
        int new_t = next_var++;
        std::vector<int> equation;
        equation.reserve(fire_lits.size() + 2);
        // old_t XOR fires... XOR new_t = parity. An x-line has RHS 1.
        equation.push_back(parity ? old_t : -old_t);
        equation.insert(equation.end(), fire_lits.begin(), fire_lits.end());
        equation.push_back(new_t);
        write_native_xor(out, equation);
        state[target_wire] = new_t;
      }
      i = j;
    }
  } else {
    std::vector<int> target_updates(total_wires, 0);
    for (const auto &g : circuit.gates) {
      int old_t = state[g.target];
      if (g.controls.empty()) {
        if (g.comp) continue; // fires = NOT true = false
        int new_t = next_var++;
        target_updates[g.target]++;
        const bool xor_bridge =
            native_transition(g.target) && native_xor_break_every &&
            target_updates[g.target] % native_xor_break_every == 0;
        if (native_transition(g.target) && !xor_bridge) {
          write_native_xor(out, {old_t, new_t});
        } else {
          write_not_equals(out, old_t, new_t);
        }
        state[g.target] = new_t;
        continue;
      }

      int conj = next_var++;
      target_updates[g.target]++;
      const bool xor_bridge =
          native_transition(g.target) && native_xor_break_every &&
          target_updates[g.target] % native_xor_break_every == 0;
      for (const auto &lit : g.controls) {
        int l = state[lit.wire];
        if (!lit.positive) l = -l;
        out << -conj << ' ' << l << " 0\n";
      }
      out << conj;
      for (const auto &lit : g.controls) {
        int l = state[lit.wire];
        if (!lit.positive) l = -l;
        out << ' ' << -l;
      }
      out << " 0\n";

      int new_t = next_var++;
      int fire_lit = g.comp ? -conj : conj;
      if (native_transition(g.target) && !xor_bridge) {
        write_native_xor(out, {-old_t, fire_lit, new_t});
      } else {
        write_xor_lit(out, old_t, fire_lit, new_t);
      }
      state[g.target] = new_t;
      total_controls += g.controls.size();
    }
  }

  for (int i = 0; i < xor_bits; i++) {
    int input_lit = input_start + i + 1;
    int output_lit = state[output_start + i];
    if (direct_output) {
      const int value = target[i] ^ fixed_input[i];
      out << (value ? output_lit : -output_lit) << " 0\n";
    } else if (native_xor) {
      write_native_xor_equals(out, output_lit, input_lit, target[i]);
    } else {
      write_xor_equals(out, output_lit, input_lit, target[i]);
    }
  }

  if (next_var - 1 != var_count) throw std::runtime_error("internal var count mismatch");

  std::cerr << "format mpmct1\n";
  std::cerr << "gates " << circuit.gates.size() << "\n";
  std::cerr << "total_controls " << total_controls << "\n";
  std::cerr << "total_wires " << total_wires << "\n";
  std::cerr << "original_wires " << original_wires << "\n";
  std::cerr << "input_start " << input_start << "\n";
  std::cerr << "xor_bits " << xor_bits << "\n";
  std::cerr << "output_start " << output_start << "\n";
  std::cerr << "target " << argv[8] << "\n";
  std::cerr << "aux_zero " << (aux_zero ? "yes" : "no") << "\n";
  std::cerr << "aux_first_half_zero " << (aux_first_half_zero ? "yes" : "no") << "\n";
  std::cerr << "aux_second_half_zero " << (aux_second_half_zero ? "yes" : "no") << "\n";
  std::cerr << "fix_input " << (fix_input ? "yes" : "no") << "\n";
  std::cerr << "native_xor " << (native_xor ? "yes" : "no") << "\n";
  std::cerr << "group_target_runs " << (group_target_runs ? "yes" : "no") << "\n";
  std::cerr << "native_xor_break_every " << native_xor_break_every << "\n";
  if (native_xor_target_range) {
    std::cerr << "native_xor_target_range " << native_xor_target_start << ':'
              << native_xor_target_count << "\n";
  } else {
    std::cerr << "native_xor_target_range all\n";
  }
  std::cerr << "direct_output " << (direct_output ? "yes" : "no") << "\n";
  std::cerr << "vars " << var_count << "\n";
  std::cerr << "clauses " << clause_count << "\n";
  return 0;
}
