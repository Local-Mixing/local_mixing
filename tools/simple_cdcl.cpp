#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <string>
#include <vector>

struct Clause {
  std::vector<int> lits;
  bool learnt = false;
};

struct Watcher {
  int cref;
};

static inline int var(int lit) { return std::abs(lit); }
static inline int neg(int lit) { return -lit; }
static inline int lit_index(int lit) {
  int v = var(lit) - 1;
  return 2 * v + (lit < 0);
}

struct Solver {
  int nvars = 0;
  std::vector<Clause> clauses;
  std::vector<std::vector<Watcher>> watches;
  std::vector<int8_t> assigns;  // -1 false, 0 undef, 1 true
  std::vector<int> levels;
  std::vector<int> reasons;
  std::vector<int> trail;
  std::vector<int> trail_lim;
  size_t qhead = 0;
  std::vector<double> activity;
  double var_inc = 1.0;
  std::priority_queue<std::pair<double, int>> order;
  uint64_t conflicts = 0;
  uint64_t decisions = 0;

  int decision_level() const { return static_cast<int>(trail_lim.size()); }

  int value_lit(int lit) const {
    int a = assigns[var(lit)];
    if (a == 0) return 0;
    return lit > 0 ? a : -a;
  }

  void bump_var(int v) {
    activity[v] += var_inc;
    order.push({activity[v], v});
    if (activity[v] > 1e100) {
      for (int i = 1; i <= nvars; i++) activity[i] *= 1e-100;
      var_inc *= 1e-100;
    }
  }

  bool enqueue(int lit, int reason) {
    int v = var(lit);
    int val = value_lit(lit);
    if (val == 1) return true;
    if (val == -1) return false;
    assigns[v] = lit > 0 ? 1 : -1;
    levels[v] = decision_level();
    reasons[v] = reason;
    trail.push_back(lit);
    return true;
  }

  int add_clause(std::vector<int> lits, bool learnt) {
    std::vector<int> clean;
    clean.reserve(lits.size());
    for (int lit : lits) {
      bool duplicate = false;
      for (int old : clean) {
        if (old == lit) {
          duplicate = true;
          break;
        }
        if (old == -lit) return -2;  // tautology
      }
      if (duplicate) continue;
      clean.push_back(lit);
    }
    if (clean.empty()) return -1;
    int cref = static_cast<int>(clauses.size());
    clauses.push_back({std::move(clean), learnt});
    Clause &c = clauses.back();
    if (c.lits.size() == 1) {
      if (!enqueue(c.lits[0], cref)) return -1;
    } else {
      watches[lit_index(c.lits[0])].push_back({cref});
      watches[lit_index(c.lits[1])].push_back({cref});
    }
    return cref;
  }

  int propagate() {
    while (qhead < trail.size()) {
      int p = trail[qhead++];
      int false_lit = neg(p);
      auto &ws = watches[lit_index(false_lit)];
      size_t i = 0, j = 0;
      while (i < ws.size()) {
        Watcher w = ws[i++];
        Clause &c = clauses[w.cref];
        if (c.lits.size() == 1) {
          ws[j++] = w;
          continue;
        }
        if (c.lits[0] != false_lit) std::swap(c.lits[0], c.lits[1]);
        if (c.lits[0] != false_lit) {
          ws[j++] = w;
          continue;
        }
        int other = c.lits[1];
        if (value_lit(other) == 1) {
          ws[j++] = w;
          continue;
        }
        bool found = false;
        for (size_t k = 2; k < c.lits.size(); k++) {
          if (value_lit(c.lits[k]) != -1) {
            std::swap(c.lits[0], c.lits[k]);
            watches[lit_index(c.lits[0])].push_back(w);
            found = true;
            break;
          }
        }
        if (found) continue;
        ws[j++] = w;
        if (value_lit(other) == -1) {
          while (i < ws.size()) ws[j++] = ws[i++];
          ws.resize(j);
          return w.cref;
        }
        if (!enqueue(other, w.cref)) {
          while (i < ws.size()) ws[j++] = ws[i++];
          ws.resize(j);
          return w.cref;
        }
      }
      ws.resize(j);
    }
    return -1;
  }

  void backtrack(int level) {
    if (decision_level() <= level) return;
    int keep = 0;
    while (keep < static_cast<int>(trail.size()) && levels[var(trail[keep])] <= level) {
      keep++;
    }
    for (int i = static_cast<int>(trail.size()) - 1; i >= keep; i--) {
      int v = var(trail[i]);
      assigns[v] = 0;
      levels[v] = 0;
      reasons[v] = -1;
      order.push({activity[v], v});
    }
    trail.resize(keep);
    trail_lim.resize(level);
    qhead = trail.size();
  }

  std::vector<int> analyze(int confl, int &backtrack_level) {
    std::vector<char> seen(nvars + 1, 0);
    std::vector<int> learnt;
    learnt.push_back(0);
    int path_c = 0;
    int p = 0;
    int idx = static_cast<int>(trail.size()) - 1;
    int cref = confl;
    int analysis_level = 0;
    for (int q : clauses[confl].lits) {
      analysis_level = std::max(analysis_level, levels[var(q)]);
    }

    do {
      Clause &c = clauses[cref];
      for (int q : c.lits) {
        if (q == p) continue;
        int v = var(q);
        if (!seen[v] && levels[v] > 0) {
          seen[v] = 1;
          bump_var(v);
          if (levels[v] == analysis_level) {
            path_c++;
          } else {
            learnt.push_back(q);
          }
        }
      }
      do {
        assert(idx >= 0);
        p = trail[idx--];
      } while (!seen[var(p)] || levels[var(p)] != analysis_level);
      seen[var(p)] = 0;
      path_c--;
      cref = reasons[var(p)];
    } while (path_c > 0);

    learnt[0] = neg(p);
    backtrack_level = 0;
    for (size_t i = 1; i < learnt.size(); i++) {
      backtrack_level = std::max(backtrack_level, levels[var(learnt[i])]);
    }
    var_inc *= 1.05;
    return learnt;
  }

  int pick_branch_lit() {
    while (!order.empty()) {
      int v = order.top().second;
      double a = order.top().first;
      order.pop();
      if (v <= 128 && assigns[v] == 0 &&
          std::abs(a - activity[v]) <= 1e-9 * (1.0 + std::abs(a))) {
        return v;
      }
    }
    for (int v = 1; v <= std::min(nvars, 128); v++) {
      if (assigns[v] == 0) return v;
    }
    return 0;
  }

  bool solve(uint64_t max_conflicts) {
    int confl = propagate();
    if (confl != -1) return false;
    while (true) {
      confl = propagate();
      if (confl != -1) {
        conflicts++;
        if (conflicts % 1000 == 0) {
          std::cerr << "c conflicts " << conflicts << " decisions " << decisions
                    << " level " << decision_level() << " learnts "
                    << std::count_if(clauses.begin(), clauses.end(),
                                     [](const Clause &c) { return c.learnt; })
                    << "\n";
        }
        if (max_conflicts && conflicts >= max_conflicts) return false;
        if (decision_level() == 0) return false;
        int bt = 0;
        auto learnt = analyze(confl, bt);
        backtrack(bt);
        int cr = add_clause(learnt, true);
        if (cr == -1) {
          std::cerr << "c learned empty clause at conflict " << conflicts << "\n";
          return false;
        }
        if (cr >= 0 && !enqueue(clauses[cr].lits[0], cr)) {
          std::cerr << "c failed to enqueue asserting lit " << clauses[cr].lits[0]
                    << " at conflict " << conflicts << " backtrack " << bt
                    << " learnt_size " << learnt.size()
                    << " lit_level " << levels[var(clauses[cr].lits[0])]
                    << " lit_value " << value_lit(clauses[cr].lits[0]) << "\n";
          return false;
        }
      } else {
        int next = pick_branch_lit();
        if (next == 0) return true;
        decisions++;
        trail_lim.push_back(static_cast<int>(trail.size()));
        enqueue(next, -1);
      }
    }
  }
};

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: simple_cdcl input.cnf [max_conflicts=0]\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "failed to open " << argv[1] << "\n";
    return 2;
  }
  uint64_t max_conflicts = argc == 3 ? std::stoull(argv[2]) : 0;
  Solver s;
  std::string tok;
  std::vector<int> lits;
  while (in >> tok) {
    if (tok == "c") {
      std::string line;
      std::getline(in, line);
    } else if (tok == "p") {
      std::string cnf;
      int nclauses = 0;
      in >> cnf >> s.nvars >> nclauses;
      s.watches.assign(2 * s.nvars, {});
      s.assigns.assign(s.nvars + 1, 0);
      s.levels.assign(s.nvars + 1, 0);
      s.reasons.assign(s.nvars + 1, -1);
      s.activity.assign(s.nvars + 1, 0.0);
      for (int v = 1; v <= s.nvars; v++) {
        s.activity[v] = v <= 128 ? 1.0 : 0.0;
        s.order.push({s.activity[v], v});
      }
      s.clauses.reserve(nclauses + 100000);
    } else {
      int lit = std::stoi(tok);
      if (lit == 0) {
        int cr = s.add_clause(lits, false);
        if (cr == -1) {
          std::cout << "s UNSATISFIABLE\n";
          return 20;
        }
        lits.clear();
      } else {
        lits.push_back(lit);
      }
    }
  }
  bool sat = s.solve(max_conflicts);
  if (!sat) {
    std::cout << "s UNKNOWN\n";
    std::cerr << "c conflicts " << s.conflicts << " decisions " << s.decisions << "\n";
    return 0;
  }
  std::cout << "s SATISFIABLE\n";
  std::cout << "v ";
  for (int v = 1; v <= s.nvars; v++) {
    int lit = s.assigns[v] >= 0 ? v : -v;
    std::cout << lit << ' ';
    if (v % 20 == 0) std::cout << "\nv ";
  }
  std::cout << "0\n";
  std::cerr << "c conflicts " << s.conflicts << " decisions " << s.decisions << "\n";
  return 10;
}
