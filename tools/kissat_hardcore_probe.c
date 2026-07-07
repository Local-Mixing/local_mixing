#define _POSIX_C_SOURCE 200809L

#include "internal.h"
#include "inline.h"
#include "kissat.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

static kissat *volatile alarm_solver;
static volatile sig_atomic_t alarm_seen;

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void alarm_handler(int sig) {
  (void)sig;
  alarm_seen = 1;
  if (alarm_solver) kissat_terminate(alarm_solver);
}

static void arm_timer(double seconds) {
  struct itimerval timer;
  memset(&timer, 0, sizeof timer);
  if (seconds > 0) {
    timer.it_value.tv_sec = (time_t)seconds;
    timer.it_value.tv_usec =
        (suseconds_t)((seconds - (double)timer.it_value.tv_sec) * 1e6);
    if (!timer.it_value.tv_sec && !timer.it_value.tv_usec)
      timer.it_value.tv_usec = 1;
  }
  if (setitimer(ITIMER_REAL, &timer, 0)) {
    fprintf(stderr, "failed to arm timer: %s\n", strerror(errno));
    exit(1);
  }
}

static void disarm_timer(void) {
  struct itimerval timer;
  memset(&timer, 0, sizeof timer);
  if (setitimer(ITIMER_REAL, &timer, 0)) {
    fprintf(stderr, "failed to disarm timer: %s\n", strerror(errno));
    exit(1);
  }
}

static void *xcalloc(size_t n, size_t size) {
  void *ptr = calloc(n, size);
  if (!ptr && n && size) {
    fprintf(stderr, "out of memory allocating %zu x %zu bytes\n", n, size);
    exit(1);
  }
  return ptr;
}

static void *xrealloc(void *ptr, size_t size) {
  void *res = realloc(ptr, size);
  if (!res && size) {
    fprintf(stderr, "out of memory reallocating %zu bytes\n", size);
    exit(1);
  }
  return res;
}

static char *xstrdup(const char *s) {
  const size_t n = strlen(s) + 1;
  char *res = xcalloc(n, 1);
  memcpy(res, s, n);
  return res;
}

typedef struct {
  FILE *file;
  unsigned char buf[1 << 20];
  size_t pos;
  size_t end;
} scanner;

static int next_char(scanner *s) {
  if (s->pos == s->end) {
    s->end = fread(s->buf, 1, sizeof s->buf, s->file);
    s->pos = 0;
    if (!s->end) return EOF;
  }
  return s->buf[s->pos++];
}

static void unread_char(scanner *s) {
  if (s->pos) s->pos--;
}

static int read_int(scanner *s, int *out) {
  int ch;
  do {
    ch = next_char(s);
    if (ch == EOF) return 0;
    if (ch == 'c') {
      while ((ch = next_char(s)) != EOF && ch != '\n') {
      }
    }
  } while (isspace(ch));

  int sign = 1;
  if (ch == '-') {
    sign = -1;
    ch = next_char(s);
  }
  if (!isdigit(ch)) return -1;

  int value = 0;
  do {
    value = 10 * value + (ch - '0');
    ch = next_char(s);
  } while (isdigit(ch));
  if (ch != EOF) unread_char(s);
  *out = sign * value;
  return 1;
}

static void skip_header(FILE *file, int *max_var, int *clauses) {
  char line[4096];
  *max_var = 0;
  *clauses = 0;
  while (fgets(line, sizeof line, file)) {
    if (line[0] == 'c') continue;
    if (line[0] == 'p') {
      if (sscanf(line, "p cnf %d %d", max_var, clauses) != 2) {
        fprintf(stderr, "bad DIMACS header: %s\n", line);
        exit(2);
      }
      return;
    }
    if (!isspace((unsigned char)line[0])) {
      fprintf(stderr, "unexpected line before header: %s\n", line);
      exit(2);
    }
  }
  fprintf(stderr, "missing DIMACS header\n");
  exit(2);
}

static void parse_dimacs(kissat *solver, const char *path, int *max_var,
                         int *clauses) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "failed to open %s: %s\n", path, strerror(errno));
    exit(1);
  }

  skip_header(file, max_var, clauses);
  kissat_reserve(solver, *max_var);

  scanner s;
  memset(&s, 0, sizeof s);
  s.file = file;

  int lit;
  int seen_clauses = 0;
  while (1) {
    int res = read_int(&s, &lit);
    if (!res) break;
    if (res < 0) {
      fprintf(stderr, "bad integer in DIMACS body\n");
      exit(2);
    }
    kissat_add(solver, lit);
    if (!lit) seen_clauses++;
  }
  fclose(file);

  if (seen_clauses != *clauses) {
    fprintf(stderr, "warning: header clauses %d but saw %d\n", *clauses,
            seen_clauses);
  }
}

enum category {
  CAT_ABSENT = 0,
  CAT_ACTIVE = 1,
  CAT_FIXED = 2,
  CAT_ELIMINATED = 3,
  CAT_INACTIVE = 4,
  CAT_COUNT = 5,
};

static enum category classify_external(kissat *solver, unsigned eidx) {
  if (eidx >= SIZE_STACK(solver->import)) return CAT_ABSENT;
  const import *imp = &PEEK_STACK(solver->import, eidx);
  if (!imp->imported) return CAT_ABSENT;
  if (imp->eliminated) return CAT_ELIMINATED;

  const unsigned ilit = imp->lit;
  const unsigned iidx = IDX(ilit);
  if (iidx >= solver->vars) return CAT_INACTIVE;

  const flags *f = solver->flags + iidx;
  if (f->active) return CAT_ACTIVE;
  if (f->fixed) return CAT_FIXED;
  return CAT_INACTIVE;
}

typedef struct {
  char **by_var;
  char **labels;
  size_t labels_size;
  size_t labels_cap;
} semantic_map;

static char *intern_label(semantic_map *map, const char *label) {
  for (size_t i = 0; i < map->labels_size; i++)
    if (!strcmp(map->labels[i], label)) return map->labels[i];
  if (map->labels_size == map->labels_cap) {
    map->labels_cap = map->labels_cap ? 2 * map->labels_cap : 16;
    map->labels = xrealloc(map->labels, map->labels_cap * sizeof *map->labels);
  }
  map->labels[map->labels_size] = xstrdup(label);
  return map->labels[map->labels_size++];
}

static void read_semantic_map(semantic_map *map, const char *path,
                              int max_var) {
  FILE *file = fopen(path, "r");
  if (!file) {
    fprintf(stderr, "failed to open map %s: %s\n", path, strerror(errno));
    exit(1);
  }
  map->by_var = xcalloc((size_t)max_var + 1, sizeof *map->by_var);

  char line[4096];
  while (fgets(line, sizeof line, file)) {
    char *p = line;
    while (isspace((unsigned char)*p)) p++;
    if (!*p || *p == '#') continue;
    for (char *q = p; *q; q++)
      if (*q == ',') *q = ' ';

    char *end = 0;
    errno = 0;
    unsigned long var = strtoul(p, &end, 10);
    if (errno || end == p || var == 0 || var > (unsigned long)max_var) {
      continue;
    }
    char label[256];
    if (sscanf(end, "%255s", label) != 1) continue;
    map->by_var[var] = intern_label(map, label);
  }
  fclose(file);
}

static const char *label_for_var(const semantic_map *map, unsigned eidx,
                                 int input_vars) {
  if (map->by_var && map->by_var[eidx]) return map->by_var[eidx];
  if (input_vars > 0 && eidx <= (unsigned)input_vars) return "input";
  return "gate";
}

static void release_semantic_map(semantic_map *map) {
  for (size_t i = 0; i < map->labels_size; i++) free(map->labels[i]);
  free(map->labels);
  free(map->by_var);
}

typedef struct {
  const char *label;
  uint64_t counts[CAT_COUNT];
} group_counts;

typedef struct {
  group_counts *data;
  size_t size;
  size_t cap;
} group_table;

static group_counts *find_group(group_table *table, const char *label) {
  for (size_t i = 0; i < table->size; i++)
    if (!strcmp(table->data[i].label, label)) return table->data + i;
  if (table->size == table->cap) {
    table->cap = table->cap ? 2 * table->cap : 16;
    table->data = xrealloc(table->data, table->cap * sizeof *table->data);
  }
  group_counts *res = table->data + table->size++;
  memset(res, 0, sizeof *res);
  res->label = label;
  return res;
}

static void release_group_table(group_table *table) {
  free(table->data);
}

typedef struct {
  unsigned *parent;
  unsigned *size;
} dsu;

static unsigned dsu_find(dsu *d, unsigned x) {
  unsigned p = d->parent[x];
  if (p == x) return x;
  d->parent[x] = dsu_find(d, p);
  return d->parent[x];
}

static void dsu_union(dsu *d, unsigned a, unsigned b) {
  if (!a || !b) return;
  unsigned ra = dsu_find(d, a);
  unsigned rb = dsu_find(d, b);
  if (ra == rb) return;
  if (d->size[ra] < d->size[rb]) {
    unsigned tmp = ra;
    ra = rb;
    rb = tmp;
  }
  d->parent[rb] = ra;
  d->size[ra] += d->size[rb];
}

typedef struct {
  uint64_t irred_large_clauses;
  uint64_t active_large_clauses;
  uint64_t active_large_lits;
  uint64_t binary_active_edges;
  uint64_t len1;
  uint64_t len2;
  uint64_t len3;
  uint64_t len4;
  uint64_t len5_8;
  uint64_t len9p;
} graph_stats;

static int usable_watches(const watches *ws) {
#ifdef COMPACT
  return ws->size > 0;
#else
  return ws->begin && ws->end && ws->begin <= ws->end;
#endif
}

static void bucket_clause(graph_stats *stats, unsigned active_lits) {
  if (active_lits == 1)
    stats->len1++;
  else if (active_lits == 2)
    stats->len2++;
  else if (active_lits == 3)
    stats->len3++;
  else if (active_lits == 4)
    stats->len4++;
  else if (active_lits <= 8)
    stats->len5_8++;
  else
    stats->len9p++;
}

static unsigned *build_internal_to_external(kissat *solver, int max_var) {
  unsigned *external = xcalloc((size_t)solver->vars, sizeof *external);
  for (int eidx = 1; eidx <= max_var; eidx++) {
    if ((unsigned)eidx >= SIZE_STACK(solver->import)) continue;
    const import *imp = &PEEK_STACK(solver->import, (unsigned)eidx);
    if (!imp->imported) continue;
    if (imp->lit >= LITS) continue;
    unsigned iidx = IDX(imp->lit);
    if (iidx < solver->vars) external[iidx] = (unsigned)eidx;
  }
  return external;
}

static void scan_clause_for_graph(kissat *solver, clause *c,
                                  const unsigned *external,
                                  const unsigned char *cat, dsu *d,
                                  graph_stats *stats, unsigned **vars_ptr,
                                  size_t *vars_cap_ptr) {
    if (c->garbage || c->redundant) return;
    if (!c->size || c->size > solver->vars) return;
    stats->irred_large_clauses++;
    if (c->size > *vars_cap_ptr) {
      *vars_cap_ptr = c->size;
      *vars_ptr = xrealloc(*vars_ptr, *vars_cap_ptr * sizeof **vars_ptr);
    }
    unsigned *vars = *vars_ptr;

    unsigned nactive = 0;
    for (all_literals_in_clause(lit, c)) {
      if (lit >= LITS) continue;
      unsigned iidx = IDX(lit);
      if (iidx >= solver->vars) continue;
      unsigned eidx = external[iidx];
      if (!eidx || cat[eidx] != CAT_ACTIVE) continue;
      vars[nactive++] = eidx;
    }

    if (!nactive) return;
    stats->active_large_clauses++;
    stats->active_large_lits += nactive;
    bucket_clause(stats, nactive);
    for (unsigned i = 1; i < nactive; i++) dsu_union(d, vars[0], vars[i]);
}

static void scan_watched_clauses(kissat *solver, const unsigned *external,
                                 const unsigned char *cat, dsu *d,
                                 graph_stats *stats) {
  if (!solver->watches) return;
  const size_t arena_words = SIZE_STACK(solver->arena);
  unsigned char *seen_ref = xcalloc(arena_words ? arena_words : 1, 1);
  unsigned *vars = 0;
  size_t vars_cap = 0;

  if (solver->watching) {
    for (all_literals(lit)) {
      watches *ws = &WATCHES(lit);
      if (!usable_watches(ws)) continue;
      reference ref;
      for (all_binary_blocking_watch_ref(w, ref, *ws)) {
        if (w.type.binary) {
          unsigned other = w.binary.lit;
          if (other >= LITS) continue;
          if (lit > other) continue;
          unsigned a = external[IDX(lit)];
          unsigned b = external[IDX(other)];
          if (!a || !b || cat[a] != CAT_ACTIVE || cat[b] != CAT_ACTIVE) continue;
          stats->binary_active_edges++;
          dsu_union(d, a, b);
        } else {
          if (ref == INVALID_REF || (size_t)ref >= arena_words) continue;
          if (seen_ref[ref]) continue;
          seen_ref[ref] = 1;
          clause *c = kissat_unchecked_dereference_clause(solver, ref);
          scan_clause_for_graph(solver, c, external, cat, d, stats, &vars,
                                &vars_cap);
        }
      }
    }
  } else {
    for (all_literals(lit)) {
      watches *ws = &WATCHES(lit);
      if (!usable_watches(ws)) continue;
      for (all_binary_large_watches(w, *ws)) {
        if (w.type.binary) {
          unsigned other = w.binary.lit;
          if (other >= LITS) continue;
          if (lit > other) continue;
          unsigned a = external[IDX(lit)];
          unsigned b = external[IDX(other)];
          if (!a || !b || cat[a] != CAT_ACTIVE || cat[b] != CAT_ACTIVE) continue;
          stats->binary_active_edges++;
          dsu_union(d, a, b);
        } else {
          reference ref = w.large.ref;
          if (ref == INVALID_REF || (size_t)ref >= arena_words) continue;
          if (seen_ref[ref]) continue;
          seen_ref[ref] = 1;
          clause *c = kissat_unchecked_dereference_clause(solver, ref);
          scan_clause_for_graph(solver, c, external, cat, d, stats, &vars,
                                &vars_cap);
        }
      }
    }
  }

  free(vars);
  free(seen_ref);
}

static void print_counts(FILE *out, const char *record, int sample,
                         const char *extra, const uint64_t counts[CAT_COUNT]) {
  uint64_t total = 0;
  for (int i = 0; i < CAT_COUNT; i++) total += counts[i];
  double pct_active = total ? (double)counts[CAT_ACTIVE] / (double)total : 0.0;
  double pct_reduced =
      total ? (double)(counts[CAT_FIXED] + counts[CAT_ELIMINATED] +
                       counts[CAT_INACTIVE]) /
                  (double)total
            : 0.0;
  fprintf(out,
          "%s sample=%d %s total=%" PRIu64 " absent=%" PRIu64
          " active=%" PRIu64 " fixed=%" PRIu64 " eliminated=%" PRIu64
          " inactive=%" PRIu64 " pct_active=%.6f pct_reduced=%.6f\n",
          record, sample, extra, total, counts[CAT_ABSENT],
          counts[CAT_ACTIVE], counts[CAT_FIXED], counts[CAT_ELIMINATED],
          counts[CAT_INACTIVE], pct_active, pct_reduced);
}

static void insert_top(unsigned top[10], unsigned value) {
  for (int i = 0; i < 10; i++) {
    if (value <= top[i]) continue;
    for (int j = 9; j > i; j--) top[j] = top[j - 1];
    top[i] = value;
    return;
  }
}

static void dump_snapshot(FILE *out, kissat *solver, int sample, int max_var,
                          int input_vars, int bins, double elapsed,
                          int result, const semantic_map *map) {
  unsigned char *cat = xcalloc((size_t)max_var + 1, sizeof *cat);
  uint64_t all_counts[CAT_COUNT] = {0};
  group_table groups;
  memset(&groups, 0, sizeof groups);

  for (int eidx = 1; eidx <= max_var; eidx++) {
    enum category c = classify_external(solver, (unsigned)eidx);
    cat[eidx] = (unsigned char)c;
    all_counts[c]++;
    group_counts *group =
        find_group(&groups, label_for_var(map, (unsigned)eidx, input_vars));
    group->counts[c]++;
  }

  unsigned *external = build_internal_to_external(solver, max_var);
  dsu d;
  d.parent = xcalloc((size_t)max_var + 1, sizeof *d.parent);
  d.size = xcalloc((size_t)max_var + 1, sizeof *d.size);
  uint64_t active_vars = 0;
  for (int eidx = 1; eidx <= max_var; eidx++) {
    if (cat[eidx] != CAT_ACTIVE) continue;
    d.parent[eidx] = (unsigned)eidx;
    d.size[eidx] = 1;
    active_vars++;
  }

  graph_stats stats;
  memset(&stats, 0, sizeof stats);
  scan_watched_clauses(solver, external, cat, &d, &stats);

  uint64_t components = 0;
  uint64_t singletons = 0;
  unsigned largest = 0;
  unsigned top[10] = {0};
  for (int eidx = 1; eidx <= max_var; eidx++) {
    if (cat[eidx] != CAT_ACTIVE) continue;
    unsigned root = dsu_find(&d, (unsigned)eidx);
    if (root != (unsigned)eidx) continue;
    unsigned size = d.size[root];
    components++;
    if (size == 1) singletons++;
    if (size > largest) largest = size;
    insert_top(top, size);
  }

  fprintf(out,
          "SNAPSHOT sample=%d elapsed=%.3f result=%d max_var=%d solver_vars=%u"
          " solver_active=%u solver_unassigned=%u conflicts=%" PRIu64
          " irredundant=%" PRIu64 " redundant=%" PRIu64
          " binary=%" PRIu64 "\n",
          sample, elapsed, result, max_var, solver->vars, solver->active,
          solver->unassigned, (uint64_t)CONFLICTS,
          (uint64_t)IRREDUNDANT_CLAUSES, (uint64_t)REDUNDANT_CLAUSES,
          (uint64_t)BINARY_CLAUSES);

  print_counts(out, "ALL", sample, "label=all", all_counts);

  for (size_t i = 0; i < groups.size; i++) {
    char extra[512];
    snprintf(extra, sizeof extra, "label=%s", groups.data[i].label);
    print_counts(out, "GROUP", sample, extra, groups.data[i].counts);
  }

  if (bins < 1) bins = 1;
  for (int bin = 0; bin < bins; bin++) {
    int lo = 1 + (int)(((int64_t)bin * max_var) / bins);
    int hi = (int)(((int64_t)(bin + 1) * max_var) / bins);
    if (hi < lo) hi = lo;
    uint64_t counts[CAT_COUNT] = {0};
    for (int eidx = lo; eidx <= hi && eidx <= max_var; eidx++) counts[cat[eidx]]++;
    char extra[128];
    snprintf(extra, sizeof extra, "index=%d lo=%d hi=%d", bin, lo, hi);
    print_counts(out, "BIN", sample, extra, counts);
  }

  const double avg_large_clause =
      stats.active_large_clauses
          ? (double)stats.active_large_lits / (double)stats.active_large_clauses
          : 0.0;
  const double largest_frac =
      active_vars ? (double)largest / (double)active_vars : 0.0;
  fprintf(out,
          "GRAPH sample=%d active_vars=%" PRIu64 " components=%" PRIu64
          " largest=%u largest_frac=%.6f singletons=%" PRIu64
          " irred_large_clauses=%" PRIu64 " active_large_clauses=%" PRIu64
          " active_large_lits=%" PRIu64 " avg_active_large_clause=%.3f"
          " binary_active_edges=%" PRIu64 " len1=%" PRIu64
          " len2=%" PRIu64 " len3=%" PRIu64 " len4=%" PRIu64
          " len5_8=%" PRIu64 " len9p=%" PRIu64 "\n",
          sample, active_vars, components, largest, largest_frac, singletons,
          stats.irred_large_clauses, stats.active_large_clauses,
          stats.active_large_lits, avg_large_clause, stats.binary_active_edges,
          stats.len1, stats.len2, stats.len3, stats.len4, stats.len5_8,
          stats.len9p);

  for (int i = 0; i < 10 && top[i]; i++)
    fprintf(out, "COMPONENT sample=%d rank=%d size=%u\n", sample, i + 1,
            top[i]);

  fputc('\n', out);
  fflush(out);

  release_group_table(&groups);
  free(external);
  free(d.parent);
  free(d.size);
  free(cat);
}

static void usage(const char *name) {
  fprintf(stderr,
          "usage: %s [options] input.cnf\n"
          "options:\n"
          "  --seconds N           solve seconds before one snapshot, default 60\n"
          "  --interval N          accepted for compatibility; currently ignored\n"
          "  --bins N              order-bin count, default 40\n"
          "  --inputs N            variables 1..N are labeled input, default 384\n"
          "  --map PATH            whitespace/csv map: external_var label ...\n"
          "  --configuration NAME  Kissat config: default, sat, unsat, plain, basic\n"
          "  --seed N              Kissat random seed\n"
          "  --out PATH            write probe records to PATH instead of stdout\n"
          "  --verbose-kissat      allow normal Kissat output too\n",
          name);
}

static const char *next_value(int argc, char **argv, int *i, const char *opt) {
  if (*i + 1 >= argc) {
    fprintf(stderr, "missing value after %s\n", opt);
    exit(2);
  }
  return argv[++*i];
}

static double parse_double_option(const char *s, const char *opt) {
  char *end = 0;
  errno = 0;
  double value = strtod(s, &end);
  if (errno || end == s || *end) {
    fprintf(stderr, "bad numeric value for %s: %s\n", opt, s);
    exit(2);
  }
  return value;
}

static int parse_int_option(const char *s, const char *opt) {
  char *end = 0;
  errno = 0;
  long value = strtol(s, &end, 10);
  if (errno || end == s || *end || value < 0 || value > INT_MAX) {
    fprintf(stderr, "bad integer value for %s: %s\n", opt, s);
    exit(2);
  }
  return (int)value;
}

int main(int argc, char **argv) {
  double total_seconds = 60.0;
  double interval_seconds = 60.0;
  int bins = 40;
  int input_vars = 384;
  int seed = -1;
  int verbose_kissat = 0;
  const char *configuration = "default";
  const char *map_path = 0;
  const char *out_path = 0;
  const char *cnf_path = 0;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
      usage(argv[0]);
      return 0;
    } else if (!strcmp(arg, "--seconds")) {
      total_seconds = parse_double_option(next_value(argc, argv, &i, arg), arg);
    } else if (!strncmp(arg, "--seconds=", 10)) {
      total_seconds = parse_double_option(arg + 10, "--seconds");
    } else if (!strcmp(arg, "--interval")) {
      interval_seconds =
          parse_double_option(next_value(argc, argv, &i, arg), arg);
    } else if (!strncmp(arg, "--interval=", 11)) {
      interval_seconds = parse_double_option(arg + 11, "--interval");
    } else if (!strcmp(arg, "--bins")) {
      bins = parse_int_option(next_value(argc, argv, &i, arg), arg);
    } else if (!strncmp(arg, "--bins=", 7)) {
      bins = parse_int_option(arg + 7, "--bins");
    } else if (!strcmp(arg, "--inputs")) {
      input_vars = parse_int_option(next_value(argc, argv, &i, arg), arg);
    } else if (!strncmp(arg, "--inputs=", 9)) {
      input_vars = parse_int_option(arg + 9, "--inputs");
    } else if (!strcmp(arg, "--map")) {
      map_path = next_value(argc, argv, &i, arg);
    } else if (!strncmp(arg, "--map=", 6)) {
      map_path = arg + 6;
    } else if (!strcmp(arg, "--configuration") || !strcmp(arg, "--config")) {
      configuration = next_value(argc, argv, &i, arg);
    } else if (!strncmp(arg, "--configuration=", 16)) {
      configuration = arg + 16;
    } else if (!strncmp(arg, "--config=", 9)) {
      configuration = arg + 9;
    } else if (!strcmp(arg, "--default") || !strcmp(arg, "--sat") ||
               !strcmp(arg, "--unsat") || !strcmp(arg, "--plain") ||
               !strcmp(arg, "--basic")) {
      configuration = arg + 2;
    } else if (!strcmp(arg, "--seed")) {
      seed = parse_int_option(next_value(argc, argv, &i, arg), arg);
    } else if (!strncmp(arg, "--seed=", 7)) {
      seed = parse_int_option(arg + 7, "--seed");
    } else if (!strcmp(arg, "--out")) {
      out_path = next_value(argc, argv, &i, arg);
    } else if (!strncmp(arg, "--out=", 6)) {
      out_path = arg + 6;
    } else if (!strcmp(arg, "--verbose-kissat")) {
      verbose_kissat = 1;
    } else if (arg[0] == '-') {
      fprintf(stderr, "unknown option: %s\n", arg);
      usage(argv[0]);
      return 2;
    } else if (!cnf_path) {
      cnf_path = arg;
    } else {
      fprintf(stderr, "multiple CNF paths: %s and %s\n", cnf_path, arg);
      return 2;
    }
  }

  if (!cnf_path || total_seconds < 0 || interval_seconds <= 0 || bins <= 0) {
    usage(argv[0]);
    return 2;
  }

  FILE *out = stdout;
  if (out_path) {
    out = fopen(out_path, "w");
    if (!out) {
      fprintf(stderr, "failed to open output %s: %s\n", out_path,
              strerror(errno));
      return 1;
    }
  }

  kissat *solver = kissat_init();
  struct sigaction action;
  memset(&action, 0, sizeof action);
  action.sa_handler = alarm_handler;
  sigemptyset(&action.sa_mask);
  if (sigaction(SIGALRM, &action, 0)) {
    fprintf(stderr, "failed to install alarm handler: %s\n", strerror(errno));
    kissat_release(solver);
    if (out_path) fclose(out);
    return 1;
  }
  alarm_solver = solver;

  if (!kissat_has_configuration(configuration) ||
      !kissat_set_configuration(solver, configuration)) {
    fprintf(stderr, "unknown Kissat configuration: %s\n", configuration);
    kissat_release(solver);
    if (out_path) fclose(out);
    return 2;
  }
  if (seed >= 0) kissat_set_option(solver, "seed", seed);
  if (!verbose_kissat) kissat_set_option(solver, "quiet", 1);

  int max_var, clauses;
  double parse_start = now_seconds();
  parse_dimacs(solver, cnf_path, &max_var, &clauses);
  double parse_elapsed = now_seconds() - parse_start;
  if (input_vars > max_var) input_vars = max_var;

  semantic_map map;
  memset(&map, 0, sizeof map);
  if (map_path) read_semantic_map(&map, map_path, max_var);

  fprintf(out,
          "# kissat_hardcore_probe cnf=%s max_var=%d clauses=%d"
          " parse_seconds=%.3f configuration=%s seed=%d sample_seconds=%.3f"
          " bins=%d inputs=%d map=%s\n",
          cnf_path, max_var, clauses, parse_elapsed, configuration, seed,
          total_seconds, bins, input_vars, map_path ? map_path : "");
  fflush(out);

  double solve_start = now_seconds();
  int result = 0;
  alarm_seen = 0;
  if (total_seconds > 0) arm_timer(total_seconds);
  result = kissat_solve(solver);
  disarm_timer();
  const double elapsed = now_seconds() - solve_start;
  dump_snapshot(out, solver, 0, max_var, input_vars, bins, elapsed, result,
                &map);

  fprintf(out, "RESULT result=%d elapsed=%.3f samples=1 alarm=%d\n", result,
          elapsed, (int)alarm_seen);

  alarm_solver = 0;
  disarm_timer();
  release_semantic_map(&map);
  kissat_release(solver);
  if (out_path) fclose(out);
  return result == 10 || result == 20 || result == 0 ? 0 : result;
}
