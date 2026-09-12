#define _POSIX_C_SOURCE 200809L

#include "internal.h"
#include "inline.h"
#include "kissat.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
  double start;
  double limit;
} termination_state;

static double now_seconds(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static int terminate_after_limit(void *ptr) {
  termination_state *state = (termination_state *)ptr;
  return now_seconds() - state->start >= state->limit;
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

static void print_counts(const char *prefix, uint64_t counts[CAT_COUNT]) {
  uint64_t total = 0;
  for (int i = 0; i < CAT_COUNT; i++) total += counts[i];
  printf("%s total=%" PRIu64 " absent=%" PRIu64 " active=%" PRIu64
         " fixed=%" PRIu64 " eliminated=%" PRIu64 " inactive=%" PRIu64,
         prefix, total, counts[CAT_ABSENT], counts[CAT_ACTIVE],
         counts[CAT_FIXED], counts[CAT_ELIMINATED], counts[CAT_INACTIVE]);
  if (total) {
    printf(" pct_active=%.6f pct_reduced=%.6f",
           (double)counts[CAT_ACTIVE] / (double)total,
           (double)(counts[CAT_FIXED] + counts[CAT_ELIMINATED] +
                    counts[CAT_INACTIVE]) /
               (double)total);
  }
  putchar('\n');
}

static void dump_heatmap(kissat *solver, int max_var, int bins,
                         double elapsed, int result) {
  const int input_vars = max_var < 384 ? max_var : 384;
  const int gate_vars = max_var > 384 ? max_var - 384 : 0;

  printf("SNAPSHOT elapsed=%.3f result=%d max_var=%d gate_vars=%d "
         "solver_vars=%u solver_active=%u solver_unassigned=%u\n",
         elapsed, result, max_var, gate_vars, solver->vars, solver->active,
         solver->unassigned);

  uint64_t all_counts[CAT_COUNT] = {0};
  for (int eidx = 1; eidx <= max_var; eidx++) {
    all_counts[classify_external(solver, (unsigned)eidx)]++;
  }
  print_counts("ALL", all_counts);

  const char *names[3] = {"A", "B", "C"};
  for (int block = 0; block < 3; block++) {
    uint64_t counts[CAT_COUNT] = {0};
    int lo = 1 + 128 * block;
    int hi = lo + 127;
    if (lo > input_vars) hi = lo - 1;
    if (hi > input_vars) hi = input_vars;
    for (int eidx = lo; eidx <= hi; eidx++) {
      counts[classify_external(solver, (unsigned)eidx)]++;
    }
    char prefix[64];
    snprintf(prefix, sizeof prefix, "INPUT_%s", names[block]);
    print_counts(prefix, counts);
  }

  printf("BIN index start_gate end_gate absent active fixed eliminated inactive "
         "total pct_active pct_reduced\n");
  for (int bin = 0; bin < bins; bin++) {
    int start = (int)(((int64_t)bin * gate_vars) / bins);
    int end = (int)(((int64_t)(bin + 1) * gate_vars) / bins) - 1;
    uint64_t counts[CAT_COUNT] = {0};
    for (int gate = start; gate <= end; gate++) {
      unsigned eidx = (unsigned)(385 + gate);
      counts[classify_external(solver, eidx)]++;
    }
    uint64_t total = 0;
    for (int i = 0; i < CAT_COUNT; i++) total += counts[i];
    double pct_active =
        total ? (double)counts[CAT_ACTIVE] / (double)total : 0.0;
    double pct_reduced =
        total ? (double)(counts[CAT_FIXED] + counts[CAT_ELIMINATED] +
                         counts[CAT_INACTIVE]) /
                    (double)total
              : 0.0;
    printf("%d %d %d %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64
           " %" PRIu64 " %" PRIu64 " %.6f %.6f\n",
           bin, start, end, counts[CAT_ABSENT], counts[CAT_ACTIVE],
           counts[CAT_FIXED], counts[CAT_ELIMINATED], counts[CAT_INACTIVE],
           total, pct_active, pct_reduced);
  }
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    fprintf(stderr, "usage: kissat_core_snapshot cnf time_seconds [bins]\n");
    return 2;
  }

  const char *path = argv[1];
  double limit = atof(argv[2]);
  int bins = argc == 4 ? atoi(argv[3]) : 40;
  if (limit < 0 || bins <= 0) {
    fprintf(stderr, "bad limit or bin count\n");
    return 2;
  }

  kissat *solver = kissat_init();
  int max_var, clauses;
  double parse_start = now_seconds();
  parse_dimacs(solver, path, &max_var, &clauses);
  double parse_elapsed = now_seconds() - parse_start;
  fprintf(stderr, "parsed max_var=%d clauses=%d in %.3fs\n", max_var,
          clauses, parse_elapsed);

  termination_state term;
  term.start = now_seconds();
  term.limit = limit;
  if (limit > 0) kissat_set_terminate(solver, &term, terminate_after_limit);

  int result = kissat_solve(solver);
  double solve_elapsed = now_seconds() - term.start;
  dump_heatmap(solver, max_var, bins, solve_elapsed, result);
  kissat_release(solver);
  return result == 10 || result == 20 || result == 0 ? 0 : result;
}
