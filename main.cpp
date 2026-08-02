#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>

#define MAX_THREADS 32

using namespace std;

// Private registers y + barrier between phases => EREW
static void pps_unscaled(vector<long long> &M, bool print_steps)
{
	int n = (int)M.size();
	if (n <= 1) return;

	vector<long long> y(M);

	for (int j = 0; (1 << j) < n; j++) {
		int off = 1 << j;

		#pragma omp parallel for num_threads(n)
		for (int i = off; i < n; i++)
			y[i] = y[i] + M[i - off];

		#pragma omp parallel for num_threads(n)
		for (int i = off; i < n; i++)
			M[i] = y[i];

		if (print_steps) {
			printf("Krok %d (2^%d = %2d): ", j + 1, j, off);
			for (int i = 0; i < n; i++) printf("%lld ", M[i]);
			printf("\n");
		}
	}
}

// Block variant, T(n,p) = O(n/p + log p)
static void pps_scaled(vector<long long> &X, int p)
{
	int n = (int)X.size();
	vector<long long> Z(p, 0);

	// Phase 1: local prefix sums within each block
	#pragma omp parallel num_threads(p)
	{
		int tid  = omp_get_thread_num();
		int base = n / p, rem = n % p;
		int start = tid * base + (tid < rem ? tid : rem);
		int end   = start + base + (tid < rem ? 1 : 0);

		long long acc = 0;
		for (int k = start; k < end; k++) { acc += X[k]; X[k] = acc; }
		Z[tid] = (end > start) ? X[end - 1] : 0;
	}

	// Phase 2: prefix sum over block boundaries
	pps_unscaled(Z, false);

	// Phase 3: add offset Z[i-1] to block i
	#pragma omp parallel num_threads(p)
	{
		int tid = omp_get_thread_num();
		if (tid > 0) {
			int base = n / p, rem = n % p;
			int start = tid * base + (tid < rem ? tid : rem);
			int end   = start + base + (tid < rem ? 1 : 0);

			long long off = Z[tid - 1];
			for (int k = start; k < end; k++) X[k] += off;
		}
	}
}

static void print_array(const char *label, const vector<long long> &A)
{
	int n = (int)A.size();
	printf("%s", label);
	if (n <= 64) {
		for (int i = 0; i < n; i++) printf("%lld ", A[i]);
	} else {
		for (int i = 0; i < 10; i++) printf("%lld ", A[i]);
		printf("... ");
		for (int i = n - 5; i < n; i++) printf("%lld ", A[i]);
	}
	printf("\n");
}

static bool verify(const vector<long long> &A)
{
	long long s = 0;
	for (size_t i = 0; i < A.size(); i++) {
		s += (long long)(i + 1);
		if (A[i] != s) return false;
	}
	return true;
}

int main(int argc, char **argv)
{
	int choice = 0;
	printf("Paralelni prefixove soucty (EREW PRAM)\n");
	printf("  [1] Neskalovany algoritmus (max. 32 prvku)\n");
	printf("  [2] Skalovany algoritmus (N neomezeno, max. 32 vlaken)\n");
	printf("Zvolte algoritmus: ");
	if (scanf("%d", &choice) != 1 || (choice != 1 && choice != 2)) {
		printf("Neplatna volba.\n");
		return 1;
	}

	long long N = 0;
	printf("Zadejte velikost vstupniho pole N: ");
	if (scanf("%lld", &N) != 1 || N < 1) {
		printf("N musi byt cele cislo >= 1.\n");
		return 1;
	}

	if (choice == 1) {
		int n = (N < MAX_THREADS) ? (int)N : MAX_THREADS;
		if (N > MAX_THREADS)
			printf("Pole je omezeno na %d prvku, pouzivam n = %d.\n", MAX_THREADS, n);

		vector<long long> M(n);
		for (int i = 0; i < n; i++) M[i] = i + 1;

		print_array("\nVstup:    ", M);
		printf("\n");
		pps_unscaled(M, true);
		print_array("\nVysledek: ", M);
		printf(verify(M) ? "[OK] Shoduje se se sekvencnim vypoctem.\n"
		                     : "[CHYBA] Neshoduje se!\n");
	} else {
		int p = 0;
		printf("Zadejte pocet aktivnich vlaken p (max %d): ", MAX_THREADS);
		if (scanf("%d", &p) != 1 || p < 1) {
			printf("p musi byt cele cislo >= 1.\n");
			return 1;
		}
		if (p > MAX_THREADS) { p = MAX_THREADS; printf("Pocet vlaken omezen na %d.\n", p); }
		if (p > N)           { p = (int)N;      printf("Pocet vlaken snizen na N (%d).\n", p); }

		int n = (int)N;
		vector<long long> X(n);
		for (int i = 0; i < n; i++) X[i] = i + 1;

		double t0 = omp_get_wtime();
		pps_scaled(X, p);
		double t1 = omp_get_wtime();

		printf("\nParametry: N = %d, p = %d, q = n/p ~ %d\n", n, p, n / p);
		print_array("Vysledek: ", X);
		printf("Y[%d] = %lld  (ocekavano N(N+1)/2 = %lld)\n",
		       n - 1, X[n - 1], (long long)n * (n + 1) / 2);
		printf("Cas vypoctu: %.6f s\n", t1 - t0);
		printf(verify(X) ? "[OK] Shoduje se se sekvencnim vypoctem.\n"
		                     : "[CHYBA] Neshoduje se!\n");
	}

	return 0;
}
