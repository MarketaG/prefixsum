// ============================================================================
//  Uloha c.3 - Paralelni prefixove soucty (PPS)
//  C++ / OpenMP
//
//  Program implementuje dva algoritmy z prednasky c.12 (EREW PRAM):
//
//   [1] NESKALOVANY PPS (Algoritmus 5, Hillis-Steele)
//       - n procesoru pro n hodnot, slozitost O(log n), cenove neoptimalni
//       - vstup omezen na max. 32 prvku (kazdy dilci soucet = samostatne
//         vlakno; max. 32 paralelne bezicich vlaken)
//       - hodnoty vstupu: posloupnost 1 .. min(N, 32)
//
//   [2] SKALOVANY PPS (Algoritmus 6, blokova varianta)
//       - N neomezeno, pocet vlaken p zadava uzivatel (max. 32)
//       - hodnoty vstupu: posloupnost 1 .. N
//       - faze 1: lokalni prefixove soucty bloku          O(n/p)
//         faze 2: neskalovany PPS nad okraji bloku (Z)    O(log p)
//         faze 3: pricteni offsetu Z[i-1] k blokum i>=1   O(n/p)
//         => T(n,p) = O(n/p + log p), cenove optimalni pro n >> p
//
//  Preklad:  g++ -fopenmp -O2 -o pps main.cpp
//     MSVC:  cl /openmp /EHsc main.cpp
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <omp.h>

#define MAX_THREADS 32

using namespace std;

// ---------------------------------------------------------------------------
// Neskalovany EREW PRAM PPS (Algoritmus 5) - inclusive scan in-place nad M.
// Kazdy "procesor" P_i ma soukromy registr y[i]. V kazdem z ceil(log2 n)
// kroku probehnou dve oddelene paralelni faze:
//   A) y[i] = y[i] + M[i - 2^j]   ... kazdy cte JINOU bunku -> exkluzivni cteni
//   B) M[i] = y[i]                ... kazdy pise JINOU bunku -> exkluzivni zapis
// Bariera mezi fazemi (implicitni na konci parallel-for) brani soubehu nad M,
// diky pomocnemu registru y je tedy algoritmus EREW (nikoliv CREW).
// ---------------------------------------------------------------------------
static void pps_neskalovany(vector<long long> &M, bool vypis_kroky)
{
	int n = (int)M.size();
	if (n <= 1) return;

	vector<long long> y(M);                     // y[i] <- M[i] (soukrome registry)

	for (int j = 0; (1 << j) < n; j++) {
		int off = 1 << j;                       // 2^j

		#pragma omp parallel for num_threads(n)
		for (int i = off; i < n; i++)
			y[i] = y[i] + M[i - off];           // faze A - exkluzivni cteni

		#pragma omp parallel for num_threads(n)
		for (int i = off; i < n; i++)
			M[i] = y[i];                        // faze B - exkluzivni zapis

		if (vypis_kroky) {
			printf("Krok %d (2^%d = %2d): ", j + 1, j, off);
			for (int i = 0; i < n; i++) printf("%lld ", M[i]);
			printf("\n");
		}
	}
}

// ---------------------------------------------------------------------------
// Skalovany EREW PRAM PPS (Algoritmus 6) - inclusive scan in-place nad X,
// p = pocet vlaken (bloku).
// ---------------------------------------------------------------------------
static void pps_skalovany(vector<long long> &X, int p)
{
	int n = (int)X.size();
	vector<long long> Z(p, 0);                  // prave krajni hodnoty bloku

	// Faze 1: kazde vlakno sekvencne spocte prefixovy soucet nad svym blokem
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

	// Faze 2: PPS nad polem Z pomoci neskalovaneho algoritmu (Algoritmus 5)
	pps_neskalovany(Z, false);                  // Z[i] = soucet bloku 0..i

	// Faze 3: k prvkum bloku i >= 1 pricti offset Z[i-1]
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

// ---------------------------------------------------------------------------
static void vypis_pole(const char *popis, const vector<long long> &A)
{
	int n = (int)A.size();
	printf("%s", popis);
	if (n <= 64) {
		for (int i = 0; i < n; i++) printf("%lld ", A[i]);
	} else {
		for (int i = 0; i < 10; i++) printf("%lld ", A[i]);
		printf("... ");
		for (int i = n - 5; i < n; i++) printf("%lld ", A[i]);
	}
	printf("\n");
}

static bool zkontroluj(const vector<long long> &A)
{
	long long s = 0;
	for (size_t i = 0; i < A.size(); i++) {
		s += (long long)(i + 1);
		if (A[i] != s) return false;
	}
	return true;
}

// ---------------------------------------------------------------------------
int main(int argc, char **argv)
{
	int volba = 0;
	printf("Paralelni prefixove soucty (EREW PRAM)\n");
	printf("  [1] Neskalovany algoritmus (max. 32 prvku)\n");
	printf("  [2] Skalovany algoritmus (N neomezeno, max. 32 vlaken)\n");
	printf("Zvolte algoritmus: ");
	if (scanf("%d", &volba) != 1 || (volba != 1 && volba != 2)) {
		printf("Neplatna volba.\n");
		return 1;
	}

	long long N = 0;
	printf("Zadejte velikost vstupniho pole N: ");
	if (scanf("%lld", &N) != 1 || N < 1) {
		printf("N musi byt cele cislo >= 1.\n");
		return 1;
	}

	if (volba == 1) {
		// ------------------- neskalovana varianta -------------------------
		int n = (N < MAX_THREADS) ? (int)N : MAX_THREADS;
		if (N > MAX_THREADS)
			printf("Pole je omezeno na %d prvku, pouzivam n = %d.\n", MAX_THREADS, n);

		vector<long long> M(n);
		for (int i = 0; i < n; i++) M[i] = i + 1;      // hodnoty 1 .. min(N,32)

		vypis_pole("\nVstup:    ", M);
		printf("\n");
		pps_neskalovany(M, true);
		vypis_pole("\nVysledek: ", M);
		printf(zkontroluj(M) ? "[OK] Shoduje se se sekvencnim vypoctem.\n"
		                     : "[CHYBA] Neshoduje se!\n");
	} else {
		// -------------------- skalovana varianta --------------------------
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
		for (int i = 0; i < n; i++) X[i] = i + 1;      // hodnoty 1 .. N

		double t0 = omp_get_wtime();
		pps_skalovany(X, p);
		double t1 = omp_get_wtime();

		printf("\nParametry: N = %d, p = %d, q = n/p ~ %d\n", n, p, n / p);
		vypis_pole("Vysledek: ", X);
		printf("Y[%d] = %lld  (ocekavano N(N+1)/2 = %lld)\n",
		       n - 1, X[n - 1], (long long)n * (n + 1) / 2);
		printf("Cas vypoctu: %.6f s\n", t1 - t0);
		printf(zkontroluj(X) ? "[OK] Shoduje se se sekvencnim vypoctem.\n"
		                     : "[CHYBA] Neshoduje se!\n");
	}

	return 0;
}
