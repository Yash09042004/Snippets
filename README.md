# Snippets

# Competitive Programming Templates

```cpp

#include <bits/stdc++.h>
using namespace std;

#define ll long long
const ll MOD = 1e9 + 7;
const int MAXN = 1e6 + 5;

//--------------------------------------------------- Prime Checking and Sieve ----------------------------------------//

// Check if a number is prime
template <class T> bool isPrime(T n) {
    if (n <= 1) return false;
    for (T i = 2; i * i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

// Sieve of Eratosthenes
vector<bool> prime(MAXN, true);
void sieve() {
    prime[0] = prime[1] = false;
    for (int i = 2; i * i <= MAXN; i++) {
        if (prime[i]) {
            for (int j = i * i; j <= MAXN; j += i) {
                prime[j] = false;
            }
        }
    }
}

// Smallest Prime Factor (SPF) Sieve
vector<ll> spf(MAXN);
void spfsieve() {
    for (int i = 2; i < MAXN; i++) {
        if (spf[i] == 0) {
            spf[i] = i;
            for (ll j = (ll)i * i; j < MAXN; j += i) {
                if (spf[j] == 0) {
                    spf[j] = i;
                }
            }
        }
    }
}

//--------------------------------------------------- Range XOR ------------------------------------------------------//

ll xor_upto(ll n) {
    if (n % 4 == 0) return n;
    if (n % 4 == 1) return 1;
    if (n % 4 == 2) return n + 1;
    return 0;
}

ll range_xor(ll l, ll r) {
    return xor_upto(r) ^ xor_upto(l - 1);
}

//--------------------------------------------- Modular Arithmetic ----------------------------------------------------//

ll mod_add(ll a, ll b, ll m) { return (((a + b) % m) + m) % m; }
ll mod_sub(ll a, ll b, ll m) { return (((a - b) % m) + m) % m; }
ll mod_mul(ll a, ll b, ll m) { return (((a * b) % m) + m) % m; }
ll mod_inverse(ll a, ll m) { return binpowmod(a, m - 2, m); }  // for prime m
ll mod_div(ll a, ll b, ll m) { return mod_mul(a, mod_inverse(b, m), m); }

// Modular Exponentiation
template <typename T> T binpowmod(T base, T exp, T mod) {
    T result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) result = (result * base) % mod;
        base = (base * base) % mod;
        exp >>= 1;
    }
    return result;
}

//----------------------------------------- Combinatorics and Factorials ---------------------------------------------//

vector<ll> fact(MAXN), inv_fact(MAXN);
void factorial(ll mod) {
    fact[0] = inv_fact[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        fact[i] = fact[i - 1] * i % mod;
        inv_fact[i] = mod_inverse(fact[i], mod);
    }
}

ll nCr(ll n, ll r, ll mod) {
    if (r > n || r < 0) return 0;
    return fact[n] * inv_fact[r] % mod * inv_fact[n - r] % mod;
}

//-------------------------------------------- Search Algorithms ------------------------------------------------------//

// Binary Search
template <typename T>
int binary_search(const vector<T>& arr, T target) {
    int left = 0, right = arr.size() - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// Two-pointer technique for finding pairs with a target sum
bool two_pointer_sum(const vector<int>& arr, int target) {
    int left = 0, right = arr.size() - 1;
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) return true;
        else if (sum < target) left++;
        else right--;
    }
    return false;
}

//------------------------------------------- Greedy Algorithms -------------------------------------------------------//

// Fractional Knapsack Problem
bool cmp(pair<int, int> a, pair<int, int> b) {
    return (double)a.first / a.second > (double)b.first / b.second;
}

double fractional_knapsack(vector<pair<int, int>>& items, int W) {
    sort(items.begin(), items.end(), cmp);
    double total_value = 0;
    for (auto& item : items) {
        if (W >= item.second) {
            W -= item.second;
            total_value += item.first;
        } else {
            total_value += item.first * ((double)W / item.second);
            break;
        }
    }
    return total_value;
}

//------------------------------------------- Graph Algorithms --------------------------------------------------------//

// Breadth-First Search
vector<int> adj[MAXN];
vector<bool> visited(MAXN, false);

void bfs(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;
    while (!q.empty()) {
        int v = q.front();
        q.pop();
        for (int u : adj[v]) {
            if (!visited[u]) {
                visited[u] = true;
                q.push(u);
            }
        }
    }
}

// Depth-First Search
void dfs(int v) {
    visited[v] = true;
    for (int u : adj[v]) {
        if (!visited[u]) {
            dfs(u);
        }
    }
}

// Dijkstra’s Algorithm for Shortest Paths
template <typename T = int> struct Dijkstra {
    struct Edge {
        T v, w;
        Edge(T V = 0, T W = 0): v(V), w(W) {}
        bool operator < (const Edge& e) const { return w > e.w; }
    };
    vector<vector<Edge>> adj;
    
    Dijkstra(int edges, bool indirected = true) {
        adj = vector<vector<Edge>>(edges);
        for (int i = 0, u, v, w; i < edges; i++) {
            cin >> u >> v >> w;
            adj[u].push_back(Edge(v, w));
            if (indirected) adj[v].push_back(Edge(u, w));
        }
    }

    vector<T> get_dist(int src) {
        int n = adj.size();
        vector<T> dist(n, LLONG_MAX);
        dist[src] = 0;
        priority_queue<Edge> Dij;
        Dij.push(Edge(src, 0));
        while (!Dij.empty()) {
            auto [u, cost] = Dij.top();
            Dij.pop();
            for (auto& [v, w] : adj[u]) {
                if (dist[v] > dist[u] + w) {
                    dist[v] = dist[u] + w;
                    Dij.push(Edge(v, dist[v]));
                }
            }
        }
        return dist;
    }
};

//----------------------------------------- Big Number Multiplication -------------------------------------------------//

// Function to multiply two large numbers represented as strings
string multiplyBigNumbers(const string& num1, const string& num2) {
    if (num1 == "0" || num2 == "0") return "0";
    int n = num1.size(), m = num2.size();
    vector<int> result(n + m, 0);
    for (int i = n - 1; i >= 0; i--) {
        for (int j = m - 1; j >= 0; j--) {
            int mul = (num1[i] - '0') * (num2[j] - '0');
            int sum = mul + result[i + j + 1];
            result[i + j + 1] = sum % 10;
            result[i + j] += sum / 10;
        }
    }
    string resultStr;
    for (int num : result) {
        if (!(resultStr.empty() && num == 0)) resultStr.push_back(num + '0');
    }
    return resultStr;
}

// Grundy Number (Nim Game Theory)
int grundy(int n) {
    if (n == 0) return 0;
    unordered_set<int> s;
    for (int i = 1; i <= n / 2; i++) {
        s.insert(grundy(n - i));
    }
    int g = 0;
    while (s.count(g)) g++;
    return g;
}

//----------------------------------------- Disjoint Set Union -------------------------------------------------//

class disjointSet {
    private:
    vector<int> rank, size, parent;
    public:
    disjointSet(int n) {
        rank.resize(n+1, 0);
        size.resize(n+1);
        parent.resize(n+1);
        for(int i = 0; i < n+1; i++) {
            parent[i] = i;
            size[i] = 1;
        }
    }
    int findUPar(int node) 
    {
        if(node == parent[node]) return node;

        return parent[node] = findUPar(parent[node]);
    }
    void unionByRank(int u, int v)
    {
        int ulp_u = findUPar(u);
        int ulp_v = findUPar(v);

        if(ulp_u == ulp_v) return;

        if(rank[ulp_u] < rank[ulp_v])
        {
            parent[ulp_u] = ulp_v;
        }
        else if(rank[ulp_u] > rank[ulp_v])
        {
            parent[ulp_v] = ulp_u;
        }
        else
        {
            parent[ulp_v] = ulp_u;
        rank[u]++;
        }
    }
    void unionBySize(int u, int v)
    {
        int ulp_u = findUPar(u);
        int ulp_v = findUPar(v);

        if(ulp_u == ulp_v) return;

        if(size[ulp_u] < size[ulp_v])
        {
            parent[ulp_u] = ulp_v;
            size[ulp_u] += size[ulp_v];
        }
        else
        {
            parent[ulp_v] = ulp_u;
            size[ulp_v] += size[ulp_u];
        }
    }
};

//----------------------------------------- Binary Exponentiation -------------------------------------------------//

long long binpow(long long a, long long b) {
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a;
        a = a * a;
        b >>= 1;
    }
    return res;
}

//----------------------------------------- Binary Exponentiation with modulo -------------------------------------------------//

long long binpow(long long a, long long b, long long m) {
    a %= m;
    long long res = 1;
    while (b > 0) {
        if (b & 1)
            res = res * a % m;
        a = a * a % m;
        b >>= 1;
    }
    return res;
}

//----------------------------------------- MEX -------------------------------------------------//

//find the minimal non-negative element that is not present in the array 

int mex(vector<int> const& A) {
    static bool used[MAX_N+1] = { 0 };

    // mark the given numbers
    for (int x : A) {
        if (x <= MAX_N)
            used[x] = true;
    }

    // find the mex
    int result = 0;
    while (used[result])
        ++result;

    // clear the array again
    for (int x : A) {
        if (x <= MAX_N)
            used[x] = false;
    }

    return result;
}

void Solve() {
    // Add code to solve the problem here
}
