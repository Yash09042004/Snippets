
# Competitive Programming Templates

### Constants
```cpp
#define ll long long
#define pi 3.14159265358979323846
const ll MOD = 1e9 + 7;
const int MAXN = 1e6 + 5;
```

### Prime Checking, Factorization and Sieve 

```cpp
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

// Sieve of Eratosthenes(Set)
set<int> sieveOfEratosthenes(int lower, int upper) {
    if (upper < 2) return {};lower = max(lower, 2);
    vector<bool> isPrime(upper + 1, true);
    isPrime[0] = isPrime[1] = false;
    for (int p = 2; p * p <= upper; p++) {
        if (isPrime[p]) {
            for (int multiple = p * p; multiple <= upper; multiple += p) { isPrime[multiple] = false; }
        }
    }
    set<int> primeSet;
    for (int num = lower; num <= upper; num++) {
        if (isPrime[num]) {
            primeSet.insert(num);
        }
    }
    return primeSet;
}

// Prime Factorization
map<int, int> primeFactorization(int n, set<int>& primeSet) {
    std::map<int, int> factors;
    for (int prime : primeSet) {
        if (prime * prime > n) break;  
        while (n % prime == 0) {
            factors[prime]++;
            n /= prime;
        }
    }
    if (n > 1) {
        factors[n] = 1;
    }
    return factors;
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
```
### Range XOR 
```cpp
ll xor_upto(ll n) {
    if (n % 4 == 0) return n;
    if (n % 4 == 1) return 1;
    if (n % 4 == 2) return n + 1;
    return 0;
}

ll range_xor(ll l, ll r) {
    return xor_upto(r) ^ xor_upto(l - 1);
}
```
### Modular Arithmetic
```cpp
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
```

### Combinatorics and Factorials 
```cpp
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



// Factorial class to performs operations on Factorials without overflow 
class Factorial{
    map<int, int> primeFactorization(int n, set<int>& primeSet) {
        std::map<int, int> factors;
        for (int prime : primeSet) {
            if (prime * prime > n) break;  
            while (n % prime == 0) {
                factors[prime]++;
                n /= prime;
            }
        }
        if (n > 1) {
            factors[n] = 1;
        }
        return factors;
    }
int n;
map<int,int> primeFactors;
set<int> primeSet;
public:
    Factorial(int n, set<int> primeSet ){
        this->n = n;
        for( int i = 2 ; i<=n ; i++ ){
            map<int,int> currentFactors = primeFactorization(i,primeSet);
            for( auto it : currentFactors){
                primeFactors[it.first] += it.second;
            }
        }
    }
    Factorial( map<int, int> primeFactors ){
        this->primeFactors = primeFactors;
    }
    ll getValue(){
        ll result = 1;
        for( auto it : primeFactors ){
            result *= pow(it.first,it.second);
        }
        return result;
    }

    map<int,int> getMap(){
        return this->primeFactors;
    }

    ll operator+(Factorial m){
        return this->getValue()+m.getValue();
    }
    ll operator+(int m){
        return this->getValue()+m;
    }
    ll operator+(ll m){
        return this->getValue()+m;
    }
    ll operator-(Factorial m){
        return this->getValue()-m.getValue();
    }
    ll operator-(int m){
        return this->getValue()-m;
    }
    ll operator-(ll m){
        return this->getValue()-m;
    }
    Factorial operator*(Factorial m){
        map<int,int> ans;
        for( auto it : m.primeFactors ){
            ans[it.first] += it.second;
        }
        for( auto it : this->primeFactors ){
            ans[it.first] += it.second;
        }
        return ans;
    }
    Factorial operator/(Factorial m){
        map<int,int> ans;
        for( auto it : m.primeFactors ){
            ans[it.first] -= it.second;
        }
        for( auto it : this->primeFactors ){
            ans[it.first] += it.second;
        }
        return ans;
    }
};
```
### Subset generation
```cpp
vector<vector<int>> generateSubsets(const vector<int>& set) {
    int n = set.size();
    vector<vector<int>> superset;
    for (int i = 0; i < (1 << n); ++i) {
        vector<int> subset;
        for (int j = 0; j < n; ++j) {
            if (i & (1 << j)) {
                subset.push_back(set[j]);
            }
        }
        superset.push_back(subset);
    }
    return superset;
}

```

### Generate Permutations

```cpp
vector<vector<int>> generatePermutations(vector<int> set) {
    vector<vector<int>> allPermutations;
    sort(set.begin(), set.end()); // Sort only necessary for lexicographical order
    do {
        allPermutations.push_back(set);
    } while (next_permutation(set.begin(), set.end()));
    return allPermutations;
}
```


### Search Algorithms
```cpp
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
```
### Greedy Algorithms 
```cpp
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
```
### Graph Algorithms 

```cpp
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
```
### Big Number Multiplication 
```cpp
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

```
### Disjoint Set Union 

```cpp
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

``` 

### Binary Exponentiation 
```cpp

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

// -------------------------- With Modulo ----------------------

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
```
### MEX 

```cpp

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
```

### Max Subarray Sum (Kadane's Algorithm)

```cpp
int SubarrayMax(vector<int> arr)
{
    int best = 0, sum = 0;
    for (int k = 0; k < arr.size(); k++)
    {
        sum = max(arr[k], sum + arr[k]);
        best = max(best, sum);
    }
    return best;
}
```

### Sliding Window Maximum

```cpp
vector<int> slidingWindowMaximum(vector<int>& nums, int k) {
    vector<int> result;
    deque<int> dq; 
    for (int i = 0; i < nums.size(); ++i) {
        if (!dq.empty() && dq.front() <= i - k) {
            dq.pop_front();
        }
        while (!dq.empty() && nums[dq.back()] <= nums[i]) {
            dq.pop_back();
        }
        dq.push_back(i);
        if (i >= k - 1) {
            result.push_back(nums[dq.front()]);
        }
    }
    return result;
}
```


### Hamming Distance

```cpp
// For integers (Faster)
int hamming(int a, int b) {
    return __builtin_popcount(a^b);
}

// For Strings
int hamming(string a, string b) {
    int d = 0;
    for (int i = 0; i < a.size(); i++) {
        if (a[i] != b[i]) d++;
    }
    return d;
}


```