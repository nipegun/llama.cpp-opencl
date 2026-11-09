#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"
#include "log.h"

#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <random>
#include <optional>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <atomic>

namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    int32_t use_count;
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    RPC_CMD_MEMSET_TENSOR,
    RPC_CMD_SET_TENSOR_2D,
    RPC_CMD_GET_TENSOR_2D,
    RPC_CMD_COMM_INIT,
    RPC_CMD_COMM_ALLREDUCE,
    RPC_CMD_COMM_FREE,
    // llamacppOpenCL11: sent only to a server that announces them (rpc_server_feature): an older server does not know them
    RPC_CMD_SUPPORTS_OP,
    RPC_CMD_AUTH_NONCE,
    RPC_CMD_AUTH,
    RPC_CMD_NONE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

static const char * rpc_cmd_name(enum rpc_cmd cmd) {
    switch (cmd) {
        case RPC_CMD_ALLOC_BUFFER:      return "ALLOC_BUFFER";
        case RPC_CMD_GET_ALIGNMENT:     return "GET_ALIGNMENT";
        case RPC_CMD_GET_MAX_SIZE:      return "GET_MAX_SIZE";
        case RPC_CMD_BUFFER_GET_BASE:   return "BUFFER_GET_BASE";
        case RPC_CMD_FREE_BUFFER:       return "FREE_BUFFER";
        case RPC_CMD_BUFFER_CLEAR:      return "BUFFER_CLEAR";
        case RPC_CMD_SET_TENSOR:        return "SET_TENSOR";
        case RPC_CMD_SET_TENSOR_HASH:   return "SET_TENSOR_HASH";
        case RPC_CMD_GET_TENSOR:        return "GET_TENSOR";
        case RPC_CMD_COPY_TENSOR:       return "COPY_TENSOR";
        case RPC_CMD_GRAPH_COMPUTE:     return "GRAPH_COMPUTE";
        case RPC_CMD_GET_DEVICE_MEMORY: return "GET_DEVICE_MEMORY";
        case RPC_CMD_INIT_TENSOR:       return "INIT_TENSOR";
        case RPC_CMD_GET_ALLOC_SIZE:    return "GET_ALLOC_SIZE";
        case RPC_CMD_HELLO:             return "HELLO";
        case RPC_CMD_DEVICE_COUNT:      return "DEVICE_COUNT";
        case RPC_CMD_GRAPH_RECOMPUTE:   return "GRAPH_RECOMPUTE";
        case RPC_CMD_MEMSET_TENSOR:     return "MEMSET_TENSOR";
        case RPC_CMD_SET_TENSOR_2D:     return "SET_TENSOR_2D";
        case RPC_CMD_GET_TENSOR_2D:     return "GET_TENSOR_2D";
        case RPC_CMD_COMM_INIT:         return "COMM_INIT";
        case RPC_CMD_COMM_ALLREDUCE:    return "COMM_ALLREDUCE";
        case RPC_CMD_COMM_FREE:         return "COMM_FREE";
        case RPC_CMD_SUPPORTS_OP:       return "SUPPORTS_OP";
        case RPC_CMD_AUTH_NONCE:        return "AUTH_NONCE";
        case RPC_CMD_AUTH:              return "AUTH";
        case RPC_CMD_NONE:              return "NONE";
        default:                        return "UNKNOWN";
    }
}

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

// Maximum number of graphs cached per device; client and server must use the same value
// so that both sides clear their caches at the same point in the message stream
const size_t GRAPH_CACHE_MAX = 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

// llamacppOpenCL11: the features of a server, in the byte of the HELLO response that was padding (zero in older servers,
// and older clients ignore it), so that the protocol version stays the same
enum rpc_server_feature : uint8_t {
    RPC_FEATURE_SUPPORTS_OP = 1, // answers RPC_CMD_SUPPORTS_OP
    RPC_FEATURE_AUTH        = 2, // needs RPC_CMD_AUTH_NONCE and RPC_CMD_AUTH before any other command (GGML_RPC_TOKEN)
    RPC_FEATURE_COMM_DISABLED = 4, // operator has not enabled server-to-server networking
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t features; // rpc_server_feature
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

// RPC_CMD_AUTH_NONCE (no request): a random nonce; RPC_CMD_AUTH: HMAC-SHA256 of the nonce with the token, so the token
// does not travel over the network
struct rpc_msg_auth_nonce_rsp {
    uint8_t nonce[32];
};

struct rpc_msg_auth_req {
    uint8_t mac[32];
};

struct rpc_msg_auth_rsp {
    uint8_t ok;
};

// RPC_CMD_SUPPORTS_OP: | device (4 bytes) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |, the node
// first, then its sources and the tensors that they view (at most 2*(1 + GGML_MAX_SRC))
struct rpc_msg_supports_op_rsp {
    uint8_t supported;
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_memset_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
    uint64_t uid;
};

struct rpc_msg_get_tensor_2d_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
    uint64_t n_copies;
    uint64_t stride;
};

struct rpc_msg_comm_init_req {
    uint32_t device;
    uint32_t rank;
    uint32_t world;
    uint32_t port;      // rank 0: port to listen on; rank > 0: rank 0's comm port
    char     host[64];  // rank > 0: rank 0's host
};

struct rpc_msg_comm_init_rsp {
    uint8_t ok;
};

struct rpc_msg_comm_allreduce_req {
    uint32_t   device;
    rpc_tensor tensor;
};

struct rpc_msg_comm_free_req {
    uint32_t device;
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

class rpc_dispatcher;
struct ggml_backend_rpc_context {
    std::shared_ptr<rpc_dispatcher> dispatcher;
    std::string                     endpoint;
    uint32_t                        device;
    std::string                     name;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<rpc_dispatcher>   dispatcher;
    void                            * base_ptr;
    uint64_t                          remote_ptr;
};

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len, uint64_t hash = 0xcbf29ce484222325ULL) {
    const uint64_t fnv_prime = 0x100000001b3ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    if (!sock->send_data(msg, msg_size)) {
        return false;
    }
    return sock->flush();
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

// Control messages are bounded before allocating. Tensor uploads use a validated
// destination and fixed-size chunks instead of this helper.
static constexpr size_t cRpcMaxGraphFrame = 64u * 1024u * 1024u;
static constexpr size_t cRpcTransferChunk = 16u * 1024u * 1024u;
static constexpr size_t cRpcMaxSupportsFrame = 2*sizeof(uint32_t) + 2*(1 + GGML_MAX_SRC)*sizeof(rpc_tensor);

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input, size_t pMaxSize, size_t pMinSize) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size < pMinSize || size > pMaxSize || size > SIZE_MAX) {
      LOG_ERROR("RPC control frame size out of bounds: %" PRIu64 "\n", size);
      return false;
    }
    try {
        input.resize((size_t) size);
    } catch (const std::length_error &) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    } catch (const std::bad_alloc &) {
        LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

// ---- authentication (llamacppOpenCL11) ----
//
// With GGML_RPC_TOKEN set, the server requires it (RPC_FEATURE_AUTH): it sends a random nonce (RPC_CMD_AUTH_NONCE)
// and the client answers HMAC-SHA256(token, nonce) (RPC_CMD_AUTH), so the token does not travel over the network. The
// connection is not encrypted: it keeps out the clients that do not know the token, not someone who can read or change
// the traffic. SHA-256 is FIPS 180-4 and HMAC is RFC 2104.

static const uint32_t rpc_sha256_k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

struct rpc_sha256 {
    uint32_t h[8];
    uint8_t  buf[64];
    size_t   nbuf; // bytes in buf
    uint64_t len;  // bytes hashed
};

static uint32_t rpc_rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

static void rpc_sha256_block(uint32_t h[8], const uint8_t * p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i + 1] << 16 | (uint32_t) p[4*i + 2] << 8 | (uint32_t) p[4*i + 3];
    }
    for (int i = 16; i < 64; i++) {
        const uint32_t s0 = rpc_rotr(w[i - 15], 7) ^ rpc_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rpc_rotr(w[i - 2], 17) ^ rpc_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], k = h[7];
    for (int i = 0; i < 64; i++) {
        const uint32_t t1 = k + (rpc_rotr(e, 6) ^ rpc_rotr(e, 11) ^ rpc_rotr(e, 25)) + ((e & f) ^ (~e & g)) + rpc_sha256_k[i] + w[i];
        const uint32_t t2 = (rpc_rotr(a, 2) ^ rpc_rotr(a, 13) ^ rpc_rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        k = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h[0] += a;
    h[1] += b;
    h[2] += c;
    h[3] += d;
    h[4] += e;
    h[5] += f;
    h[6] += g;
    h[7] += k;
}

static void rpc_sha256_init(rpc_sha256 & s) {
    static const uint32_t h0[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    memcpy(s.h, h0, sizeof(h0));
    s.nbuf = 0;
    s.len  = 0;
}

static void rpc_sha256_update(rpc_sha256 & s, const uint8_t * data, size_t n) {
    s.len += n;
    while (n > 0) {
        const size_t c = std::min(n, sizeof(s.buf) - s.nbuf);
        memcpy(s.buf + s.nbuf, data, c);
        s.nbuf += c;
        data   += c;
        n      -= c;
        if (s.nbuf == sizeof(s.buf)) {
            rpc_sha256_block(s.h, s.buf);
            s.nbuf = 0;
        }
    }
}

static void rpc_sha256_final(rpc_sha256 & s, uint8_t out[32]) {
    const uint64_t bits = s.len * 8;
    const uint8_t  one  = 0x80;
    const uint8_t  zero = 0;
    rpc_sha256_update(s, &one, 1);
    while (s.nbuf != 56) {
        rpc_sha256_update(s, &zero, 1);
    }
    uint8_t lenb[8];
    for (int i = 0; i < 8; i++) {
        lenb[i] = (uint8_t) (bits >> (56 - 8*i));
    }
    rpc_sha256_update(s, lenb, sizeof(lenb));
    for (int i = 0; i < 8; i++) {
        out[4*i]     = (uint8_t) (s.h[i] >> 24);
        out[4*i + 1] = (uint8_t) (s.h[i] >> 16);
        out[4*i + 2] = (uint8_t) (s.h[i] >> 8);
        out[4*i + 3] = (uint8_t) s.h[i];
    }
}

static void rpc_hmac_sha256(const uint8_t * key, size_t key_len, const uint8_t * msg, size_t msg_len, uint8_t out[32]) {
    uint8_t    k[64] = {};
    rpc_sha256 s;
    if (key_len > sizeof(k)) {
        rpc_sha256_init(s);
        rpc_sha256_update(s, key, key_len);
        rpc_sha256_final(s, k);
    } else {
        memcpy(k, key, key_len);
    }
    uint8_t pad[64];
    uint8_t inner[32];
    for (size_t i = 0; i < sizeof(k); i++) {
        pad[i] = k[i] ^ 0x36;
    }
    rpc_sha256_init(s);
    rpc_sha256_update(s, pad, sizeof(pad));
    rpc_sha256_update(s, msg, msg_len);
    rpc_sha256_final(s, inner);
    for (size_t i = 0; i < sizeof(k); i++) {
        pad[i] = k[i] ^ 0x5c;
    }
    rpc_sha256_init(s);
    rpc_sha256_update(s, pad, sizeof(pad));
    rpc_sha256_update(s, inner, sizeof(inner));
    rpc_sha256_final(s, out);
}

// the token of GGML_RPC_TOKEN (empty: no authentication)
static std::string rpc_auth_token() {
    const char * token = getenv("GGML_RPC_TOKEN");
    return token != nullptr ? std::string(token) : std::string();
}

// who answers a nonce, the first byte of the message of the HMAC: the answer of one role is no use in another one (a
// server that connects to a false rank 0 does not give out an answer that authenticates a client)
enum rpc_auth_role : uint8_t {
    RPC_AUTH_CLIENT     = 1, // a client to its server (RPC_CMD_AUTH)
    RPC_AUTH_COMM_PEER  = 2, // a server to rank 0 of a communicator (rpc_comm_auth)
    RPC_AUTH_COMM_RANK0 = 3, // rank 0 of a communicator to the server (rpc_comm_auth)
};

static void rpc_auth_mac(const std::string & token, rpc_auth_role role, const uint8_t nonce[32], uint8_t mac[32]) {
    uint8_t msg[1 + 32];
    msg[0] = role;
    memcpy(msg + 1, nonce, 32);
    rpc_hmac_sha256((const uint8_t *) token.data(), token.size(), msg, sizeof(msg), mac);
}

// compares two HMAC in a time that does not depend on where they differ
static bool rpc_auth_mac_equal(const uint8_t a[32], const uint8_t b[32]) {
    uint8_t diff = 0;
    for (size_t i = 0; i < 32; i++) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

static bool rpc_random_bytes(uint8_t * out, size_t n) {
    try {
        std::random_device rd;
        for (size_t i = 0; i < n; i += 4) {
            const uint32_t r = rd();
            for (size_t j = 0; j < 4 && i + j < n; j++) {
                out[i + j] = (uint8_t) (r >> (8*j));
            }
        }
    } catch (const std::exception & e) {
        LOG_ERROR("[%s] no random device: %s\n", __func__, e.what());
        return false;
    }
    return true;
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// A communicator is disabled unless the operator supplies its rank-zero endpoint.
// Only canonical IPv4 literals are accepted, so no request-triggered DNS lookup
// can redirect a connection. The same setting is used on every participating server.
struct rpc_comm_config {
  std::string host;
  uint32_t port = 0;
};

static bool fRpcCommConfig(rpc_comm_config & pConfig) {
  const char * vEndpoint = std::getenv("GGML_RPC_COMM_ENDPOINT");
  if (vEndpoint == nullptr || *vEndpoint == '\0') {
    return true;
  }
  int vPort = 0;
  if (!parse_endpoint(vEndpoint, pConfig.host, vPort) || vPort < 1 || vPort > 65535) {
    return false;
  }
  size_t vStart = 0;
  std::string vCanonical;
  try {
    for (int vIndex = 0; vIndex < 4; ++vIndex) {
      const size_t vEnd = pConfig.host.find('.', vStart);
      if ((vIndex < 3) != (vEnd != std::string::npos)) {
        return false;
      }
      const std::string vPart = pConfig.host.substr(vStart, vEnd - vStart);
      size_t vUsed = 0;
      const int vOctet = std::stoi(vPart, &vUsed);
      if (vUsed != vPart.size() || vOctet < 0 || vOctet > 255 ||
          (vIndex == 0 && (vOctet == 0 || vOctet >= 224))) {
        return false;
      }
      vCanonical += (vIndex == 0 ? "" : ".") + std::to_string(vOctet);
      vStart = vEnd == std::string::npos ? pConfig.host.size() : vEnd + 1;
    }
  } catch (const std::exception &) {
    return false;
  }
  if (vCanonical + ":" + std::to_string(vPort) != vEndpoint) {
    return false;
  }
  pConfig.port = (uint32_t) vPort;
  return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return sock->flush();
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// with busy spinning on, the dispatcher still blocks on its queue after this long without commands
static constexpr auto RPC_BUSY_SPIN_IDLE_TIME = std::chrono::milliseconds(100);

static inline void rpc_cpu_relax() {
#if defined(__aarch64__) && (defined(__clang__) || defined(__GNUC__))
    __asm__ volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
// features: the rpc_server_feature of the server (llamacppOpenCL11)
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock, uint8_t & features) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    features = response.features;
    LOG_DBG("[%s] handshake ok: server version %d.%d.%d, features: %u, transport: %s\n",
            __func__, response.major, response.minor, response.patch, response.features, sock->transport_name());
    return true;
}

// the client side of the authentication (see rpc_auth_token): false if the server rejects the token or there is none
static bool rpc_authenticate(const std::shared_ptr<socket_t> & sock, const std::string & endpoint) {
    const std::string token = rpc_auth_token();
    if (token.empty()) {
        LOG_ERROR("the RPC server %s requires a token: set GGML_RPC_TOKEN to the token of the server\n", endpoint.c_str());
        return false;
    }
    rpc_msg_auth_nonce_rsp nonce;
    if (!send_rpc_cmd(sock, RPC_CMD_AUTH_NONCE, nullptr, 0, &nonce, sizeof(nonce))) {
        return false;
    }
    rpc_msg_auth_req request;
    rpc_auth_mac(token, RPC_AUTH_CLIENT, nonce.nonce, request.mac);
    rpc_msg_auth_rsp response = {};
    if (!send_rpc_cmd(sock, RPC_CMD_AUTH, &request, sizeof(request), &response, sizeof(response)) || response.ok != 1) {
        LOG_ERROR("the RPC server %s rejected the token of GGML_RPC_TOKEN\n", endpoint.c_str());
        return false;
    }
    return true;
}

template <typename T>
class message_queue {
public:
    message_queue() {}

    bool push(const T &value) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted) {
            return false;
        }
        queue.push(value);
        cvar.notify_all();
        return true;
    }

    bool pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        cvar.wait(lock, [this] { return !queue.empty() || interrupted; });
        if (interrupted) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    bool try_pop(T* out) {
        std::unique_lock<std::mutex> lock(mutex);
        if (interrupted || queue.empty()) {
            return false;
        }
        *out = queue.front();
        queue.pop();
        return true;
    }

    void interrupt() {
        std::unique_lock<std::mutex> lock(mutex);
        interrupted = true;
        lock.unlock();
        cvar.notify_all();
    }

private:
    bool interrupted = false;
    std::queue<T> queue;
    std::mutex mutex;
    std::condition_variable cvar;
};

class rpc_dispatcher {
public:
    rpc_dispatcher() {
    }

    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size);
    void send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size);

    ggml_backend_event_t event_new(ggml_backend_dev_t dev);
    void event_free(ggml_backend_event_t event);
    void event_synchronize(ggml_backend_event_t event);
    void event_record(ggml_backend_event_t event);
    void synchronize();
    void busy_spin_acquire();
    void busy_spin_release();
    void graph_compute(uint32_t device, const ggml_cgraph * cgraph);

    void start(const std::string & endpoint);
    void work();

    // the rpc_server_feature of the server (llamacppOpenCL11)
    uint8_t features() const {
        return server_features;
    }

    ~rpc_dispatcher();

private:
    struct rpc_msg {
        rpc_cmd                       cmd;
        std::shared_ptr<const void>   input;
        size_t                        input_size;
        void                        * output;
        size_t                        output_size;
        std::promise<void>            completion;
    };
    using rpc_msg_ptr   = std::shared_ptr<rpc_msg>;
    using rpc_msg_queue = message_queue<rpc_msg_ptr>;
    struct rpc_event {
        rpc_msg_ptr              msg;
        std::shared_future<void> sf;
    };
    std::mutex graph_mutex;
    std::unordered_map<uint32_t, std::unordered_set<uint64_t>> graph_uids;
    rpc_msg_queue    queue;
    socket_ptr       sock;
    uint8_t          server_features = 0;
    std::atomic_uint busy_spin_users = 0;
    std::atomic_bool running;
    std::thread      thread;
};

static void rpc_dispatcher_trampoline(rpc_dispatcher * dispatcher)
{
    dispatcher->work();
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = nullptr;
    msg->output_size = 0;
    GGML_ASSERT(queue.push(msg));
}

void rpc_dispatcher::send(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
    auto future = msg->completion.get_future();
    future.wait();
}

void rpc_dispatcher::send_async(enum rpc_cmd cmd, std::shared_ptr<const void> input, size_t input_size, void * output, size_t output_size) {
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = cmd;
    msg->input = input;
    msg->input_size = input_size;
    msg->output = output;
    msg->output_size = output_size;
    GGML_ASSERT(queue.push(msg));
}

ggml_backend_event_t rpc_dispatcher::event_new(ggml_backend_dev_t dev) {
    rpc_event * ev = new rpc_event;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ ev,
    };
}

void rpc_dispatcher::event_free(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    delete ev;
}

void rpc_dispatcher::event_synchronize(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->sf.wait();
}

void rpc_dispatcher::event_record(ggml_backend_event_t event) {
    rpc_event * ev = (rpc_event *)event->context;
    ev->msg = std::make_shared<rpc_msg>();
    ev->msg->cmd = RPC_CMD_NONE;
    ev->sf = ev->msg->completion.get_future().share();
    GGML_ASSERT(queue.push(ev->msg));
}

void rpc_dispatcher::synchronize() {
    // to ensure all messages are processed, submit dummy message and wait for it to complete
    auto msg = std::make_shared<rpc_msg>();
    msg->cmd = RPC_CMD_NONE;
    GGML_ASSERT(queue.push(msg));
    msg->completion.get_future().wait();
}

void rpc_dispatcher::busy_spin_acquire() {
    busy_spin_users.fetch_add(1, std::memory_order_relaxed);
}

void rpc_dispatcher::busy_spin_release() {
    const unsigned previous = busy_spin_users.fetch_sub(1, std::memory_order_relaxed);
    GGML_ASSERT(previous > 0);
}

void rpc_dispatcher::start(const std::string & endpoint) {
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_ABORT("Failed to parse endpoint: %s\n", endpoint.c_str());
    }
    if (!rpc_transport_init()) {
        GGML_ABORT("RPC transport initialization failed\n");
    }

    sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        GGML_ABORT("Failed to connect to %s\n", endpoint.c_str());
    }
    if (!negotiate_hello(sock, server_features)) {
        GGML_ABORT("RPC handshake failed for %s\n", endpoint.c_str());
    }
    if ((server_features & RPC_FEATURE_AUTH) != 0 && !rpc_authenticate(sock, endpoint)) {
        GGML_ABORT("RPC authentication failed for %s\n", endpoint.c_str());
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    running = true;
    thread = std::thread(rpc_dispatcher_trampoline, this);
}

void rpc_dispatcher::work() {
    auto last_cmd = std::chrono::steady_clock::now();
    while (running) {
        rpc_msg_ptr msg_ptr;
        // spin only while commands keep coming, so an idle dispatcher does not keep a core busy
        const bool spin = busy_spin_users.load(std::memory_order_relaxed) != 0 &&
                          std::chrono::steady_clock::now() - last_cmd < RPC_BUSY_SPIN_IDLE_TIME;
        if (spin) {
            if (!queue.try_pop(&msg_ptr)) {
                rpc_cpu_relax();
                continue;
            }
        } else if (!queue.pop(&msg_ptr)) {
            break;
        }
        if (msg_ptr->cmd != RPC_CMD_NONE) {
            LOG_DBG2("[%s] %s (in: %s, out: %s)\n", __func__, rpc_cmd_name(msg_ptr->cmd),
                     rpc_log_format_size(msg_ptr->input_size).c_str(), rpc_log_format_size(msg_ptr->output_size).c_str());
            const bool trace = rpc_log_debug_level() >= 3;
            const auto t0 = std::chrono::steady_clock::now();
            if (msg_ptr->output) {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size, msg_ptr->output, msg_ptr->output_size);
                RPC_STATUS_ASSERT(status);
            } else {
                bool status = send_rpc_cmd(sock, msg_ptr->cmd, msg_ptr->input.get(), msg_ptr->input_size);
                RPC_STATUS_ASSERT(status);
            }
            if (trace) {
                const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - t0;
                LOG_DBG3("[%s] %s done in %.3f ms\n", __func__, rpc_cmd_name(msg_ptr->cmd), elapsed.count());
            }
        } else {
            LOG_DBG3("[%s] sync barrier\n", __func__);
        }
        msg_ptr->completion.set_value();
        last_cmd = std::chrono::steady_clock::now();
    }
}

rpc_dispatcher::~rpc_dispatcher() {
    running = false;
    queue.interrupt();
    sock = nullptr;
    if (thread.joinable()) {
        thread.join();
    }
}

static std::shared_ptr<rpc_dispatcher> get_dispatcher(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<rpc_dispatcher>> dispatchers;

    auto it = dispatchers.find(endpoint);
    if (it != dispatchers.end()) {
        if (auto dispatcher = it->second.lock()) {
            return dispatcher;
        }
    }

    auto dispatcher = std::make_shared<rpc_dispatcher>();
    dispatcher->start(endpoint);
    dispatchers[endpoint] = dispatcher;
    return dispatcher;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_free_buffer_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request->remote_ptr);
    ctx->dispatcher->send(RPC_CMD_FREE_BUFFER, request, sizeof(*request));
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    auto request = std::make_shared<rpc_msg_buffer_get_base_req>();
    request->remote_ptr = ctx->remote_ptr;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, ctx->remote_ptr);
    rpc_msg_buffer_get_base_rsp response;
    ctx->dispatcher->send(RPC_CMD_BUFFER_GET_BASE, request, sizeof(*request), &response, sizeof(response));
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor, const rpc_dispatcher * dispatcher = nullptr) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        // ref: https://github.com/ggml-org/llama.cpp/pull/26500
        if (ctx != nullptr && (dispatcher == nullptr || ctx->dispatcher.get() == dispatcher)) {
            result.buffer = ctx->remote_ptr;
            result.data = reinterpret_cast<uint64_t>(tensor->data);
        } else {
            result.buffer = 0;
            result.data = 0;
        }
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    result.use_count = 0;

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        LOG_DBG2("[%s] tensor: %s\n", __func__, rpc_log_format_tensor(tensor).c_str());
        auto request = std::make_shared<rpc_msg_init_tensor_req>();
        request->tensor = serialize_tensor(tensor);
        ctx->dispatcher->send(RPC_CMD_INIT_TENSOR, request, sizeof(*request));
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_memset_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size   = size;
    request->value  = value;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, value: %u\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str(), value);
    ctx->dispatcher->send(RPC_CMD_MEMSET_TENSOR, request, sizeof(*request));
}

// input serialization format: | rpc_tensor | cache_flag (1 byte) | offset (8 bytes) | data (size bytes)
static std::shared_ptr<uint8_t> serialize_set_tensor(const rpc_tensor & rpc_tensor, uint8_t cache_flag, uint64_t offset, const void * data, size_t size, size_t & input_size) {
    input_size = sizeof(rpc_tensor) + sizeof(cache_flag) + sizeof(offset) + size;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * p = input;
    memcpy(p, &rpc_tensor, sizeof(rpc_tensor)); p += sizeof(rpc_tensor);
    memcpy(p, &cache_flag, sizeof(cache_flag)); p += sizeof(cache_flag);
    memcpy(p, &offset,     sizeof(offset));     p += sizeof(offset);
    memcpy(p, data, size);
    return std::shared_ptr<uint8_t>(input, std::default_delete<uint8_t[]>());
}

// the hash cache is meant for weights, so that a model reload can skip re-sending them.
// compute-buffer inputs (the activations ggml_backend_sched copies between backends) must not
// take this path, otherwise with `rpc-server -c` every ubatch above the threshold is written
// to the cache directory and later served from there.
static bool rpc_use_hash_cache(const ggml_tensor * tensor, size_t size) {
    return size > HASH_THRESHOLD && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    // input serialization format: | rpc_tensor | offset (8 bytes) | size (8 bytes) | n_copies (8 bytes) | stride (8 bytes) | data (size * n_copies bytes) |
    size_t input_size = sizeof(rpc_tensor) + 4*sizeof(uint64_t) + size*n_copies;
    uint8_t * input = new uint8_t[input_size]();
    uint8_t * dest = input;
    memcpy(dest, &rpc_tensor, sizeof(rpc_tensor));
    dest += sizeof(rpc_tensor);
    uint64_t header[4] = { offset, size, n_copies, stride_tensor };
    memcpy(dest, header, sizeof(header));
    dest += sizeof(header);
    for (size_t i = 0; i < n_copies; i++) {
        memcpy(dest + i*size, (const char *)data + i*stride_data, size);
    }
    std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
    ctx->dispatcher->send(RPC_CMD_SET_TENSOR_2D, input_ptr, input_size);
}

static void ggml_backend_rpc_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s, n_copies: %zu, stride: %zu\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), offset, rpc_log_format_size(size).c_str(), n_copies, stride_tensor);
    auto request = std::make_shared<rpc_msg_get_tensor_2d_req>();
    request->tensor   = serialize_tensor(tensor);
    request->offset   = offset;
    request->size     = size;
    request->n_copies = n_copies;
    request->stride   = stride_tensor;
    if (stride_data == size) {
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), data, size*n_copies);
    } else {
        std::vector<uint8_t> packed(size*n_copies);
        ctx->dispatcher->send(RPC_CMD_GET_TENSOR_2D, request, sizeof(*request), packed.data(), packed.size());
        for (size_t i = 0; i < n_copies; i++) {
            memcpy((char *)data + i*stride_data, packed.data() + i*size, size);
        }
    }
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->dispatcher != dst_ctx->dispatcher) {
            LOG_DBG2("[%s] src and dst are on different servers, falling back to a host copy\n", __func__);
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());
        auto request = std::make_shared<rpc_msg_copy_tensor_req>();
        request->src = serialize_tensor(src);
        request->dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        ctx->dispatcher->send(RPC_CMD_COPY_TENSOR, request, sizeof(*request), &response, sizeof(response));
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    auto request = std::make_shared<rpc_msg_buffer_clear_req>();
    request->remote_ptr = ctx->remote_ptr;
    request->value = value;
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request->remote_ptr, request->value);
    ctx->dispatcher->send(RPC_CMD_BUFFER_CLEAR, request, sizeof(*request));
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_rpc_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_rpc_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_rpc_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
    /* .set_tensor_n    = */ NULL,
    /* .get_tensor_n    = */ NULL,
    /* .cpy_tensor_n    = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    auto request = std::make_shared<rpc_msg_alloc_buffer_req>();
    request->device = buft_ctx->device;
    request->size = size;
    rpc_msg_alloc_buffer_rsp response;

    auto dispatcher = get_dispatcher(buft_ctx->endpoint);
    dispatcher->send(RPC_CMD_ALLOC_BUFFER, request, sizeof(*request), &response, sizeof(response));
    if (response.remote_ptr != 0) {
        LOG_DBG("[%s] endpoint: %s, device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str(),
                response.remote_ptr, rpc_log_format_size(response.remote_size).c_str());
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{dispatcher, nullptr, response.remote_ptr},
            response.remote_size);
        return buffer;
    } else {
        LOG_WARN("[%s] endpoint: %s, device: %u, size: %s -> failed\n",
                __func__, buft_ctx->endpoint.c_str(), buft_ctx->device, rpc_log_format_size(request->size).c_str());
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_alignment_req>();
    request->device = device;
    rpc_msg_get_alignment_rsp response;
    dispatcher->send(RPC_CMD_GET_ALIGNMENT, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, device, rpc_log_format_size(response.alignment).c_str());
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<rpc_dispatcher> & dispatcher, uint32_t device) {
    auto request = std::make_shared<rpc_msg_get_max_size_req>();
    request->device = device;
    rpc_msg_get_max_size_rsp response;
    dispatcher->send(RPC_CMD_GET_MAX_SIZE, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, device, rpc_log_format_size(response.max_size).c_str());
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // [TAG_ALLOC_SIZE_EXPAND]
    // ops that may require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= ggml_op_alloc_size_may_expand(tensor->op);

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;

        // the reported size must never be below ggml_nbytes: rpc_tensor stores nb[] as uint32_t,
        // so a stride over 4 GiB is truncated on the wire and the remote size comes back too small
        // TODO: change rpc_tensor nb to 64-bit int
        const size_t min_size = ggml_nbytes(tensor);

        // Cache key for calls to read the alloc_size.
        // We deliberately exclude src tensor dimensions from the key because:
        // 1. For CPU backends, alloc_size = ggml_nbytes(output) regardless of src shapes
        // 2. For GPU backends, the reservation graph uses max dimensions, so the
        //    cached value from reservation is always >= any subsequent request
        // 3. Including src dims causes cache misses per-ubatch (e.g. growing KV cache)
        //    which blocks the main thread behind in-flight GRAPH_COMPUTE commands
        struct alloc_size_cache_key {
            uint32_t device;
            uint32_t type;
            uint32_t op;
            int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
            uint32_t ne[GGML_MAX_DIMS];
            uint64_t nb[GGML_MAX_DIMS];
        };

        alloc_size_cache_key key = {};
        key.device = buft_ctx->device;
        key.type = tensor->type;
        key.op = tensor->op;
        memcpy(key.op_params, tensor->op_params, sizeof(key.op_params));
        for (int i = 0; i < GGML_MAX_DIMS; i++) {
            key.ne[i] = (uint32_t)tensor->ne[i];
            key.nb[i] = (uint64_t)tensor->nb[i];
        }

        uint64_t cache_hash = fnv_hash((const uint8_t *)&key, sizeof(key));
        cache_hash = fnv_hash((const uint8_t *)buft_ctx->endpoint.data(), buft_ctx->endpoint.size(), cache_hash);

        // alloc sizes are immutable for a given tensor configuration
        static std::mutex cache_mutex;
        static std::unordered_map<uint64_t, size_t> cache;

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            auto it = cache.find(cache_hash);
            if (it != cache.end()) {
                LOG_DBG3("[%s] cache hit: %s -> %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                         rpc_log_format_size(it->second).c_str());
                return std::max<size_t>(it->second, min_size);
            }
        }

        auto request = std::make_shared<rpc_msg_get_alloc_size_req>();
        request->device = buft_ctx->device;
        request->tensor = serialize_tensor(tensor);

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request->srcs[i] = serialize_tensor(tensor->src[i]);
        }

        LOG_DBG2("[%s] cache miss: %s, querying server\n", __func__, rpc_log_format_tensor(tensor).c_str());
        rpc_msg_get_alloc_size_rsp response;
        auto dispatcher = get_dispatcher(buft_ctx->endpoint);
        dispatcher->send(RPC_CMD_GET_ALLOC_SIZE, request, sizeof(*request), &response, sizeof(response));

        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache[cache_hash] = response.alloc_size;
        }

        LOG_DBG2("[%s] %s -> alloc_size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 rpc_log_format_size(response.alloc_size).c_str());

        return std::max<size_t>(response.alloc_size, min_size);
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name            = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer        = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .alloc_buffer_n      = */ NULL,
    /* .get_alignment       = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size      = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .get_alloc_size_n    = */ NULL,
    /* .is_host             = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    uint8_t cache_flag = 0;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    if (rpc_use_hash_cache(tensor, size)) {
        auto request = std::make_shared<rpc_msg_set_tensor_hash_req>();
        request->tensor = rpc_tensor;
        request->offset = offset;
        request->hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        // TODO: make this async
        ctx->dispatcher->send(RPC_CMD_SET_TENSOR_HASH, request, sizeof(*request), &response, sizeof(response));
        LOG_DBG2("[%s] tensor: %s, hash: 0x%" PRIx64 ", cache: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(),
                 request->hash, response.result ? "hit" : "miss");
        if (response.result) {
            // the server has the same data, no need to send it
            return;
        }
        // the server has no cache entry for this tensor - ask it to save one
        cache_flag = 1;
    }
    size_t input_size;
    auto input = serialize_set_tensor(rpc_tensor, cache_flag, offset, data, size, input_size);
    ctx->dispatcher->send_async(RPC_CMD_SET_TENSOR, input, input_size);
}

static void ggml_backend_rpc_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_context * ctx = (ggml_backend_rpc_context *)backend->context;
    LOG_DBG2("[%s] tensor: %s, offset: %zu, size: %s\n", __func__, rpc_log_format_tensor(tensor).c_str(), offset,
             rpc_log_format_size(size).c_str());
    auto request = std::make_shared<rpc_msg_get_tensor_req>();
    request->tensor = serialize_tensor(tensor);
    request->offset = offset;
    request->size = size;
    ctx->dispatcher->send_async(RPC_CMD_GET_TENSOR, request, sizeof(*request), data, size);
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->synchronize();
}

static void add_tensor(ggml_tensor * tensor, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], cgraph, dispatcher, tensors, visited);
    }
    add_tensor(tensor->view_src, cgraph, dispatcher, tensors, visited);
    rpc_tensor result = serialize_tensor(tensor, dispatcher);
    const size_t hash_pos = ggml_hash_find(&cgraph->visited_hash_set, tensor);
    if (hash_pos != GGML_HASHSET_FULL && ggml_bitset_get(cgraph->visited_hash_set.used, hash_pos)) {
        result.use_count = cgraph->use_counts[hash_pos];
    }
    tensors.push_back(result);
}

static uint8_t * serialize_graph(uint32_t device, const ggml_cgraph * cgraph, const rpc_dispatcher * dispatcher, size_t * output_size) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], cgraph, dispatcher, tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    *output_size = 2*sizeof(uint32_t) + sizeof(uint64_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    uint8_t * output = new uint8_t[*output_size]();
    uint8_t * dest = output;
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &cgraph->uid, sizeof(cgraph->uid));
    dest += sizeof(cgraph->uid);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
    return output;
}

void rpc_dispatcher::graph_compute(uint32_t device, const ggml_cgraph * cgraph) {
    std::lock_guard<std::mutex> lock(graph_mutex);
    GGML_ASSERT(cgraph->n_nodes > 0);
    auto & device_graph_uids = graph_uids[device];
    bool reuse = cgraph->uid != 0 && device_graph_uids.count(cgraph->uid) > 0;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, reuse: %s\n", __func__, device, cgraph->uid, cgraph->n_nodes, reuse ? "yes" : "no");
    if (reuse) {
        auto request = std::make_shared<rpc_msg_graph_recompute_req>();
        request->device = device;
        request->uid    = cgraph->uid;
        send_async(RPC_CMD_GRAPH_RECOMPUTE, request, sizeof(*request));
    } else {
        if (cgraph->uid != 0) {
            if (device_graph_uids.size() >= GRAPH_CACHE_MAX) {
                LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                        __func__, device, device_graph_uids.size());
                device_graph_uids.clear();
            }
            device_graph_uids.insert(cgraph->uid);
        }
        size_t input_size = 0;
        uint8_t * input = serialize_graph(device, cgraph, this, &input_size);
        LOG_DBG2("[%s] serialized graph: %s\n", __func__, rpc_log_format_size(input_size).c_str());
        std::shared_ptr<uint8_t> input_ptr(input, std::default_delete<uint8_t[]>());
        send_async(RPC_CMD_GRAPH_COMPUTE, input_ptr, input_size);
    }
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->graph_compute(rpc_ctx->device, cgraph);
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_rpc_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    rpc_ctx->dispatcher->event_record(event);
}

static void ggml_backend_rpc_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    // this is noop for RPC as we have a single stream
    GGML_UNUSED(backend);
    GGML_UNUSED(event);
}

static void ggml_backend_rpc_set_tensor_2d_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static void ggml_backend_rpc_get_tensor_2d_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_tensor_get_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);
    GGML_UNUSED(backend);
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ ggml_backend_rpc_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_rpc_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_rpc_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_rpc_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ ggml_backend_rpc_event_record,
    /* .event_wait              = */ ggml_backend_rpc_event_wait,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto dispatcher = get_dispatcher(endpoint);
    size_t alignment = get_alignment(dispatcher, device);
    size_t max_size = get_max_size(dispatcher, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    LOG_DBG("[%s] created %s (alignment: %s, max_size: %s)\n", __func__, buft_name.c_str(),
            rpc_log_format_size(alignment).c_str(), rpc_log_format_size(max_size).c_str());
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    auto dispatcher = get_dispatcher(endpoint);
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .dispatcher = */ dispatcher,
        /* .endpoint   = */ endpoint,
        /* .device     = */ device,
        /* .name       = */ dev_name,
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    LOG_DBG("[%s] %s\n", __func__, dev_name.c_str());
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto dispatcher = get_dispatcher(endpoint);
    auto request = std::make_shared<rpc_msg_get_device_memory_req>();
    request->device = device;
    rpc_msg_get_device_memory_rsp response;
    dispatcher->send(RPC_CMD_GET_DEVICE_MEMORY, request, sizeof(*request), &response, sizeof(response));
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, device,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    *free = response.free_mem;
    *total = response.total_mem;
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir, std::string auth_token, rpc_comm_config pCommConfig)
        : backends(std::move(all_backends)), cache_dir(cache_dir), auth_token(std::move(auth_token)), vCommConfig(std::move(pCommConfig)) {
        stored_graphs.resize(backends.size());
        comm_states.resize(backends.size());
    }
    ~rpc_server();

    // static (llamacppOpenCL11): the handshake of a connection runs before its rpc_server exists (rpc_serve_handshake)
    static void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool memset_tensor(const rpc_msg_memset_tensor_req & request);
    bool set_tensor(const socket_ptr & pSock);
    bool set_tensor_2d(const socket_ptr & pSock);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool supports_op(const std::vector<uint8_t> & input, rpc_msg_supports_op_rsp & response);
    bool comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response);
    bool comm_allreduce(const rpc_msg_comm_allreduce_req & request);
    bool comm_free(const rpc_msg_comm_free_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

private:
    void sync_all_backends();
    void comm_free_device(uint32_t dev_id);
    static bool comm_allreduce_pairwise(void * comm_ctx_v, ggml_tensor ** tensors);
    static bool comm_bootstrap_exchange(void * user_data, const void * send_buf, void * recv_buf, size_t size);
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);


    // Per-device state for one communicator.
    struct comm_state {
        // Sockets to the other servers. Rank 0 holds one per peer in rank order, so peers[i] is
        // rank i + 1. Every other rank holds one to rank 0. The pairwise path reduces over peers[0].
        std::vector<socket_ptr> peers;
        uint32_t                rank = 0;
        uint32_t                world = 0;
        ggml_backend_t          pairwise_backend = nullptr;
        ggml_backend_buffer_ptr scratch;
        size_t                  scratch_size = 0;
        std::vector<uint8_t>    send_buf;
        std::vector<uint8_t>    recv_buf;
        std::vector<float>      host_f32; // the partials in f32 when the bf16 casts run on the host (llamacppOpenCL11)

        // The backend's communicator. Null on the pairwise path, which has none.
        void *                              backend_comm_ctx = nullptr;

        // The chosen reduction, set once a path is agreed. Receives backend_comm_ctx, or
        // comm_state on the pairwise path. A communicator covers one device, so the caller
        // passes a one-element tensor array.
        ggml_backend_comm_allreduce_tensor_t allreduce_fn    = nullptr;
    };

    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    // GGML_RPC_TOKEN of the server (empty: none), also for the peers of a communicator (rpc_comm_auth)
    std::string auth_token;
    rpc_comm_config vCommConfig;
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // computed graphs cached per backend, keyed by uid
    std::vector<std::unordered_map<uint64_t, stored_graph>> stored_graphs;
    std::vector<comm_state> comm_states;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG2("[%s] device: %u, tensor: %s, buffer: %p, data: %p\n", __func__, dev_id, rpc_log_format_tensor(tensor).c_str(),
             (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        // llamacppOpenCL11: neither the CPU (posix_memalign) nor OpenCL (clCreateBuffer) clear a new buffer, and the
        // server can reuse the memory that an earlier client freed (its weights, prompts, KV cache): a client that read
        // a new buffer (GET_TENSOR) before writing it got the data of another one. Cleared before the client gets it
        ggml_backend_buffer_clear(buffer, 0);
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %u, size: %s -> remote_ptr: 0x%" PRIx64 ", remote_size: %s\n",
                __func__, dev_id, rpc_log_format_size(request.size).c_str(), response.remote_ptr,
                rpc_log_format_size(response.remote_size).c_str());
        buffers.insert(buffer);
    } else {
        LOG_WARN("[%s] device: %u, size: %s -> failed\n", __func__, dev_id, rpc_log_format_size(request.size).c_str());
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG2("[%s] device: %u, alignment: %s\n", __func__, dev_id, rpc_log_format_size(alignment).c_str());
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG2("[%s] device: %u, max_size: %s\n", __func__, dev_id, rpc_log_format_size(max_size).c_str());
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    // Discard all cached graphs to avoid use-after-free in graph_recompute,
    // since their nodes may hold pointers to the buffer being freed. The client
    // will have to re-serialize its graphs from now on.
    for (auto & sgs : stored_graphs) {
        for (auto & sg : sgs) {
            sg.second.graph = nullptr;
        }
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    sync_all_backends();
    LOG_DBG2("[%s] remote_ptr: 0x%" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

bool rpc_server::memset_tensor(const rpc_msg_memset_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    const uint64_t tensor_size = ggml_nbytes(tensor);
    if (request.offset > tensor_size || request.size > tensor_size - request.offset) {
        LOG_ERROR("[%s] tensor region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds [0, %" PRIu64 ")\n",
                       __func__, request.offset, request.size, tensor_size);
        return false;
    }

    const uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
    const uint64_t buffer_size = ggml_backend_buffer_get_size(tensor->buffer);
    if (request.tensor.data < buffer_start) {
        LOG_ERROR("[%s] tensor data before buffer start\n", __func__);
        return false;
    }
    const uint64_t data_offset = request.tensor.data - buffer_start;
    if (data_offset > buffer_size ||
        request.offset > buffer_size - data_offset ||
        request.size > buffer_size - data_offset - request.offset) {
        LOG_ERROR("[%s] tensor region out of buffer bounds\n", __func__);
        return false;
    }
    if (tensor->buffer->iface.memset_tensor == nullptr) {
        LOG_ERROR("[%s] memset not implemented by backend buffer\n", __func__);
        return false;
    }

    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, value: %u\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void *) tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.value);
    ggml_backend_tensor_memset(tensor, request.value, request.offset, request.size);
    return true;
}

// true if the dimensions and strides sent by a client describe a layout ggml can create, so that the bounds checked
// in deserialize_tensor and in the commands (with ggml_nelements and ggml_nbytes) cover the bytes the backends access:
// - a type with blocks (the quantized types) has whole blocks in a row and nb[0] equal to the size of a block: the
//   backends access a row as ne[0]/blck_size whole blocks, while ggml_nbytes counts ne[0]*nb[0]/blck_size bytes
//   (with nb[0] = nb[1] = 1, a Q6_K tensor of ne[0] = 512 counted 2 bytes, and the q6r layout of the OpenCL
//   backend converted 420 bytes into host buffers of 1 byte)
// - the number of elements fits in an int64_t and ggml_nbytes in 64 bits: with values near 2^32, ggml_nelements
//   wrapped around (and the bounds were not checked), and so did the sum of ggml_nbytes
// The type is already valid (type < GGML_TYPE_COUNT, blck_size > 0).
static bool rpc_tensor_layout_ok(const rpc_tensor * tensor) {
    const ggml_type type      = (ggml_type) tensor->type;
    const uint64_t  blck_size = ggml_blck_size(type);
    const uint64_t  type_size = ggml_type_size(type);
    if (blck_size > 1 && (tensor->ne[0] % blck_size != 0 || tensor->nb[0] != type_size)) {
        return false;
    }

    uint64_t n_elements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        const uint64_t ne = tensor->ne[i];
        if (ne != 0 && n_elements > (uint64_t) INT64_MAX / ne) {
            return false;
        }
        n_elements *= ne;
    }
    if (n_elements == 0) {
        return true; // no data: ggml_nbytes is 0
    }

    // ggml_nbytes: the bytes of a row, plus (ne[i] - 1)*nb[i] for the other dimensions (all of them with blck_size 1)
    uint64_t nbytes = blck_size > 1 ? (uint64_t) tensor->ne[0] / blck_size * type_size : type_size;
    for (int i = blck_size > 1 ? 1 : 0; i < GGML_MAX_DIMS; i++) {
        const uint64_t ne = (uint64_t) tensor->ne[i] - 1;
        const uint64_t nb = tensor->nb[i];
        if (nb != 0 && ne > (UINT64_MAX - nbytes) / nb) {
            return false;
        }
        nbytes += ne*nb;
    }
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    // the dimensions and strides come from the client: reject the layouts whose bounds ggml_nbytes cannot check
    if (!rpc_tensor_layout_ok(tensor)) {
        LOG_ERROR("[%s] invalid layout received for type %s: ne [%u, %u, %u, %u], nb [%u, %u, %u, %u]\n",
                  __func__, ggml_type_name((ggml_type) tensor->type),
                  tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
                  tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer && ggml_nelements(result) > 0) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        if (tensor->data + tensor_size < tensor->data ||
            tensor->data < buffer_start || tensor->data + tensor_size > buffer_start + buffer_size) {
            LOG_ERROR("[%s] tensor '%s' (op %s, type %s, ne [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]) "
                      "data [0x%" PRIx64 ", 0x%" PRIx64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, tensor->name, ggml_op_name((ggml_op) tensor->op), ggml_type_name(result->type),
                      result->ne[0], result->ne[1], result->ne[2], result->ne[3],
                      tensor->data, tensor->data + tensor_size, buffer_start, buffer_start + buffer_size);
            return nullptr;
        }
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}

// llamacppOpenCL11: true if the bytes [offset, offset + size) are inside the tensor (its ggml_nbytes). The commands
// check them against the buffer, but ggml_backend_tensor_set and ggml_backend_tensor_get assert that they are inside
// the tensor (and that its data is not null): a client that sent more bytes, or a tensor without elements (whose data
// deserialize_tensor does not check), aborted the server. A client always meets it (the same asserts on its side)
static bool rpc_tensor_range_ok(const ggml_tensor * tensor, uint64_t offset, uint64_t size) {
    const uint64_t nbytes = ggml_nbytes(tensor);
    return offset <= nbytes && size <= nbytes - offset;
}


// The tensor and its complete write span are validated before any payload is read.
static bool fRpcTensorWriteRange(const ggml_tensor * pTensor, uint64_t pOffset, uint64_t pSize) {
  if (!rpc_tensor_range_ok(pTensor, pOffset, pSize)) {
    return false;
  }
  if (pSize == 0) {
    return true;
  }
  const uint64_t vBase = (uint64_t) ggml_backend_buffer_get_base(pTensor->buffer);
  const uint64_t vBufferSize = ggml_backend_buffer_get_size(pTensor->buffer);
  const uint64_t vData = (uint64_t) pTensor->data;
  if (vData < vBase || vData - vBase > vBufferSize) {
    return false;
  }
  const uint64_t vAvailable = vBufferSize - (vData - vBase);
  return pOffset <= vAvailable && pSize <= vAvailable - pOffset;
}

bool rpc_server::set_tensor(const socket_ptr & pSock) {
  // Wire layout is unchanged: tensor | cache flag | offset | payload.
  const uint64_t cHeaderSize = sizeof(rpc_tensor) + sizeof(uint8_t) + sizeof(uint64_t);
  uint64_t vFrameSize = 0;
  rpc_tensor vWireTensor;
  uint8_t vCacheFlag = 0;
  uint64_t vOffset = 0;
  if (!pSock->recv_data(&vFrameSize, sizeof(vFrameSize)) || vFrameSize < cHeaderSize || vFrameSize > SIZE_MAX ||
      !pSock->recv_data(&vWireTensor, sizeof(vWireTensor)) || !pSock->recv_data(&vCacheFlag, sizeof(vCacheFlag)) ||
      !pSock->recv_data(&vOffset, sizeof(vOffset))) {
    return false;
  }
  const size_t vSize = (size_t) (vFrameSize - cHeaderSize);
  sync_all_backends();
  ggml_context_ptr vContext { ggml_init({ggml_tensor_overhead(), nullptr, true}) };
  if (!vContext) {
    return false;
  }
  ggml_tensor * vTensor = deserialize_tensor(vContext.get(), &vWireTensor);
  if (vTensor == nullptr || vTensor->buffer == nullptr || !fRpcTensorWriteRange(vTensor, vOffset, vSize)) {
    LOG_ERROR("[%s] invalid tensor upload range\n", __func__);
    return false;
  }
  std::vector<uint8_t> aChunk(std::min(vSize, cRpcTransferChunk));
  // Keep the existing content-addressed cache without retaining the upload in RAM.
  // tmpfile is private and automatically removed, also on a truncated upload.
  auto fCloseFile = [](FILE * pFile) { std::fclose(pFile); };
  std::unique_ptr<FILE, decltype(fCloseFile)> vCache(
    cache_dir && vCacheFlag ? std::tmpfile() : nullptr, fCloseFile);
  uint64_t vHash = 0xcbf29ce484222325ULL;
  for (size_t vDone = 0; vDone < vSize;) {
    const size_t vChunkSize = std::min(aChunk.size(), vSize - vDone);
    if (!pSock->recv_data(aChunk.data(), vChunkSize)) {
      return false;
    }
    ggml_backend_tensor_set(vTensor, aChunk.data(), (size_t) vOffset + vDone, vChunkSize);
    if (vCache) {
      vHash = fnv_hash(aChunk.data(), vChunkSize, vHash);
      if (std::fwrite(aChunk.data(), 1, vChunkSize, vCache.get()) != vChunkSize) {
        vCache.reset(); // a cache failure must not invalidate the tensor upload
      }
    }
    vDone += vChunkSize;
  }
  if (vCache && std::fseek(vCache.get(), 0, SEEK_SET) == 0) {
    char aHash[17];
    snprintf(aHash, sizeof(aHash), "%016" PRIx64, vHash);
    const fs::path vPath = fs::path(cache_dir) / aHash;
    std::ofstream vOutput(vPath, std::ios::binary);
    bool vComplete = vOutput.good();
    for (size_t vDone = 0; vComplete && vDone < vSize;) {
      const size_t vChunkSize = std::min(aChunk.size(), vSize - vDone);
      vComplete = std::fread(aChunk.data(), 1, vChunkSize, vCache.get()) == vChunkSize;
      if (vComplete) {
        vOutput.write((const char *) aChunk.data(), vChunkSize);
        vComplete = vOutput.good();
      }
      vDone += vChunkSize;
    }
    vOutput.close();
    if (!vComplete || vOutput.fail()) {
      std::error_code vError;
      fs::remove(vPath, vError);
    }
  }
  return true;
}

bool rpc_server::set_tensor_2d(const socket_ptr & pSock) {
  const uint64_t cHeaderSize = sizeof(rpc_tensor) + 4*sizeof(uint64_t);
  uint64_t vFrameSize = 0;
  rpc_tensor vWireTensor;
  uint64_t aHeader[4];
  if (!pSock->recv_data(&vFrameSize, sizeof(vFrameSize)) || vFrameSize < cHeaderSize || vFrameSize > SIZE_MAX ||
      !pSock->recv_data(&vWireTensor, sizeof(vWireTensor)) || !pSock->recv_data(aHeader, sizeof(aHeader))) {
    return false;
  }
  const uint64_t vOffset = aHeader[0], vSize = aHeader[1], vCopies = aHeader[2], vStride = aHeader[3];
  const uint64_t vDataSize = vFrameSize - cHeaderSize;
  if (vCopies == 0 || vSize == 0 || vSize > vDataSize / vCopies || vSize*vCopies != vDataSize ||
      (vStride != 0 && vCopies - 1 > (UINT64_MAX - vSize) / vStride)) {
    return false;
  }
  const uint64_t vSpan = (vCopies - 1)*vStride + vSize;
  sync_all_backends();
  ggml_context_ptr vContext { ggml_init({ggml_tensor_overhead(), nullptr, true}) };
  if (!vContext) {
    return false;
  }
  ggml_tensor * vTensor = deserialize_tensor(vContext.get(), &vWireTensor);
  if (vTensor == nullptr || vTensor->buffer == nullptr || !fRpcTensorWriteRange(vTensor, vOffset, vSpan)) {
    LOG_ERROR("[%s] invalid 2D tensor upload range\n", __func__);
    return false;
  }
  std::vector<uint8_t> aChunk((size_t) std::min<uint64_t>(vSize, cRpcTransferChunk));
  for (uint64_t vRow = 0; vRow < vCopies; ++vRow) {
    for (uint64_t vDone = 0; vDone < vSize;) {
      const size_t vChunkSize = (size_t) std::min<uint64_t>(aChunk.size(), vSize - vDone);
      if (!pSock->recv_data(aChunk.data(), vChunkSize)) {
        return false;
      }
      ggml_backend_tensor_set(vTensor, aChunk.data(), (size_t) (vOffset + vRow*vStride + vDone), vChunkSize);
      vDone += vChunkSize;
    }
  }
  return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }

    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        LOG_DBG2("[%s] tensor: %s, offset: %" PRIu64 ", hash: 0x%" PRIx64 ", cache miss\n",
                __func__, rpc_log_format_tensor(tensor).c_str(), request.offset, request.hash);
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, hash: 0x%" PRIx64 ", cache hit\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(size).c_str(), request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    if (!rpc_tensor_range_ok(tensor, request.offset, size)) {
        LOG_ERROR("[%s] tensor write region (offset=%" PRIu64 ", size=%zu) out of tensor bounds (%zu)\n",
                  __func__, request.offset, size, ggml_nbytes(tensor));
        return false;
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p\n", __func__, rpc_log_format_tensor(tensor).c_str(),
            (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s\n", __func__,
             rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
             rpc_log_format_size(request.size).c_str());

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }
    if (!rpc_tensor_range_ok(tensor, request.offset, request.size)) {
        LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", size=%" PRIu64 ") out of tensor bounds (%zu)\n",
                  __func__, request.offset, request.size, ggml_nbytes(tensor));
        return false;
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::get_tensor_2d(const rpc_msg_get_tensor_2d_req & request, std::vector<uint8_t> & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG2("[%s] tensor: %s, buffer: %p, data: %p, offset: %" PRIu64 ", size: %s, n_copies: %" PRIu64 ", stride: %" PRIu64 "\n",
            __func__, rpc_log_format_tensor(tensor).c_str(), (void*)tensor->buffer, tensor->data, request.offset,
            rpc_log_format_size(request.size).c_str(), request.n_copies, request.stride);

    // sanitize tensor->data
    {
        if (request.n_copies == 0 || request.size == 0 || request.size > UINT64_MAX / request.n_copies) {
            return false;
        }
        if (request.stride != 0 && request.n_copies - 1 > (UINT64_MAX - request.size) / request.stride) {
            return false;
        }
        const uint64_t span = (request.n_copies - 1)*request.stride + request.size;
        const uint64_t p0 = (uint64_t) ggml_backend_buffer_get_base(tensor->buffer);
        const uint64_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data < p0 || request.tensor.data > p1 || request.offset > p1 - request.tensor.data ||
                span > p1 - request.tensor.data - request.offset) {
            LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", span=%" PRIu64 ") out of buffer bounds [0x%" PRIx64 ", 0x%" PRIx64 ")\n",
                      __func__, request.tensor.data, request.offset, span, p0, p1);
            return false;
        }
        if (request.offset > ggml_nbytes(tensor) || span > ggml_nbytes(tensor) - request.offset) {
            LOG_ERROR("[%s] tensor read region (offset=%" PRIu64 ", span=%" PRIu64 ") out of tensor bounds (%zu)\n",
                      __func__, request.offset, span, ggml_nbytes(tensor));
            return false;
        }
    }

    response.resize(request.size * request.n_copies, 0);
    ggml_backend_tensor_get_2d(tensor, response.data(), request.offset, request.size, request.n_copies, request.stride, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    sync_all_backends();
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    // llamacppOpenCL11: the backends copy ggml_nbytes(src) bytes to the data of dst (the CPU with memcpy), and
    // deserialize_tensor does not check the data of a tensor without elements: an empty dst with any address and a
    // src with elements wrote the bytes of the client at that address of the server (dst_data + src_size only had to
    // stay below the end of the buffer, so any address below the buffer was valid, and the sum could wrap around). A
    // client only copies tensors with the same layout (ggml_backend_tensor_copy asserts it), so the bytes of src are
    // the bytes of dst, which deserialize_tensor checked against its buffer; and an empty copy has nothing to copy
    if (!ggml_are_same_layout(src, dst)) {
        LOG_ERROR("[%s] src and dst have different layouts\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (src_size == 0) {
        response.result = 1;
        return true;
    }

    // [dst_data, dst_data + src_size) inside the buffer of dst, without sums that wrap around
    if (dst_data < dst_base || dst_data - dst_base > dst_buf_sz || src_size > dst_buf_sz - (dst_data - dst_base)) {
        LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }
    // llamacppOpenCL11: and, in the same buffer, src and dst apart: OpenCL fails with CL_MEM_COPY_OVERLAP, and CL_CHECK
    // ends the server (memcpy is undefined). The buffers of OpenCL share their addresses (cl_ptr_base), so only the
    // same buffer counts
    if (src->buffer == dst->buffer) {
        const uint64_t src_data = (uint64_t) src->data;
        if (src_data < dst_data + src_size && dst_data < src_data + src_size) {
            LOG_ERROR("[%s] src and dst overlap\n", __func__);
            return false;
        }
    }

    LOG_DBG2("[%s] src: %s -> dst: %s\n", __func__, rpc_log_format_tensor(src).c_str(), rpc_log_format_tensor(dst).c_str());

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

// true if a backend can access the data of t, a node of a graph that it computes or a source of that node: t is in a
// buffer of this connection (deserialize_tensor checked that its data are inside), or it has no elements; and a view
// is in the buffer of the tensor it views (the OpenCL backend takes the cl_mem from view_src->buffer and the offset
// from t->data, which deserialize_tensor checked against t->buffer)
static bool rpc_graph_tensor_ok(const ggml_tensor * t) {
    if (t->buffer == nullptr && ggml_nelements(t) > 0) {
        return false;
    }
    return t->view_src == nullptr || t->view_src->buffer == t->buffer;
}

// ---- the operations of a graph (server: graph_compute; client: ggml_backend_rpc_device_supports_op) ----
//
// A backend computes a node with the dimensions, strides, types and op_params of its tensors as they come, and relies
// on what the ggml functions that build each operation guarantee (ggml.c: ggml_add_impl, ggml_mul_mat, ...): the
// shape and type of the result, the shapes of the sources that a kernel indexes with the dimensions of another
// tensor, the rows it reads or writes in one piece, the work buffer of the CPU (ggml_graph_plan sizes it with the
// dimensions of one tensor, the kernel uses those of another). A client sends all of them, and a malicious one made
// the kernels read and write outside their tensors (and the CPU backend call a function of the client: the custom
// operations carry one in op_params). rpc_node_ok checks what those functions assert and what the CPU and OpenCL
// kernels need, for the operations that the OpenCL backend of this fork runs (ggml_cl_be_supports_op) and the views;
// any other operation fails. The server rejects a graph with a node that fails, and the client does not send them:
// the scheduler runs them on another backend. To run another operation through RPC, add its checks here.

// d > 0 and d divides n (the broadcasts of ggml)
static bool rpc_divides(int64_t d, int64_t n) {
    return d > 0 && n % d == 0;
}

static bool rpc_ne_eq(const ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
    return t->ne[0] == ne0 && t->ne[1] == ne1 && t->ne[2] == ne2 && t->ne[3] == ne3;
}

// t has the type and the dimensions of s (ggml_dup_tensor or ggml_view_tensor of s)
static bool rpc_same_type_shape(const ggml_tensor * t, const ggml_tensor * s) {
    return t->type == s->type && ggml_are_same_shape(t, s);
}

// t has the type, the dimensions and the strides of v (t is a view of v: the result of ggml_cpy and ggml_set_rows)
static bool rpc_same_layout(const ggml_tensor * t, const ggml_tensor * v) {
    if (t->type != v->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] != v->ne[i] || t->nb[i] != v->nb[i]) {
            return false;
        }
    }
    return true;
}

// the rows of t (dimensions 1 to 3) follow each other at the stride of dimension 1 (top_k of the CPU walks them so)
static bool rpc_rows_flat(const ggml_tensor * t) {
    size_t next = t->nb[1]*t->ne[1];
    for (int i = 2; i < GGML_MAX_DIMS; i++) {
        if (t->ne[i] != 1 && t->nb[i] != next) {
            return false;
        }
        next *= t->ne[i];
    }
    return true;
}

// every tensor of a node: fewer than 2^31 values (the kernels count rows and values with int); a type with blocks as
// ggml creates it (whole blocks in a row, nb[0] the size of a block: rpc_tensor_layout_ok, for the tensors of a
// client); and, with one value per block, a stride of at least one value along dimension 0: the kernels read and
// write the ne[0] values of a row in one piece, which then stay within the bytes that ggml_nbytes counts
static bool rpc_tensor_access_ok(const ggml_tensor * t) {
    const int64_t blck = ggml_blck_size(t->type);
    if (ggml_nelements(t) > INT32_MAX) {
        return false;
    }
    if (blck > 1) {
        return t->ne[0] % blck == 0 && t->nb[0] == ggml_type_size(t->type);
    }
    return t->ne[0] <= 1 || t->nb[0] >= ggml_type_size(t->type);
}

// true if the node t can run in a server (see above)
static bool rpc_node_ok(const ggml_tensor * t) {
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    const ggml_tensor * s2 = t->src[2];

    switch (t->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true; // nothing is computed
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_SCALE:
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_CONCAT:
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_ARGMAX:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_SET_ROWS:
        case GGML_OP_GET_ROWS:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_MUL_MAT:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_TOP_K:
            break;
        default:
            return false; // also the custom operations and the unknown values of op
    }

    if (!rpc_tensor_access_ok(t)) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i] != nullptr && !rpc_tensor_access_ok(t->src[i])) {
            return false;
        }
    }

    switch (t->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            // ggml_add_impl, ggml_mul_impl; binary_op of the CPU asserts the rows of the result and of src0
            return s0 && s1 && ggml_can_repeat(s1, s0) && rpc_same_type_shape(t, s0) &&
                   t->nb[0] == ggml_type_size(t->type) && s0->nb[0] == ggml_type_size(s0->type);
        case GGML_OP_SCALE:
            // ggml_scale_impl
            return s0 && rpc_same_type_shape(t, s0) && ggml_is_contiguous_rows(s0) && ggml_is_contiguous_rows(t);
        case GGML_OP_UNARY:
            // ggml_unary_impl
            return s0 && (uint32_t) ggml_get_unary_op(t) < GGML_UNARY_OP_COUNT && rpc_same_type_shape(t, s0) &&
                   ggml_is_contiguous_rows(s0) && ggml_is_contiguous_rows(t);
        case GGML_OP_GLU:
            // ggml_glu_impl: with src1 (split), the result has the shape of src0; without it, half of the values of a row
            if (!s0 || (uint32_t) ggml_get_glu_op(t) >= GGML_GLU_OP_COUNT || t->type != s0->type ||
                !ggml_is_contiguous_1(s0) || !ggml_is_contiguous_1(t)) {
                return false;
            }
            if (s1) {
                return ggml_is_contiguous_1(s1) && rpc_same_type_shape(s1, s0) && ggml_are_same_shape(t, s0);
            }
            return rpc_ne_eq(t, s0->ne[0]/2, s0->ne[1], s0->ne[2], s0->ne[3]);
        case GGML_OP_CONCAT: {
            // ggml_concat; concat of the CPU writes whole rows of the result
            const int32_t dim = ggml_get_op_params_i32(t, 0);
            if (!s0 || !s1 || dim < 0 || dim >= GGML_MAX_DIMS || s1->type != s0->type || t->type != s0->type ||
                t->nb[0] != ggml_type_size(t->type)) {
                return false;
            }
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if ((d != dim && s1->ne[d] != s0->ne[d]) || t->ne[d] != (d == dim ? s0->ne[d] + s1->ne[d] : s0->ne[d])) {
                    return false;
                }
            }
            return true;
        }
        case GGML_OP_SSM_CONV: {
            // ggml_ssm_conv: sx {d_conv - 1 + n_t, d_inner, n_s}, c {d_conv, d_inner}, result {d_inner, n_t, n_s};
            // ssm_conv of the CPU reads them as F32 with whole rows of sx
            if (!s0 || !s1 || !ggml_is_3d(s0) || !ggml_is_matrix(s1) || s0->type != GGML_TYPE_F32 ||
                s1->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32) {
                return false;
            }
            const int64_t n_t = s0->ne[0] - s1->ne[0] + 1;
            return n_t >= 0 && s0->ne[1] == s1->ne[1] && rpc_ne_eq(t, s1->ne[1], n_t, s0->ne[2], 1) &&
                   s0->nb[0] == sizeof(float) && s1->nb[0] == sizeof(float) && s0->nb[1] == s0->ne[0]*sizeof(float);
        }
        case GGML_OP_GATED_DELTA_NET: {
            // ggml_gated_delta_net(q, k, v, g, beta, state, K) and its kernel of the CPU, which reads S_v values of the
            // rows of q and k, the gate and beta of every head, token and sequence of v, and writes the result (the
            // outputs, then K states per sequence) as one block
            const ggml_tensor * q     = t->src[0];
            const ggml_tensor * k     = t->src[1];
            const ggml_tensor * v     = t->src[2];
            const ggml_tensor * g     = t->src[3];
            const ggml_tensor * beta  = t->src[4];
            const ggml_tensor * state = t->src[5];
            if (!q || !k || !v || !g || !beta || !state || ggml_nelements(v) == 0) {
                return false;
            }
            for (const ggml_tensor * x : { q, k, v, g, beta, state, t }) {
                if (x->type != GGML_TYPE_F32) {
                    return false;
                }
            }
            const int64_t S_v = v->ne[0];
            const int64_t H   = v->ne[1];
            const int64_t T   = v->ne[2];
            const int64_t S   = v->ne[3];
            const int64_t K   = ggml_get_op_params_i32(t, 0);
            if (K < 1 || !ggml_is_contiguous_rows(q) || !ggml_is_contiguous_rows(k) || !ggml_is_contiguous_rows(v) ||
                !ggml_is_contiguous(g) || !ggml_is_contiguous(beta) || !ggml_is_contiguous(state) || !ggml_is_contiguous(t) ||
                q->ne[0] != S_v || k->ne[0] != S_v || q->ne[1] < 1 || k->ne[1] < 1 || q->ne[2] != T || k->ne[2] != T ||
                !rpc_divides(q->ne[3], S) || !rpc_divides(k->ne[3], S) ||
                (g->ne[0] != 1 && g->ne[0] != S_v) || g->ne[1] < H || g->ne[2] < T || g->ne[3] < S ||
                beta->ne[0] != 1 || beta->ne[1] < H || beta->ne[2] < T || beta->ne[3] < S ||
                !rpc_ne_eq(state, S_v, S_v, H, S)) {
                return false;
            }
            // the result is {S_v*H, T*S + K*S_v*S} (v has fewer than 2^31 values, so no product overflows)
            if (t->ne[0] != S_v*H || t->ne[2] != 1 || t->ne[3] != 1 || t->ne[1] < T*S) {
                return false;
            }
            return (t->ne[1] - T*S) % (S_v*S) == 0 && (t->ne[1] - T*S) / (S_v*S) == K;
        }
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            // ggml_rms_norm_impl, ggml_l2_norm_impl (eps >= 0: the CPU asserts it)
            return s0 && rpc_same_type_shape(t, s0) && ggml_get_op_params_f32(t, 0) >= 0.0f;
        case GGML_OP_ARGMAX:
            // ggml_argmax: a matrix, result I32 {ne1}
            return s0 && ggml_is_matrix(s0) && s0->type == GGML_TYPE_F32 && t->type == GGML_TYPE_I32 &&
                   rpc_ne_eq(t, s0->ne[1], 1, 1, 1);
        case GGML_OP_PAD: {
            // ggml_pad_ext: the left and right padding of every dimension in op_params 0..7, circular in 8; pad of the
            // CPU (F32) writes the result as one block, and wraps around src0 if circular
            if (!s0 || s0->type != GGML_TYPE_F32 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
                return false;
            }
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                const int64_t lp = ggml_get_op_params_i32(t, 2*i);
                const int64_t rp = ggml_get_op_params_i32(t, 2*i + 1);
                if (t->ne[i] != s0->ne[i] + lp + rp) {
                    return false;
                }
            }
            return ggml_get_op_params_i32(t, 8) == 0 || ggml_nelements(s0) > 0;
        }
        case GGML_OP_ARANGE: {
            // ggml_arange: a vector of ceil((stop - start)/step) values, which the CPU asserts and writes as one block
            const float start = ggml_get_op_params_f32(t, 0);
            const float stop  = ggml_get_op_params_f32(t, 1);
            const float step  = ggml_get_op_params_f32(t, 2);
            if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || !rpc_ne_eq(t, t->ne[0], 1, 1, 1) ||
                !std::isfinite(start) || !std::isfinite(stop) || !(stop > start) || !(step > 0.0f)) {
                return false;
            }
            const float steps = ceilf((stop - start) / step);
            return std::isfinite(steps) && steps <= (float) INT32_MAX && (int64_t) steps == t->ne[0];
        }
        case GGML_OP_CPY:
            // ggml_cpy_impl (the result is a view of src1) and ggml_cast (src1 is the result): the copies of the CPU walk
            // the destination with src1, and those to a quantized type keep ne[0] values of src0 in the work buffer,
            // which ggml_graph_plan sizes with ne[0] of the result
            return s0 && s1 && ggml_nelements(s0) == ggml_nelements(t) && (s1 == t || rpc_same_layout(t, s1)) &&
                   (!ggml_is_quantized(t->type) || s0->ne[0] <= t->ne[0]);
        case GGML_OP_DUP:
            // ggml_dup_impl
            return s0 && rpc_same_type_shape(t, s0);
        case GGML_OP_CONT:
            // ggml_cont_impl, ggml_cont_4d
            return s0 && t->type == s0->type && ggml_nelements(t) == ggml_nelements(s0);
        case GGML_OP_SET_ROWS:
            // ggml_set_rows(a = src2, b = src0, c = src1): the result is a view of a
            return s0 && s1 && t->ne[0] == s0->ne[0] && t->ne[2] == s0->ne[2] && t->ne[3] == s0->ne[3] &&
                   s0->ne[1] == s1->ne[0] && rpc_divides(s1->ne[1], s0->ne[2]) && rpc_divides(s1->ne[2], s0->ne[3]) &&
                   s1->ne[3] == 1 && (s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16) &&
                   (s1->type == GGML_TYPE_I64 || s1->type == GGML_TYPE_I32) &&
                   ggml_is_contiguous_rows(t) && ggml_is_contiguous_rows(s0) && (s2 == nullptr || rpc_same_layout(t, s2));
        case GGML_OP_GET_ROWS:
            // ggml_get_rows(a = src0, b = src1)
            return s0 && s1 && s0->ne[2] == s1->ne[1] && s0->ne[3] == s1->ne[2] && s1->ne[3] == 1 &&
                   s1->type == GGML_TYPE_I32 && t->type == (s0->type == GGML_TYPE_I32 ? GGML_TYPE_I32 : GGML_TYPE_F32) &&
                   rpc_ne_eq(t, s0->ne[0], s1->ne[0], s1->ne[1], s1->ne[2]);
        case GGML_OP_SOFT_MAX: {
            // ggml_soft_max_impl (mask: src1), ggml_soft_max_add_sinks (src2)
            const float max_bias = ggml_get_op_params_f32(t, 1);
            if (!s0 || !ggml_is_contiguous(s0) || !rpc_same_type_shape(t, s0) || (max_bias > 0.0f && !s1)) {
                return false;
            }
            if (s1 && ((s1->type != GGML_TYPE_F16 && s1->type != GGML_TYPE_F32) || !ggml_is_contiguous(s1) ||
                       s1->ne[0] != s0->ne[0] || s1->ne[1] < s0->ne[1] || !rpc_divides(s1->ne[2], s0->ne[2]) ||
                       !rpc_divides(s1->ne[3], s0->ne[3]))) {
                return false;
            }
            return s2 == nullptr || (s2->type == GGML_TYPE_F32 && s2->ne[0] == s0->ne[2]);
        }
        case GGML_OP_ROPE: {
            // ggml_rope_impl (positions: src1, frequency factors: src2); rope of the CPU asserts n_dims and n_offs, and
            // the OpenCL kernels need them too
            const int32_t n_dims = ggml_get_op_params_i32(t, 1);
            const int32_t mode   = ggml_get_op_params_i32(t, 2);
            const int32_t n_offs = ggml_get_op_params_i32(t, 15);
            if (!s0 || !s1 || (mode & 1) != 0 || !ggml_is_vector(s1) || s1->type != GGML_TYPE_I32 ||
                s1->ne[0] != s0->ne[2]*((mode & GGML_ROPE_TYPE_MROPE) ? 4 : 1) || !rpc_same_type_shape(t, s0) ||
                n_dims < 0 || n_dims % 2 != 0 || n_offs < 0 || n_offs % 2 != 0 || (int64_t) n_offs + n_dims > s0->ne[0]) {
                return false;
            }
            return s2 == nullptr || (s2->type == GGML_TYPE_F32 && s2->ne[0] >= n_dims/2);
        }
        case GGML_OP_MUL_MAT:
            // ggml_mul_mat: ggml_can_mul_mat(a, b), a not transposed, result F32 {a->ne1, b->ne1, b->ne2, b->ne3}
            return s0 && s1 && s0->ne[0] == s1->ne[0] && rpc_divides(s0->ne[2], s1->ne[2]) &&
                   rpc_divides(s0->ne[3], s1->ne[3]) && !ggml_is_transposed(s0) && t->type == GGML_TYPE_F32 &&
                   rpc_ne_eq(t, s0->ne[1], s1->ne[1], s1->ne[2], s1->ne[3]);
        case GGML_OP_FLASH_ATTN_EXT: {
            // ggml_flash_attn_ext_impl and ggml_flash_attn_ext_add_sinks (src4), without the K/V rows of a slice
            // (src5: the CPU reads K and V at row numbers of the client without an upper bound, and the OpenCL backend
            // does not run it). The kernels also read V and the mask over the K/V rows of k and the rows of q, which the
            // constructor does not check, and write the result as one block
            const ggml_tensor * q     = s0;
            const ggml_tensor * k     = s1;
            const ggml_tensor * v     = s2;
            const ggml_tensor * mask  = t->src[3];
            const ggml_tensor * sinks = t->src[4];
            const float max_bias = ggml_get_op_params_f32(t, 1);
            if (!q || !k || !v || t->src[5] != nullptr || k->ne[0] != q->ne[0] || !rpc_divides(k->ne[2], q->ne[2]) ||
                !rpc_divides(k->ne[3], q->ne[3]) || v->ne[3] != q->ne[3] || k->ne[3] != q->ne[3] || v->ne[1] != k->ne[1] ||
                !rpc_divides(v->ne[2], q->ne[2]) || (max_bias > 0.0f && !mask) || t->type != GGML_TYPE_F32 ||
                !ggml_is_contiguous(t) || !rpc_ne_eq(t, v->ne[0], q->ne[2], q->ne[1], q->ne[3])) {
                return false;
            }
            if (mask && (mask->type != GGML_TYPE_F16 || !ggml_is_contiguous(mask) || !rpc_divides(mask->ne[2], q->ne[2]) ||
                         !rpc_divides(mask->ne[3], q->ne[3]) || mask->ne[0] < k->ne[1] || mask->ne[1] < q->ne[1])) {
                return false;
            }
            return sinks == nullptr || (sinks->type == GGML_TYPE_F32 && sinks->ne[0] == q->ne[2]);
        }
        case GGML_OP_TOP_K:
            // ggml_top_k; top_k of the CPU walks the rows of src0 and of the result with the stride of dimension 1
            return s0 && t->type == GGML_TYPE_I32 && t->ne[0] <= s0->ne[0] &&
                   rpc_ne_eq(t, t->ne[0], s0->ne[1], s0->ne[2], s0->ne[3]) && rpc_rows_flat(s0) && rpc_rows_flat(t);
        default:
            return false;
    }
}

// the source and result types of a binary_op of the CPU (ggml-cpu/binary-ops.cpp: ADD and MUL; rpc_node_ok makes the
// result of the type of src0)
static bool rpc_cpu_binary_types_ok(ggml_type t0, ggml_type t1) {
    return (t0 == GGML_TYPE_F32  && t1 == GGML_TYPE_F32)  ||
           (t0 == GGML_TYPE_F16  && (t1 == GGML_TYPE_F16 || t1 == GGML_TYPE_F32)) ||
           (t0 == GGML_TYPE_BF16 && (t1 == GGML_TYPE_BF16 || t1 == GGML_TYPE_F32));
}

// the src0 -> result types of the copies of the CPU (ggml_compute_forward_dup); a quantized result also needs the
// from_float of its type, which the supports_op of the CPU checks
static bool rpc_cpu_cpy_types_ok(ggml_type ts, ggml_type td) {
    const bool fs = ts == GGML_TYPE_F32 || ts == GGML_TYPE_F16 || ts == GGML_TYPE_BF16;
    const bool fd = td == GGML_TYPE_F32 || td == GGML_TYPE_F16 || td == GGML_TYPE_BF16;
    return ts == td ||
           (fs && (fd || td == GGML_TYPE_F8_E4M3 || ggml_is_quantized(td) || (ts == GGML_TYPE_F32 && td == GGML_TYPE_I32))) ||
           (ts == GGML_TYPE_F8_E4M3 && fd) ||
           (ts == GGML_TYPE_I32 && td == GGML_TYPE_F32) ||
           (ggml_is_quantized(ts) && td == GGML_TYPE_F32);
}

// on a server with the CPU device: the types that the kernels of the CPU implement for the operations of rpc_node_ok
// (their dispatch aborts with the others, ggml-cpu/ops.cpp), where rpc_node_ok does not fix them already; its supports_op
// accepts almost everything (llamacppOpenCL11)
static bool rpc_cpu_types_ok(const ggml_tensor * t) {
    const ggml_tensor * s0 = t->src[0];
    const ggml_tensor * s1 = t->src[1];
    switch (t->op) {
        case GGML_OP_ADD:
            // or a quantized src0 with an F32 src1 (add_q_f32)
            return rpc_cpu_binary_types_ok(s0->type, s1->type) || (ggml_is_quantized(s0->type) && s1->type == GGML_TYPE_F32);
        case GGML_OP_MUL:
            return rpc_cpu_binary_types_ok(s0->type, s1->type);
        case GGML_OP_SCALE:
            return s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_BF16;
        case GGML_OP_UNARY:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
            return s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16;
        case GGML_OP_SSM_CONV:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_TOP_K:
            return s0->type == GGML_TYPE_F32;
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            return rpc_cpu_cpy_types_ok(s0->type, t->type);
        case GGML_OP_GET_ROWS:
            return s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16 || s0->type == GGML_TYPE_BF16 ||
                   s0->type == GGML_TYPE_I32 || ggml_is_quantized(s0->type);
        case GGML_OP_MUL_MAT:
            // src1: the supports_op of the CPU
            return s0->type == GGML_TYPE_F32 || s0->type == GGML_TYPE_F16 || s0->type == GGML_TYPE_BF16 || ggml_is_quantized(s0->type);
        case GGML_OP_FLASH_ATTN_EXT: {
            // q in F32, K and V in the types of the caches of llama.cpp, and the precisions of the kernel (op_params 3)
            const ggml_tensor * k = t->src[1];
            const ggml_tensor * v = t->src[2];
            auto kv_ok = [](ggml_type type) {
                return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 || type == GGML_TYPE_Q8_0 ||
                       type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1 ||
                       type == GGML_TYPE_IQ4_NL;
            };
            const int32_t prec = ggml_get_op_params_i32(t, 3);
            return s0->type == GGML_TYPE_F32 && kv_ok(k->type) && kv_ok(v->type) &&
                   (prec == GGML_PREC_DEFAULT || prec == GGML_PREC_F32);
        }
        default:
            // CONCAT copies any type; the types of the rest are already those of rpc_node_ok (GATED_DELTA_NET, ARGMAX,
            // PAD, ARANGE in F32, SET_ROWS from F32 or F16)
            return true;
    }
}

// true if the backend of the server runs the node t, which passed rpc_node_ok (graph_compute and supports_op): the
// supports_op of the backend (the kernels of OpenCL need its types and layouts) and, on the CPU, rpc_cpu_types_ok. Not
// the supports_op of an empty node (a graph without outputs has some), which has no value to write: the OpenCL backend
// skips it (and its supports_op may reject it, e.g. TOP_K with k = 0); the CPU computes it, so its types still count.
static bool rpc_backend_runs(ggml_backend_t backend, const ggml_tensor * t) {
    if (ggml_op_is_empty(t->op)) {
        return true;
    }
    if (ggml_backend_dev_type(ggml_backend_get_device(backend)) == GGML_BACKEND_DEVICE_TYPE_CPU && !rpc_cpu_types_ok(t)) {
        return false;
    }
    return ggml_is_empty(t) || ggml_backend_supports_op(backend, t);
}

// llamacppOpenCL11: without recursion. create_node called itself for every source and view_src, with no limit on the
// depth: a client that sent a chain of tens of thousands of tensors (each one a source of the previous one) overflowed
// the stack of the server before graph_compute could reject the graph. A depth limit would not do: a client sends
// the sources of the sources of a graph split, back to the start of the graph (serialize_graph), and a model with many
// layers is a long chain. Now the tensors still to link wait in a vector, and they are created in the same order as
// before (a tensor, then each of its sources and its view_src, depth first), each one once (tensor_map)
ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    // a tensor created whose sources (link 0 to GGML_MAX_SRC - 1) and view_src (link GGML_MAX_SRC) are not linked yet
    struct pending_node {
        uint64_t             id;
        const rpc_tensor   * tensor;
        struct ggml_tensor * result;
        int                  next_link;
    };
    std::vector<pending_node> pending;

    // the tensor of id_node: the one already created, or a new one (null if it does not exist or is not valid)
    auto get_node = [&](uint64_t id_node) -> ggml_tensor * {
        auto it_map = tensor_map.find(id_node);
        if (it_map != tensor_map.end()) {
            return it_map->second;
        }
        // Safely find the tensor pointer
        auto it_ptr = tensor_ptrs.find(id_node);
        if (it_ptr == tensor_ptrs.end()) {
            return nullptr;
        }
        const rpc_tensor * tensor = it_ptr->second;

        struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
        if (result == nullptr) {
            return nullptr;
        }
        if (result->buffer == nullptr && result->data != nullptr) {
            LOG_ERROR("[%s] invalid data ptr\n", "create_node"); // __func__ is the name of the lambda here
            return nullptr;
        }
        tensor_map[id_node] = result;
        pending.push_back({ id_node, tensor, result, 0 });
        return result;
    };

    struct ggml_tensor * root = get_node(id);
    while (root != nullptr && !pending.empty()) {
        // an index, as get_node can add elements and move the vector
        const size_t top  = pending.size() - 1;
        const int    link = pending[top].next_link++;
        if (link > GGML_MAX_SRC) {
            pending[top].result->view_offs = pending[top].tensor->view_offs;
            pending.pop_back();
            continue;
        }
        const uint64_t id_link = link < GGML_MAX_SRC ? pending[top].tensor->src[link] : pending[top].tensor->view_src;
        struct ggml_tensor * linked = nullptr;
        // an ID of 0 is no tensor
        if (id_link != 0) {
            linked = get_node(id_link);
            if (linked == nullptr) {
                if (link < GGML_MAX_SRC) {
                    LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                              __func__, link, id_link, pending[top].id);
                } else {
                    LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                              __func__, id_link, pending[top].id);
                }
                return nullptr;
            }
        }
        if (link < GGML_MAX_SRC) {
            pending[top].result->src[link] = linked;
        } else {
            pending[top].result->view_src = linked;
        }
    }
    return root;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | uid (8 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 3*sizeof(uint32_t) + sizeof(uint64_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint64_t uid;
    memcpy(&uid, src, sizeof(uid));
    src += sizeof(uid);
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    const size_t cGraphHeaderSize = 3*sizeof(uint32_t) + sizeof(uint64_t);
    if (n_nodes > (input.size() - cGraphHeaderSize) / sizeof(uint64_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    const size_t vTensorBytes = input.size() - cGraphHeaderSize - (size_t) n_nodes*sizeof(uint64_t);
    if (n_nodes > n_tensors || n_tensors != vTensorBytes / sizeof(rpc_tensor) || vTensorBytes % sizeof(rpc_tensor) != 0) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 ", n_nodes: %u, n_tensors: %u\n", __func__, device, uid, n_nodes, n_tensors);

    // graphs with uid == 0 are not cached, see GRAPH_CACHE_MAX for the eviction policy
    if (uid != 0 && stored_graphs[device].size() >= GRAPH_CACHE_MAX) {
        LOG_DBG("[%s] clearing graph cache for device %u (%zu entries)\n",
                __func__, device, stored_graphs[device].size());
        stored_graphs[device].clear();
    }
    stored_graph sg_tmp;
    stored_graph & sg = uid != 0 ? stored_graphs[device][uid] : sg_tmp;

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (sg.buffer.size() < buf_size) {
        sg.buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ sg.buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    // the compute itself is asynchronous, so building the graph is the only server-side cost
    // observable here - the round trip is timed on the client
    const auto t_deserialize = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);
        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
        if (graph->nodes[i] != nullptr) {
            LOG_DBG3("[%s] node %u: id=%" PRId64 ", %s\n", __func__, i, id, rpc_log_format_tensor(graph->nodes[i]).c_str());
            const size_t hash_pos = ggml_hash_insert(&graph->visited_hash_set, graph->nodes[i]);
            graph->use_counts[hash_pos] = tensor_ptrs.at(id)->use_count;
        }
    }
    const std::chrono::duration<double, std::milli> deserialize_ms = std::chrono::steady_clock::now() - t_deserialize;

    // the nodes the backends run. A client sends no null nodes, and every node must pass rpc_node_ok (its operation,
    // the shapes and types of its tensors, see above), also the ones that are not computed: ggml_graph_plan of the CPU
    // reads the operation and the sources of every node. create_node accepts tensors without a buffer and with
    // data = 0: the client sends so the tensors of other servers (see serialize_tensor), which a graph can reach as
    // sources of sources. But the backends access the data of every node they compute (an operation and
    // GGML_TENSOR_FLAG_COMPUTE, as they skip the others) and of its sources, with the dimensions and strides of the
    // client: with data = 0, GET_ROWS / SET_ROWS on the CPU read and wrote at addresses chosen by the client, and a view
    // of an OpenCL tensor got an offset outside its cl_mem. Last, the backend must run the node (rpc_backend_runs): the
    // kernels of the OpenCL backend need the types and layouts of ggml_cl_be_supports_op, and those of the CPU its types.
    // A client asks first (RPC_CMD_SUPPORTS_OP), an older one does not, and then the connection closes here.
    for (uint32_t i = 0; i < n_nodes; i++) {
        const ggml_tensor * node = graph->nodes[i];
        if (node == nullptr || !rpc_node_ok(node)) {
            LOG_ERROR("[%s] graph node %u ('%s', op %d): operation or tensors not supported with RPC\n", __func__, i,
                      node ? node->name : "null", node ? (int) node->op : -1);
            return false;
        }
        if (ggml_op_is_empty(node->op) || (node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }
        bool ok = rpc_graph_tensor_ok(node);
        for (int j = 0; j < GGML_MAX_SRC && ok; j++) {
            ok = node->src[j] == nullptr || rpc_graph_tensor_ok(node->src[j]);
        }
        if (!ok) {
            LOG_ERROR("[%s] graph node %u ('%s', op %d) or one of its sources has no data in a buffer of this server\n",
                      __func__, i, node->name, (int) node->op);
            return false;
        }
        if (!rpc_backend_runs(backends[device], node)) {
            LOG_ERROR("[%s] graph node %u ('%s', %s) not supported by %s\n", __func__, i, node->name, ggml_op_name(node->op),
                      ggml_backend_name(backends[device]));
            return false;
        }
    }

    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    sg.graph = graph;
    LOG_DBG3("[%s] device: %u, graph build: %.3f ms\n", __func__, device, deserialize_ms.count());
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    auto it = stored_graphs[device].find(request.uid);
    if (it == stored_graphs[device].end() || it->second.graph == nullptr) {
        LOG_ERROR("[%s] device: %u, graph with uid %" PRIu64 " not found\n", __func__, device, request.uid);
        return false;
    }
    ggml_cgraph * graph = it->second.graph;
    LOG_DBG("[%s] device: %u, uid: %" PRIu64 "\n", __func__, device, request.uid);
    ggml_status status = ggml_backend_graph_compute_async(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

// llamacppOpenCL11: whether the backend of a device runs a node, for the supports_op of the client (rpc_remote_supports_op):
// the checks of graph_compute (rpc_node_ok, rpc_backend_runs) but rpc_graph_tensor_ok, which needs the buffers of an
// allocated graph. A view gets its data from the tensor it views plus its offset (the client sends data = 0 for the
// views that it did not initialize yet), as the OpenCL backend checks the offsets of the views of its repacked layouts.
// False only for a malformed request; a node that does not run is response.supported = 0.
bool rpc_server::supports_op(const std::vector<uint8_t> & input, rpc_msg_supports_op_rsp & response) {
    response.supported = 0;
    uint32_t device;
    uint32_t n_tensors;
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    memcpy(&device,    input.data(),                    sizeof(device));
    memcpy(&n_tensors, input.data() + sizeof(uint32_t), sizeof(n_tensors));
    if (device >= backends.size() || n_tensors == 0 || n_tensors > 2*(1 + GGML_MAX_SRC) ||
        input.size() != 2*sizeof(uint32_t) + (size_t) n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *) (input.data() + 2*sizeof(uint32_t));

    struct ggml_init_params params {
        /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    std::unordered_map<uint64_t, ggml_tensor *> by_id;
    std::vector<ggml_tensor *> list(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        list[i] = deserialize_tensor(ctx_ptr.get(), &tensors[i]);
        if (list[i] == nullptr) {
            return true; // a layout that the server rejects
        }
        by_id[tensors[i].id] = list[i];
    }
    auto find = [&by_id](uint64_t id) -> ggml_tensor * {
        const auto it = by_id.find(id);
        return id == 0 || it == by_id.end() ? nullptr : it->second;
    };
    for (uint32_t i = 0; i < n_tensors; i++) {
        ggml_tensor * t = list[i];
        t->view_src  = find(tensors[i].view_src);
        t->view_offs = tensors[i].view_offs;
        if (t->view_src != nullptr) {
            t->buffer = t->view_src->buffer;
            t->data   = (void *) ((uintptr_t) t->view_src->data + t->view_offs);
        }
    }
    ggml_tensor * node = list[0];
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        node->src[j] = find(tensors[0].src[j]);
        if (tensors[0].src[j] != 0 && node->src[j] == nullptr) {
            return true; // a source that the request does not carry
        }
    }
    response.supported = rpc_node_ok(node) && rpc_backend_runs(backends[device], node);
    LOG_DBG2("[%s] device: %u, '%s' (op %d): %s\n", __func__, device, node->name, (int) node->op,
             response.supported ? "supported" : "not supported");
    return true;
}

// graph compute is asynchronous; commands that read or write buffer data synchronize first
void rpc_server::sync_all_backends() {
    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }
}

void rpc_server::comm_free_device(uint32_t dev_id) {
    comm_state & state = comm_states[dev_id];
    if (state.backend_comm_ctx != nullptr) {
        auto comm_free = (ggml_backend_comm_free_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[dev_id])), "ggml_backend_comm_free");
        if (comm_free != nullptr) {
            comm_free(state.backend_comm_ctx);
        }
    }
    state = comm_state();
}

// Exchange RPC transport capabilities over a peer connection. recv_first selects which side
// receives first. Both sides receiving first would deadlock.
static bool comm_caps_handshake(const socket_ptr & peer, bool recv_first) {
    uint8_t local_caps[RPC_CONN_CAPS_SIZE]  = {};
    uint8_t remote_caps[RPC_CONN_CAPS_SIZE] = {};
    peer->get_caps(local_caps);
    if (recv_first) {
        if (!peer->recv_data(remote_caps, sizeof(remote_caps)) ||
            !peer->send_data(local_caps, sizeof(local_caps))) {
            return false;
        }
    } else {
        if (!peer->send_data(local_caps, sizeof(local_caps)) ||
            !peer->recv_data(remote_caps, sizeof(remote_caps))) {
            return false;
        }
    }
    peer->update_caps(remote_caps);
    return true;
}

// llamacppOpenCL11: with a token (GGML_RPC_TOKEN), the two ends of a comm link prove to each other that they know it,
// before comm_caps_handshake: rank 0 sends a nonce, the other server answers with its HMAC (RPC_AUTH_COMM_PEER) and a
// nonce of its own, and rank 0 answers that one (RPC_AUTH_COMM_RANK0). Both the listener and the outbound connection
// are restricted to GGML_RPC_COMM_ENDPOINT. The client only builds a communicator with servers that
// all have a token or none (ggml_backend_rpc_comm_init), as one end with a token waits for an answer that the other one
// does not send. As with the clients, the link is not encrypted.
static constexpr uint32_t RPC_COMM_MAX_REJECTED = 16; // connections without the token that rank 0 closes before it gives up

// llamacppOpenCL11: the handshake of a connection (the HELLO and the authentication of a client, and the authentication,
// the caps and the rank of a server of a communicator) must end within RPC_HANDSHAKE_TIMEOUT. The server waited with no
// time limit: a connection that sent nothing, even without the token, kept the server busy (and every other client
// out) until it closed
static constexpr auto RPC_HANDSHAKE_TIMEOUT = std::chrono::seconds(10);

// runs fn, which receives from sock: if it has not returned after RPC_HANDSHAKE_TIMEOUT, sock is shut down, so that its
// receives fail, and the result is false. The receives of the transport wait without a limit (also with RDMA, whose
// waits end when the TCP socket hangs up), so the deadline is a thread that waits for fn
template <typename F>
static bool rpc_run_with_deadline(const socket_ptr & sock, F && fn) {
    std::mutex              mutex;
    std::condition_variable cv;
    bool                    done      = false;
    bool                    timed_out = false;
    std::thread             watchdog;
    try {
        watchdog = std::thread([&]() {
            std::unique_lock<std::mutex> lock(mutex);
            if (!cv.wait_for(lock, RPC_HANDSHAKE_TIMEOUT, [&]() { return done; })) {
                timed_out = true;
                sock->shutdown();
            }
        });
    } catch (const std::system_error & e) {
        LOG_ERROR("[%s] cannot start the deadline of a handshake: %s\n", __func__, e.what());
        return false;
    }
    const bool ok = fn();
    {
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
    }
    cv.notify_one();
    watchdog.join();
    if (timed_out) {
        LOG_WARN("[%s] a connection did not complete its handshake in %d s: closed\n", __func__,
                 (int) RPC_HANDSHAKE_TIMEOUT.count());
    }
    return ok && !timed_out;
}

static bool rpc_comm_auth(const socket_ptr & peer, const std::string & token, bool rank0) {
    uint8_t nonce_rank0[32];
    uint8_t nonce_peer[32];
    uint8_t mac_peer[32];
    uint8_t mac_rank0[32];
    uint8_t expected[32];
    if (rank0) {
        if (!rpc_random_bytes(nonce_rank0, sizeof(nonce_rank0)) ||
            !peer->send_data(nonce_rank0, sizeof(nonce_rank0)) || !peer->flush() ||
            !peer->recv_data(nonce_peer, sizeof(nonce_peer)) || !peer->recv_data(mac_peer, sizeof(mac_peer))) {
            return false;
        }
        rpc_auth_mac(token, RPC_AUTH_COMM_PEER, nonce_rank0, expected);
        if (!rpc_auth_mac_equal(expected, mac_peer)) {
            LOG_ERROR("[%s] a comm peer connection has not the token of GGML_RPC_TOKEN\n", __func__);
            return false;
        }
        rpc_auth_mac(token, RPC_AUTH_COMM_RANK0, nonce_peer, mac_rank0);
        return peer->send_data(mac_rank0, sizeof(mac_rank0)) && peer->flush();
    }
    if (!peer->recv_data(nonce_rank0, sizeof(nonce_rank0)) || !rpc_random_bytes(nonce_peer, sizeof(nonce_peer))) {
        return false;
    }
    rpc_auth_mac(token, RPC_AUTH_COMM_PEER, nonce_rank0, mac_peer);
    if (!peer->send_data(nonce_peer, sizeof(nonce_peer)) || !peer->send_data(mac_peer, sizeof(mac_peer)) ||
        !peer->flush() || !peer->recv_data(mac_rank0, sizeof(mac_rank0))) {
        LOG_ERROR("[%s] rank 0 of the communicator did not authenticate (GGML_RPC_TOKEN)\n", __func__);
        return false;
    }
    rpc_auth_mac(token, RPC_AUTH_COMM_RANK0, nonce_peer, expected);
    if (!rpc_auth_mac_equal(expected, mac_rank0)) {
        LOG_ERROR("[%s] rank 0 of the communicator has not the token of GGML_RPC_TOKEN\n", __func__);
        return false;
    }
    return true;
}

// The comm link between servers uses the same caps negotiation as the client HELLO,
// so it gets the same transport upgrades (e.g. RDMA).
bool rpc_server::comm_init(const rpc_msg_comm_init_req & request, rpc_msg_comm_init_rsp & response) {
    response.ok = 0;
    if (request.device >= backends.size() || request.world < 2 || request.world > 16 || request.rank >= request.world) {
        return true;
    }
    const std::string vRequestedHost(request.host, strnlen(request.host, sizeof(request.host)));
    if (vCommConfig.port == 0 || request.port != vCommConfig.port ||
        (request.rank != 0 && vRequestedHost != vCommConfig.host)) {
      LOG_ERROR("[%s] communicator endpoint not authorized by GGML_RPC_COMM_ENDPOINT\n", __func__);
      return true;
    }
    const uint32_t dev_id = request.device;
    comm_state & state = comm_states[dev_id];
    if (state.allreduce_fn != nullptr) {
        response.ok = 1;
        return true;
    }

    // Joining a communicator blocks until every rank arrives, so drain outstanding work first.
    sync_all_backends();

    // Rank 0 listens and accepts world-1 peers; every other rank connects back to it.
    if (request.rank == 0) {
        socket_ptr srv = socket_t::create_server(vCommConfig.host.c_str(), vCommConfig.port);
        if (srv == nullptr) {
            LOG_ERROR("[%s] failed to listen on comm port %u\n", __func__, request.port);
            return true;
        }
        // Store peers[i] as rank i + 1, as comm_bootstrap_exchange needs.
        // Each peer sends its rank because peers may connect in any order.
        state.peers.resize(request.world - 1);
        uint32_t n_rejected = 0;
        for (uint32_t i = 1; i < request.world; i++) {
            socket_ptr peer = srv->accept(nullptr, (int) RPC_HANDSHAKE_TIMEOUT.count() * 1000);
            if (peer == nullptr) {
              state.peers.clear();
              return true;
            }
            // llamacppOpenCL11: with a token, a connection that does not prove it is closed and does not take the
            // place of a peer (see rpc_comm_auth); every step has a deadline (rpc_run_with_deadline),
            // because a connection that sent nothing used to block it
            if (peer != nullptr && !auth_token.empty() &&
                !rpc_run_with_deadline(peer, [&]() { return rpc_comm_auth(peer, auth_token, /*rank0 =*/ true); })) {
                if (++n_rejected <= RPC_COMM_MAX_REJECTED) {
                    i--;
                    continue;
                }
                peer = nullptr;
            }
            uint32_t peer_rank = 0;
            if (peer == nullptr ||
                !rpc_run_with_deadline(peer, [&]() {
                    return comm_caps_handshake(peer, /*recv_first =*/ true) && peer->recv_data(&peer_rank, sizeof(peer_rank));
                }) ||
                peer_rank == 0 || peer_rank >= request.world || state.peers[peer_rank - 1] != nullptr) {
                LOG_ERROR("[%s] bad peer connection (rank %u)\n", __func__, peer_rank);
                state.peers.clear();
                return true;
            }
            state.peers[peer_rank - 1] = std::move(peer);
        }
    } else {
        const std::string & host = vCommConfig.host;
        socket_ptr peer;
        // rank 0 may not be listening yet, retry for a few seconds
        for (int i = 0; i < 100 && peer == nullptr; i++) {
            peer = socket_t::connect(host.c_str(), vCommConfig.port);
            if (peer == nullptr) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (peer == nullptr) {
            LOG_ERROR("[%s] failed to connect to peer %s:%u\n", __func__, host.c_str(), request.port);
            return true;
        }
        if (!rpc_run_with_deadline(peer, [&]() {
              return (auth_token.empty() || rpc_comm_auth(peer, auth_token, /*rank0 =*/ false)) &&
                comm_caps_handshake(peer, /*recv_first =*/ false) &&
                peer->send_data(&request.rank, sizeof(request.rank)) && peer->flush();
            })) {
            return true;
        }
        state.peers.push_back(std::move(peer));
    }

    state.rank  = request.rank;
    state.world = request.world;

    // Forces the pairwise path, for comparing the two on one build. Setting it on any one rank
    // is enough, because the negotiation spreads the decision.
    const bool no_backend_communicator = std::getenv("GGML_RPC_NO_BACKEND_COMMUNICATOR") != nullptr;

    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backends[dev_id]));
    auto init_rank    = (ggml_backend_comm_init_rank_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_init_rank");
    auto allreduce_fn = (ggml_backend_comm_allreduce_tensor_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_comm_allreduce_tensor");

    // The ranks must agree on backend communicator or pairwise. Every rank shares whether its
    // backend provides a communicator, and all ranks use it only if every rank can.
    const uint8_t self_can_use_backend_comm = !no_backend_communicator && init_rank != nullptr && allreduce_fn != nullptr;
    std::vector<uint8_t> all_can_use_backend_comm(request.world);
    if (!comm_bootstrap_exchange(&state, &self_can_use_backend_comm, all_can_use_backend_comm.data(),
                                 sizeof(self_can_use_backend_comm))) {
        state.peers.clear();
        return true;
    }
    const bool use_backend_communicator = std::all_of(all_can_use_backend_comm.begin(), all_can_use_backend_comm.end(),
                                                      [](uint8_t c) { return c != 0; });

    if (use_backend_communicator) {
        void * backend_ctx = init_rank(backends[dev_id], (int) request.rank, (int) request.world, comm_bootstrap_exchange, &state);
        if (backend_ctx == nullptr) {
            LOG_ERROR("[%s] failed to join communicator as rank %u/%u\n", __func__, request.rank, request.world);
            state.peers.clear();
            return true;
        }
        state.backend_comm_ctx = backend_ctx;
        state.allreduce_fn     = allreduce_fn;
        // The sockets have done their job; the collective runs between the backends from here on.
        state.peers.clear();
        LOG_INFO("[%s] device %u using backend communicator, rank %u/%u\n", __func__, dev_id, request.rank, request.world);
        response.ok = 1;
        return true;
    }

    // No backend communicator, so fall back to the pairwise exchange, which only covers two ranks.
    if (request.world != 2) {
        LOG_ERROR("[%s] world size %u requires a backend communicator\n", __func__, request.world);
        state.peers.clear();
        return true;
    }
    state.pairwise_backend = backends[dev_id];
    state.allreduce_fn     = comm_allreduce_pairwise;
    LOG_INFO("[%s] device %u using pairwise communicator, rank %u\n", __func__, dev_id, request.rank);
    response.ok = 1;
    return true;
}

// Bootstrap callback for ggml_backend_comm_init_rank_t, over the peer sockets. Rank 0 collects
// size bytes from every rank in rank order, then sends the whole array back to every rank.
// user_data is the comm_state.
bool rpc_server::comm_bootstrap_exchange(void * user_data, const void * send_buf, void * recv_buf, size_t size) {
    comm_state & state = *(comm_state *) user_data;
    uint8_t * recv = (uint8_t *) recv_buf;
    const size_t total = (size_t) state.world * size;

    if (state.rank == 0) {
        memcpy(recv, send_buf, size);
        for (size_t i = 0; i < state.peers.size(); i++) {
            if (!state.peers[i]->recv_data(recv + (i + 1) * size, size)) {
                return false;
            }
        }
        for (const socket_ptr & peer : state.peers) {
            if (!peer->send_data(recv, total) || !peer->flush()) {
                return false;
            }
        }
        return true;
    }
    return state.peers[0]->send_data(send_buf, size) && state.peers[0]->flush() &&
           state.peers[0]->recv_data(recv, total);
}

// Reduces this rank's contribution in place, exchanging partials with the peer over peers[0].
// Used when the backend provides no communicator. comm_ctx_v is the comm_state.
bool rpc_server::comm_allreduce_pairwise(void * comm_ctx_v, ggml_tensor ** tensors) {
    comm_state & state = *(comm_state *) comm_ctx_v;
    ggml_tensor * t_dst = tensors[0];
    ggml_backend_t backend = state.pairwise_backend;

    const size_t  nbytes = ggml_nbytes(t_dst);
    const int64_t ne     = ggml_nelements(t_dst);

    size_t ctx_size = 16*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(8, false);
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    // reduce large partials in bf16 to halve the wire bytes; small (decode-sized) ones
    // stay f32 since the extra casts and sync cost more than the bytes saved
    const bool   wire_bf16  = t_dst->type == GGML_TYPE_F32 && ne >= 32768;
    const size_t wire_bytes = wire_bf16 ? (size_t) ne*2 : nbytes;
    const size_t need       = wire_bf16 ? 2*nbytes : nbytes;
    if (state.scratch_size < need) {
        state.scratch.reset(ggml_backend_alloc_buffer(backend, need));
        state.scratch_size = need;
    }
    char * scratch_base = (char *) ggml_backend_buffer_get_base(state.scratch.get());
    state.send_buf.resize(wire_bytes);
    state.recv_buf.resize(wire_bytes);

    auto new_scratch_tensor = [&](ggml_type type, size_t offset) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
        t->buffer = state.scratch.get();
        t->data   = scratch_base + offset;
        return t;
    };
    auto new_cpy_node = [&](ggml_tensor * src, ggml_tensor * dst) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx, dst->type, dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
        t->op     = GGML_OP_CPY;
        t->src[0] = src;
        t->src[1] = dst;
        t->buffer = dst->buffer;
        t->data   = dst->data;
        t->flags |= GGML_TENSOR_FLAG_COMPUTE;
        return t;
    };
    auto compute_nodes = [&](ggml_tensor * n0, ggml_tensor * n1) {
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 2, false);
        graph->nodes[0] = n0;
        graph->nodes[1] = n1;
        graph->n_nodes  = n1 != nullptr ? 2 : 1;
        ggml_status status = ggml_backend_graph_compute_async(backend, graph);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    };

    // wait for the pending subgraph that produced this partial
    ggml_backend_synchronize(backend);

    ggml_tensor * t_wire_send = nullptr;
    ggml_tensor * t_wire_recv = nullptr;
    ggml_tensor * t_peer      = new_scratch_tensor(t_dst->type, wire_bf16 ? (size_t) ne*4 : 0);
    ggml_tensor * t_cast      = nullptr;
    // llamacppOpenCL11: the casts to and from bf16 run on the backend only if it runs both copies; if not, on the host,
    // with the same wire format, so the two ranks still agree on it. The OpenCL backend of the fork has no bf16: its
    // copy kernel wrote the bf16 values as f32, 4 bytes every 2, and the NVIDIA driver invalidated the queue on the
    // misaligned writes (clFinish error -36; Beignet did not report them, but the values were wrong anyway)
    bool cast_on_backend = false;
    if (wire_bf16) {
        t_wire_send = new_scratch_tensor(GGML_TYPE_BF16, 0);
        t_wire_recv = new_scratch_tensor(GGML_TYPE_BF16, ne*2);
        ggml_tensor * t_to_wire = new_cpy_node(t_dst, t_wire_send);
        t_cast = new_cpy_node(t_wire_recv, t_peer);
        cast_on_backend = ggml_backend_supports_op(backend, t_to_wire) && ggml_backend_supports_op(backend, t_cast);
        if (cast_on_backend) {
            compute_nodes(t_to_wire, nullptr);
            ggml_backend_synchronize(backend);
            ggml_backend_tensor_get(t_wire_send, state.send_buf.data(), 0, wire_bytes);
        } else {
            t_cast = nullptr;
            state.host_f32.resize(ne);
            ggml_backend_tensor_get(t_dst, state.host_f32.data(), 0, nbytes);
            ggml_fp32_to_bf16_row_ref(state.host_f32.data(), (ggml_bf16_t *) state.send_buf.data(), ne);
        }
    } else {
        ggml_backend_tensor_get(t_dst, state.send_buf.data(), 0, wire_bytes);
    }

    // rank 0 sends first, rank 1 receives first, so large payloads cannot deadlock
    if (state.rank == 0) {
        if (!state.peers[0]->send_data(state.send_buf.data(), wire_bytes) || !state.peers[0]->flush() ||
            !state.peers[0]->recv_data(state.recv_buf.data(), wire_bytes)) {
            return false;
        }
    } else {
        if (!state.peers[0]->recv_data(state.recv_buf.data(), wire_bytes) ||
            !state.peers[0]->send_data(state.send_buf.data(), wire_bytes) || !state.peers[0]->flush()) {
            return false;
        }
    }

    if (wire_bf16 && cast_on_backend) {
        ggml_backend_tensor_set(t_wire_recv, state.recv_buf.data(), 0, wire_bytes);
    } else if (wire_bf16) {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) state.recv_buf.data(), state.host_f32.data(), ne);
        ggml_backend_tensor_set(t_peer, state.host_f32.data(), 0, nbytes);
    } else {
        ggml_backend_tensor_set(t_peer, state.recv_buf.data(), 0, wire_bytes);
    }

    ggml_tensor * t_red = ggml_new_tensor_4d(ctx, t_dst->type, t_dst->ne[0], t_dst->ne[1], t_dst->ne[2], t_dst->ne[3]);
    t_red->op     = GGML_OP_ADD;
    t_red->src[0] = t_dst;
    t_red->src[1] = t_peer;
    t_red->buffer = t_dst->buffer;
    t_red->data   = t_dst->data;
    t_red->flags |= GGML_TENSOR_FLAG_COMPUTE;

    if (t_cast != nullptr) {
        compute_nodes(t_cast, t_red);
    } else {
        compute_nodes(t_red, nullptr);
    }
    return true;
}

bool rpc_server::comm_allreduce(const rpc_msg_comm_allreduce_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    comm_state & state = comm_states[request.device];
    if (state.allreduce_fn == nullptr) {
        LOG_ERROR("[%s] no communicator for device %u\n", __func__, request.device);
        return false;
    }

    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_tensor * t_dst = deserialize_tensor(ctx_ptr.get(), &request.tensor);
    if (t_dst == nullptr || t_dst->buffer == nullptr) {
        LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    if (ggml_nbytes(t_dst) == 0) {
        return true;
    }
    // llamacppOpenCL11: a contiguous f32 tensor, as the client sends (ggml_backend_rpc_comm_allreduce_tensor): the
    // pairwise path adds the partial of the other rank as ggml_nelements consecutive floats, and the kernels need the
    // layout (an f32 tensor with nb[0] = 2 made the OpenCL kernels write misaligned, which the NVIDIA driver does not
    // survive)
    if (t_dst->type != GGML_TYPE_F32 || !ggml_is_contiguous(t_dst)) {
        LOG_ERROR("[%s] tensor %s is not a contiguous f32 tensor\n", __func__, t_dst->name);
        return false;
    }

    // Neither path needs a wait here. The pairwise path waits internally, and the backend orders
    // its reduction after the work that produced the partial sum.
    void * comm_ctx = state.backend_comm_ctx != nullptr ? state.backend_comm_ctx : &state;
    if (!state.allreduce_fn(comm_ctx, &t_dst)) {
        LOG_ERROR("[%s] allreduce failed for tensor %s\n", __func__, t_dst->name);
        return false;
    }
    return true;
}

bool rpc_server::comm_free(const rpc_msg_comm_free_req & request) {
    if (request.device >= backends.size()) {
        return false;
    }
    // Wait for outstanding work. Destroying a communicator while its kernels are still running
    // is unsafe.
    sync_all_backends();
    comm_free_device(request.device);
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG2("[%s] device: %u, free: %s, total: %s\n", __func__, dev_id,
             rpc_log_format_size(response.free_mem).c_str(), rpc_log_format_size(response.total_mem).c_str());
    return true;
}

rpc_server::~rpc_server() {
    // Wait for outstanding work. Destroying a communicator while its kernels are still running
    // is unsafe.
    sync_all_backends();

    // Free any communicator still open. A client can disconnect without sending COMM_FREE.
    for (uint32_t dev_id = 0; dev_id < comm_states.size(); dev_id++) {
        comm_free_device(dev_id);
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
}

// the server side of the authentication (see rpc_auth_token): RPC_CMD_AUTH_NONCE, then RPC_CMD_AUTH with the HMAC of the
// nonce; any other command closes the connection. A wrong token waits half a second, to slow down the guesses.
static bool rpc_serve_auth(const socket_ptr & sock, const std::string & token) {
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1) || cmd != RPC_CMD_AUTH_NONCE || !recv_msg(sock, nullptr, 0)) {
        LOG_ERROR("[%s] the client did not authenticate (GGML_RPC_TOKEN)\n", __func__);
        return false;
    }
    rpc_msg_auth_nonce_rsp nonce;
    if (!rpc_random_bytes(nonce.nonce, sizeof(nonce.nonce)) || !send_msg(sock, &nonce, sizeof(nonce))) {
        return false;
    }
    rpc_msg_auth_req request;
    if (!sock->recv_data(&cmd, 1) || cmd != RPC_CMD_AUTH || !recv_msg(sock, &request, sizeof(request))) {
        return false;
    }
    uint8_t mac[32];
    rpc_auth_mac(token, RPC_AUTH_CLIENT, nonce.nonce, mac);
    rpc_msg_auth_rsp response;
    response.ok = rpc_auth_mac_equal(mac, request.mac) ? 1 : 0;
    if (!response.ok) {
        LOG_ERROR("[%s] wrong token\n", __func__);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return send_msg(sock, &response, sizeof(response)) && response.ok;
}

// the handshake of a connection: HELLO and, with a token, the authentication (rpc_serve_auth). llamacppOpenCL11: apart
// from rpc_serve_client, as it runs in the thread of the connection with a deadline (see ggml_backend_rpc_start_server)
static bool rpc_serve_handshake(const socket_ptr & sock, const std::string & auth_token, const rpc_comm_config & pCommConfig) {
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return false;
    }
    if (cmd != RPC_CMD_HELLO) {
        LOG_ERROR("Expected HELLO command, update client\n");
        return false;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return false;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return false;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return false;
    }

    rpc_msg_hello_rsp rsp = {};
    rpc_server::hello(rsp);
    rsp.features = RPC_FEATURE_SUPPORTS_OP | (auth_token.empty() ? 0 : RPC_FEATURE_AUTH) |
      (pCommConfig.port == 0 ? RPC_FEATURE_COMM_DISABLED : 0);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return false;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);
    if (!auth_token.empty() && !rpc_serve_auth(sock, auth_token)) {
        return false;
    }
    return true;
}

// the commands of a client, after its handshake (rpc_serve_handshake)
static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock, const std::string & auth_token, const rpc_comm_config & pCommConfig) {
    rpc_server server(backends, cache_dir, auth_token, pCommConfig);
    uint8_t cmd;
    LOG_DBG("[%s] client connected, transport: %s\n", __func__, sock->transport_name());
    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        LOG_DBG2("[%s] %s\n", __func__, rpc_cmd_name((enum rpc_cmd)cmd));
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_AUTH_NONCE:
            case RPC_CMD_AUTH: {
                // only right after HELLO, and only if the server has a token (see rpc_serve_auth)
                LOG_ERROR("[%s] unexpected %s command\n", __func__, rpc_cmd_name((enum rpc_cmd) cmd));
                return;
            }
            case RPC_CMD_SUPPORTS_OP: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input, cRpcMaxSupportsFrame, 2*sizeof(uint32_t) + sizeof(rpc_tensor))) {
                    return;
                }
                rpc_msg_supports_op_rsp response;
                if (!server.supports_op(input, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_MEMSET_TENSOR: {
                rpc_msg_memset_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.memset_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                if (!server.set_tensor(sock)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_2D: {
                if (!server.set_tensor_2d(sock)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR_2D: {
                rpc_msg_get_tensor_2d_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor_2d(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input, cRpcMaxGraphFrame, 3*sizeof(uint32_t) + sizeof(uint64_t))) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_INIT: {
                rpc_msg_comm_init_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_comm_init_rsp response;
                if (!server.comm_init(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_ALLREDUCE: {
                rpc_msg_comm_allreduce_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_allreduce(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_COMM_FREE: {
                rpc_msg_comm_free_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.comm_free(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                LOG_ERROR("Unhandled command: %s\n", rpc_cmd_name((enum rpc_cmd)cmd));
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        LOG_ERROR("Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    rpc_comm_config vCommConfig;
    if (!fRpcCommConfig(vCommConfig)) {
      LOG_ERROR("GGML_RPC_COMM_ENDPOINT must be a canonical unicast IPv4:port (no DNS names or wildcard addresses)\n");
      return;
    }
    LOG_INFO("  communicator   : %s\n", vCommConfig.port == 0 ? "disabled (set GGML_RPC_COMM_ENDPOINT to enable)" :
      (vCommConfig.host + ":" + std::to_string(vCommConfig.port)).c_str());
    std::vector<ggml_backend_t> backends;
    LOG_INFO("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    LOG_INFO("  endpoint       : %s\n", endpoint);
    LOG_INFO("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    const std::string auth_token = rpc_auth_token();
    LOG_INFO("  authentication : %s\n", auth_token.empty() ? "off (set GGML_RPC_TOKEN to require a token)" : "on (GGML_RPC_TOKEN)");
    LOG_INFO("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        LOG_INFO("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
                 total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            LOG_ERROR("Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    LOG_INFO("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    LOG_INFO("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        LOG_ERROR("Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        LOG_ERROR("Failed to create server socket\n");
        return;
    }

    // llamacppOpenCL11: the server served the connections one after another, from their HELLO, in this thread, and it
    // waited for that HELLO with no time limit: a connection that sent nothing, even without the token, kept every
    // client out. Now every connection runs its handshake (rpc_serve_handshake) in a thread of its own, with a deadline
    // (rpc_run_with_deadline), and then waits for its turn: the server still serves one client at a time. At most
    // RPC_MAX_PENDING_CONNS connections in their handshake or waiting for their turn, and RPC_MAX_HANDSHAKES_PER_ADDR
    // of them in their handshake from one address, so that the connections that do not authenticate cannot take
    // every place; the others are closed at once
    static constexpr uint32_t RPC_MAX_PENDING_CONNS       = 16;
    static constexpr uint32_t RPC_MAX_HANDSHAKES_PER_ADDR = 4;
    struct conn_state {
        std::mutex                             mutex;        // the counters
        uint32_t                               n_pending = 0;
        std::unordered_map<uint32_t, uint32_t> n_handshakes; // per IPv4 address
        std::mutex                             serve_mutex;  // held while a client is served
    };
    auto state = std::make_shared<conn_state>();
    // the end of the handshake of a connection from addr (under state->mutex)
    auto handshake_done = [](conn_state & st, uint32_t addr) {
        auto it = st.n_handshakes.find(addr);
        if (it != st.n_handshakes.end() && --it->second == 0) {
            st.n_handshakes.erase(it);
        }
    };
    while (true) {
        uint32_t peer_addr = 0;
        auto client_socket = server_socket->accept(&peer_addr);
        if (client_socket == nullptr) {
            LOG_ERROR("Failed to accept client connection\n");
            return;
        }
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            auto it = state->n_handshakes.find(peer_addr);
            if (state->n_pending >= RPC_MAX_PENDING_CONNS ||
                (it != state->n_handshakes.end() && it->second >= RPC_MAX_HANDSHAKES_PER_ADDR)) {
                LOG_WARN("Too many pending client connections: a new one is closed\n");
                continue;
            }
            state->n_pending++;
            state->n_handshakes[peer_addr]++;
        }
        LOG_INFO("Accepted client connection\n");
        try {
            std::thread([state, handshake_done, client_socket, peer_addr, backends, cache_dir, auth_token, vCommConfig]() {
                const bool ok = rpc_run_with_deadline(client_socket, [&]() {
                    return rpc_serve_handshake(client_socket, auth_token, vCommConfig);
                });
                {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    handshake_done(*state, peer_addr);
                    if (!ok) {
                        state->n_pending--;
                    }
                }
                if (ok) {
                    std::lock_guard<std::mutex> serve_lock(state->serve_mutex);
                    {
                        std::lock_guard<std::mutex> lock(state->mutex);
                        state->n_pending--;
                    }
                    try {
                      rpc_serve_client(backends, cache_dir, client_socket, auth_token, vCommConfig);
                    } catch (const std::exception & vError) {
                      LOG_ERROR("RPC client command failed: %s\n", vError.what());
                    }
                }
                LOG_INFO("Client connection closed\n");
            }).detach();
        } catch (const std::system_error & e) {
            LOG_ERROR("Failed to start the thread of a client connection: %s\n", e.what());
            std::lock_guard<std::mutex> lock(state->mutex);
            handshake_done(*state, peer_addr);
            state->n_pending--;
        }
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ true,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

// llamacppOpenCL11: asks the server whether its backend runs the node op (RPC_CMD_SUPPORTS_OP, see rpc_server::supports_op),
// with a cache. The answer depends on the operation and its op_params, on the types, dimensions and strides of op and of
// its sources, on the offsets of the views and on whether a tensor is already in a buffer of this server; not on the
// data pointers, which the cache key leaves out (the views carry their offset).
static bool rpc_remote_supports_op(const std::shared_ptr<rpc_dispatcher> & dispatcher, const ggml_backend_rpc_device_context * ctx,
                                   const ggml_tensor * op) {
    // the node first, then the tensor it views and its sources with the tensors that they view, without repetitions
    std::vector<const ggml_tensor *> tensors = { op };
    auto add = [&tensors](const ggml_tensor * t) {
        if (t != nullptr && std::find(tensors.begin(), tensors.end(), t) == tensors.end()) {
            tensors.push_back(t);
        }
    };
    add(op->view_src);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] != nullptr) {
            add(op->src[i]);
            add(op->src[i]->view_src);
        }
    }
    auto index_of = [&tensors](const ggml_tensor * t) {
        const auto it = std::find(tensors.begin(), tensors.end(), t);
        return t == nullptr || it == tensors.end() ? (int32_t) -1 : (int32_t) (it - tensors.begin());
    };

    struct supports_op_key {
        uint32_t type;
        uint32_t op;
        int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
        uint64_t ne[GGML_MAX_DIMS];
        uint64_t nb[GGML_MAX_DIMS];
        uint64_t view_offs;
        int32_t  view_src;
        int32_t  src[GGML_MAX_SRC];
        uint32_t in_buffer;
    };

    const size_t   n      = tensors.size();
    const uint32_t device = ctx->device;
    const uint32_t n32    = (uint32_t) n;
    const size_t   size   = 2*sizeof(uint32_t) + n*sizeof(rpc_tensor);
    std::shared_ptr<uint8_t> input(new uint8_t[size], std::default_delete<uint8_t[]>());
    memcpy(input.get(), &device, sizeof(device));
    memcpy(input.get() + sizeof(uint32_t), &n32, sizeof(n32));
    rpc_tensor * out = (rpc_tensor *) (input.get() + 2*sizeof(uint32_t));
    std::string key = ctx->endpoint + "#" + std::to_string(device) + "#";
    for (size_t i = 0; i < n; i++) {
        const ggml_tensor * t = tensors[i];
        out[i] = serialize_tensor(t, dispatcher.get());
        if (i > 0) {
            // the server links only the sources of the node
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                out[i].src[j] = 0;
            }
        }
        supports_op_key k;
        memset(&k, 0, sizeof(k));
        k.type = t->type;
        if (i == 0) {
            k.op = t->op;
            memcpy(k.op_params, t->op_params, sizeof(k.op_params));
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                k.src[j] = index_of(t->src[j]);
            }
        }
        for (int j = 0; j < GGML_MAX_DIMS; j++) {
            k.ne[j] = (uint64_t) t->ne[j];
            k.nb[j] = (uint64_t) t->nb[j];
        }
        k.view_offs = t->view_src != nullptr ? (uint64_t) t->view_offs : 0;
        k.view_src  = index_of(t->view_src);
        k.in_buffer = out[i].buffer != 0;
        key.append((const char *) &k, sizeof(k));
    }

    static std::mutex                             cache_mutex;
    static std::unordered_map<std::string, bool> cache;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second;
        }
    }

    rpc_msg_supports_op_rsp response = {};
    dispatcher->send(RPC_CMD_SUPPORTS_OP, input, size, &response, sizeof(response));
    const bool supported = response.supported != 0;
    LOG_DBG2("[%s] %s (%s): %s\n", __func__, op->name, ggml_op_desc(op), supported ? "supported" : "not supported");
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[key] = supported;
    }
    return supported;
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    if ((op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK) &&
        !ggml_rope_get_truncate(op)) {
        return false;
    }
    // only the nodes that the server accepts (rpc_node_ok, see graph_compute): the scheduler runs the others on another
    // backend, e.g. the custom operations, whose function only exists in this process
    if (!rpc_node_ok(op)) {
        return false;
    }
    if (ggml_op_is_empty(op->op)) {
        return true;
    }
    // and, if the server answers it (an older one does not), whether its backend runs the node: graph_compute rejects
    // the nodes that it does not, which closed the connection (llamacppOpenCL11)
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *) dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    if ((dispatcher->features() & RPC_FEATURE_SUPPORTS_OP) == 0) {
        return true;
    }
    return rpc_remote_supports_op(dispatcher, ctx, op);
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static ggml_backend_event_t ggml_backend_rpc_device_event_new(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    return dispatcher->event_new(dev);
}

static void ggml_backend_rpc_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_free(event);
}

static void ggml_backend_rpc_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;
    auto dispatcher = get_dispatcher(ctx->endpoint);
    dispatcher->event_synchronize(event);
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ ggml_backend_rpc_device_event_new,
    /* .event_free           = */ ggml_backend_rpc_device_event_free,
    /* .event_synchronize    = */ ggml_backend_rpc_device_event_synchronize,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Pairwise allreduce between two RPC servers over a direct server-to-server connection.
// The client only sends fire-and-forget COMM_ALLREDUCE commands; the tensor data is
// exchanged between the servers and never passes through the client.
struct ggml_backend_rpc_comm_context {
    struct rank_info {
        std::string                     endpoint;
        uint32_t                        device;
        std::shared_ptr<rpc_dispatcher> dispatcher;
    };
    std::vector<rank_info> ranks;
};

static void ggml_backend_rpc_comm_free(void * comm_ctx_v) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return;
    }
    for (const auto & rank : comm_ctx->ranks) {
        auto request = std::make_shared<rpc_msg_comm_free_req>();
        request->device = rank.device;
        rank.dispatcher->send(RPC_CMD_COMM_FREE, request, sizeof(*request));
        rank.dispatcher->busy_spin_release();
    }
    delete comm_ctx;
}

static void * ggml_backend_rpc_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || std::getenv("GGML_RPC_NO_COMM") != nullptr) {
        return nullptr;
    }
    std::vector<ggml_backend_rpc_comm_context::rank_info> ranks;
    ranks.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_rpc(backends[i])) {
            return nullptr;
        }
        ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *) backends[i]->context;
        // one rank per endpoint: a server processes its socket sequentially, so a second
        // COMM_INIT on the same connection would deadlock behind the first
        for (const auto & rank : ranks) {
            if (rank.endpoint == rpc_ctx->endpoint) {
                LOG_WARN("%s: multiple ranks on endpoint %s are not supported\n", __func__, rpc_ctx->endpoint.c_str());
                return nullptr;
            }
        }
        ranks.push_back({rpc_ctx->endpoint, rpc_ctx->device, rpc_ctx->dispatcher});
    }
    for (const auto & vRank : ranks) {
      if (vRank.dispatcher->features() & RPC_FEATURE_COMM_DISABLED) {
        LOG_WARN("%s: server %s has not enabled GGML_RPC_COMM_ENDPOINT\n", __func__, vRank.endpoint.c_str());
        return nullptr;
      }
    }

    // llamacppOpenCL11: the servers authenticate each other only if they have a token (rpc_comm_auth): all of them or
    // none, or the one with the token would wait forever for the other one
    for (size_t i = 1; i < ranks.size(); i++) {
        if (((ranks[i].dispatcher->features() ^ ranks[0].dispatcher->features()) & RPC_FEATURE_AUTH) != 0) {
            LOG_WARN("%s: %s and %s do not both require a token (GGML_RPC_TOKEN), no communicator between them\n",
                     __func__, ranks[0].endpoint.c_str(), ranks[i].endpoint.c_str());
            return nullptr;
        }
    }

    // TODO: simplify this logic
    // rank 1 connects to rank 0 on its serving host; endpoints must be mutually reachable
    // (e.g. do not bind the servers to 127.0.0.1 when they run on different machines)
    std::string host0;
    int port0;
    if (!parse_endpoint(ranks[0].endpoint, host0, port0)) {
        return nullptr;
    }
    const uint32_t comm_port = (uint32_t) port0 + 1000;
    if (host0.size() >= 64 || port0 < 1 || port0 > 64535) {
        return nullptr;
    }

    for (const auto & rank : ranks) {
        rank.dispatcher->busy_spin_acquire();
    }

    // Send all init requests before reading any response: rank 0 blocks in accept
    // until rank 1 has connected.
    std::vector<rpc_msg_comm_init_rsp> responses(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        auto request = std::make_shared<rpc_msg_comm_init_req>();
        request->device = ranks[i].device;
        request->rank   = (uint32_t) i;
        request->world  = (uint32_t) n_backends;
        request->port   = comm_port;
        if (i > 0) {
            memcpy(request->host, host0.c_str(), host0.size());
        }
        ranks[i].dispatcher->send_async(RPC_CMD_COMM_INIT, request, sizeof(*request), &responses[i], sizeof(responses[i]));
    }
    for (size_t i = 0; i < n_backends; i++) {
        ranks[i].dispatcher->synchronize();
    }
    bool ok = true;
    for (size_t i = 0; i < n_backends; i++) {
        if (!responses[i].ok) {
            LOG_WARN("%s: rank %zu (%s) failed to initialize\n", __func__, i, ranks[i].endpoint.c_str());
            ok = false;
        }
    }
    if (!ok) {
        for (const auto & rank : ranks) {
            rank.dispatcher->busy_spin_release();
        }
        return nullptr;
    }
    LOG_INFO("%s: pairwise communicator initialized (%s <-> %s)\n", __func__,
             ranks[0].endpoint.c_str(), ranks[1].endpoint.c_str());
    return new ggml_backend_rpc_comm_context{std::move(ranks)};
}

static bool ggml_backend_rpc_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    ggml_backend_rpc_comm_context * comm_ctx = (ggml_backend_rpc_comm_context *) comm_ctx_v;
    if (comm_ctx == nullptr) {
        return false;
    }
    const size_t n_ranks = comm_ctx->ranks.size();
    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }
    for (size_t i = 0; i < n_ranks; i++) {
        if (tensors[i] == nullptr || tensors[i]->type != GGML_TYPE_F32 || ggml_nelements(tensors[i]) != ne ||
                !ggml_is_contiguously_allocated(tensors[i]) ||
                tensors[i]->buffer == nullptr || !ggml_backend_buffer_is_rpc(tensors[i]->buffer)) {
            return false;
        }
        // a rank with a disabled node has garbage in its partial and must contribute zeros,
        // which only the fallback path handles
        if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return false;
        }
    }
    for (size_t i = 0; i < n_ranks; i++) {
        auto request = std::make_shared<rpc_msg_comm_allreduce_req>();
        request->device = comm_ctx->ranks[i].device;
        request->tensor = serialize_tensor(tensors[i]);
        comm_ctx->ranks[i].dispatcher->send_async(RPC_CMD_COMM_ALLREDUCE, request, sizeof(*request));
    }
    return true;
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_rpc_comm_init;
    }
    if (std::strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_rpc_comm_free;
    }
    if (std::strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_rpc_comm_allreduce_tensor;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto dispatcher = get_dispatcher(endpoint);
    rpc_msg_device_count_rsp response;
    dispatcher->send(RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        LOG_DBG3("[%s] endpoint %s already registered\n", __func__, endpoint);
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    LOG_DBG2("[%s] endpoint: %s, device count: %u\n", __func__, endpoint, dev_count);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */    endpoint,
            /* .device      = */    ind,
            /* .name        = */    dev_name,
            /* .description = */    dev_desc,
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
