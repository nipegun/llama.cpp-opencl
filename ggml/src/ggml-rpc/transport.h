#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);
    // Must be called at every message boundary: the RDMA transport coalesces
    // writes into fixed-size frames and posts the trailing partial frame only
    // here. No-op on TCP.
    bool flush();

    // llamacppOpenCL11: shuts down both directions of the connection (the socket stays open until it is destroyed),
    // so that a receive or a send that waits on it, also in another thread, fails (the deadline of a handshake)
    void shutdown();

    // peer_addr (llamacppOpenCL11): if not null, the IPv4 address of the client (network byte order)
    // A nonnegative timeout bounds the wait for an incoming communicator peer.
    socket_ptr accept(uint32_t * peer_addr = nullptr, int pTimeoutMs = -1);

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);
    const char * transport_name() const;

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

private:
    struct impl;
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
