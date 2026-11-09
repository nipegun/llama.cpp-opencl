# Security Policy

 - [**Reporting a vulnerability**](#reporting-a-vulnerability)
 - [**Covered topics**](#covered-topics)
 - [**Using llama.cpp-opencl securely**](#using-llamacpp-opencl-securely)
   - [Untrusted models](#untrusted-models)
   - [Untrusted inputs](#untrusted-inputs)
   - [The RPC server](#the-rpc-server)
   - [llama-server](#llama-server)
   - [Backends and libraries](#backends-and-libraries)
   - [Downloads from Hugging Face](#downloads-from-hugging-face)
   - [Data privacy and shared computers](#data-privacy-and-shared-computers)

## Reporting a vulnerability

If you find a security vulnerability in `llama.cpp-opencl` that falls inside the [covered topics](#covered-topics), report it privately as a [security advisory](https://github.com/nipegun/llama.cpp-opencl/security/advisories/new). **Do not open a public issue for it.** Include the date of the code that you tested, the steps or the files that reproduce the problem and, if you can, a proof of concept.

This fork is maintained by one person, on a reasonable-effort basis: please allow up to 90 days for a fix before you make the problem public.

Most of the code comes from [llama.cpp](https://github.com/ggml-org/llama.cpp). If the problem is in code that this fork did not change, it is probably in the official repository too: report it to them as well, following [their security policy](https://github.com/ggml-org/llama.cpp/security/policy).

## Covered topics

 - The OpenCL backend and its kernels (`ggml/src/ggml-opencl/`).
 - The RPC backend and `ggml-rpc-server` (`ggml/src/ggml-rpc/`, `tools/rpc/`).
 - The loading of models and backends (`src/`, `ggml/src/`, `gguf-py/`).
 - `llama-server` (`tools/server/`), except its web UI, the features marked as experimental and those that are not meant for untrusted environments (for example, the router and MCP).

The topics of [Using llama.cpp-opencl securely](#using-llamacpp-opencl-securely) are not vulnerabilities: they are the conditions under which this software is safe to use. Denial of service bugs (a crash or a hang caused by a client or by a file) are looked at case by case.

For vulnerabilities in the `vendor/` folder, report them to the third-party project.

## Using llama.cpp-opencl securely

### Untrusted models

A GGUF file is a complex input. Be careful with models from unknown sources, and run them in an isolated environment (a container or a virtual machine). Check that the hash of a downloaded model matches a known-good value.

The Python GGUF reader rejects incomplete primitive reads and checks array payload sizes against the remaining file before creating per-element objects. These checks prevent truncated metadata from causing a non-advancing allocation loop; they are not a total memory budget for a large, structurally valid file.

The trustworthiness of a model is not binary: choose the level of caution that fits the model, your use and the risk you can accept.

### Untrusted inputs

A model can be led by its input to do or say what its user did not intend (prompt injection). If the input comes from people or programs that you do not trust:

 - Isolate the environment where the inference runs.
 - Check how the model behaves with that kind of input before you rely on it.
 - Validate and filter the input, and treat the output of the model as untrusted too.
 - Keep `llama.cpp-opencl` and the libraries that it uses up to date.

### The RPC server

`ggml-rpc-server` runs the operations that its clients send, on its GPU or on its processor, in buffers that its clients allocate. It checks every graph that it receives (the operations, the shapes and the types of the tensors, and that every tensor that it computes is in a buffer of the same connection), and it only runs the operations that its device supports. But it trusts its clients with the memory and the time of its device:

 - By default it listens only on `127.0.0.1`. Before you bind it to another address, set `GGML_RPC_TOKEN` on the server and the same value on its clients: the server then serves only the clients that know the token. The token does not travel over the network: the client answers a random challenge with an HMAC-SHA256 made with it.
 - Choose a long random token (for example, `openssl rand -hex 32`), and keep it out of the shell history, the logs and the files that other users can read.
 - The traffic is not encrypted: anyone who can see it can read the prompts, the answers and the weights, and anyone who can change it can take over a connection after its authentication. Use the RPC server only on a trusted network, or through an SSH tunnel or a VPN.
 - A client with the token can use all of the memory of the device and keep the server busy, like any other user of the GPU.
 - The server serves one client at a time. A new connection has 10 seconds to say hello and, with a token, to authenticate, and it waits for its turn after that; the server keeps at most 16 connections waiting (at most 4 of them still authenticating from the same address) and closes the others at once.
 - The server clears every buffer before a client gets it, so a client cannot read what an earlier client left in the memory of the device.
 - Server-to-server networking for tensor split (`-sm tensor`, experimental) is disabled by default, including for authenticated clients. To enable it, set `GGML_RPC_COMM_ENDPOINT` on every server to the same rank-zero literal IPv4 address and port (the first RPC endpoint used by the client, with its port plus 1000). The server rejects other client-selected endpoints. Rank zero binds only that address; other ranks connect only to it, without resolving a client-supplied name. With a token, the servers authenticate each other with it: all of them need the same token, or none.
 - RPC control frames are bounded before allocation: graph frames are limited to 64 MiB and operation-support queries to their protocol maximum. Tensor uploads validate the destination and complete span first, then receive at most 16 MiB at a time; the tensor itself can be larger. These bounds do not limit the device memory or total work that a valid client can request.

### llama-server

 - By default it listens only on `127.0.0.1`, and it lets only the pages of `localhost` (and its own web UI, at any of its addresses) use its API from a browser (`--cors-origins localhost`). The browser does not let other web sites read its answers, and the server rejects the requests that a browser sends to it from them without asking first (form posts). To use the API from a web application on another address, add that address to `--cors-origins` (`*` allows every web site, as in the official llama.cpp).
 - Every route also validates `Host`, even when the request has no `Origin`. Only localhost, the local socket address and explicitly configured server names are accepted. Add public names with `--allowed-hosts` (or `LLAMA_ARG_ALLOWED_HOSTS`); a wildcard listening address does not allow arbitrary DNS names. Matching attacker-controlled `Host` and `Origin` headers alone never grant access.
 - This fork only supports text models. Reranking image input is rejected before reading or downloading media, including query and document images. Adding multimedia support later requires an explicit policy for request-triggered remote downloads.
 - Before you make it reachable from other computers, set an API key (`--api-key`), use HTTPS, and put it behind a reverse proxy that limits the number of connections and the time to send a request: the server gives every connection an HTTP thread, and a client that sends its request very slowly keeps one busy for up to `--timeout` seconds (3600 by default) between two pieces.

### Backends and libraries

The programs load the backends (`libggml-*.so`, or `ggml-*.dll` on Windows) only from their own folder (and from `GGML_BACKEND_DIR` if it was set at build time), never from the folder they are started from. Keep the folder of the programs writable only by its owner (`/usr/local/bin` once installed).

On Windows, the libraries that a backend needs are searched in the folder of the backend, in the folder of the program and in the folders of the system, not in the current folder or in `PATH`: if a backend does not load, copy the libraries that it needs next to it.

### Downloads from Hugging Face

The Hugging Face token (`-hft`, or `HF_TOKEN`) is only sent to the model endpoint (`MODEL_ENDPOINT`, `HF_ENDPOINT`, or `https://huggingface.co/`), never to the other addresses that a download goes to.

### Data privacy and shared computers

To keep the data of different users apart, run the models of each one in an isolated environment, with its own network. On a shared computer:

 - A model or a user can use up the memory or the time of the GPU, and slow down or stop the others: limit what each one can use.
 - GPUs can leak data between the processes that use them at the same time, through side channels.
 - A model from someone else (its weights and its chat template) is an untrusted input: isolate its execution.
