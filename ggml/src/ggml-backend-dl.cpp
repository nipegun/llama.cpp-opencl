#include "ggml-backend-dl.h"

#ifdef _WIN32

// Windows 8, or 7 with KB2533623 (the values of libloaderapi.h, for older headers)
#ifndef LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR
#    define LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR 0x00000100
#endif
#ifndef LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
#    define LOAD_LIBRARY_SEARCH_DEFAULT_DIRS 0x00001000
#endif

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    // llamacppOpenCL11: the DLLs that the backend needs are searched in its own directory, in the directory of the
    // program, in System32 and in the directories of AddDllDirectory, not in the current directory or in PATH, where
    // another user could leave a DLL with the name of one of them (LoadLibraryW searches them). A backend that needs a
    // DLL from another place (PATH) does not load: copy that DLL next to it. LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR needs an
    // absolute path.
    std::error_code ec;
    const fs::path abs_path = fs::absolute(path, ec);
    HMODULE handle = LoadLibraryExW((ec ? path : abs_path).wstring().c_str(), NULL,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
