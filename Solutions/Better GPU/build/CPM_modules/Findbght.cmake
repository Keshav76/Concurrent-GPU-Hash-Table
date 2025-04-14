include("/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/cmake/CPM.cmake")
CPMAddPackage(NAME;bght;GITHUB_REPOSITORY;owensgroup/BGHT;GIT_TAG;main;OPTIONS;build_tests OFF;build_benchmarks OFF;build_examples OFF)
set(bght_FOUND TRUE)