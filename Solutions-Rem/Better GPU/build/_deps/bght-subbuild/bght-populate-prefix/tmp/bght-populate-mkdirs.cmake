# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-src"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-build"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/tmp"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/src/bght-populate-stamp"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/src"
  "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/src/bght-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/src/bght-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/users/mtech/bkeshav24/ACP/Solutions/Better GPU/build/_deps/bght-subbuild/bght-populate-prefix/src/bght-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
