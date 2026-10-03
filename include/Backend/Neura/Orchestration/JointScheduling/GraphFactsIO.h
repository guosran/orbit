//===- GraphFactsIO.h ------------------------------------------*- C++ -*-===//
//
// Lossless storage helpers for persisted ORBIT graph-facts JSON.
//
// Legacy artifacts are plain JSON bytes.  Compressed artifacts use a small
// versioned header followed by an LLVM zlib or zstd payload.  The JSON text
// itself is never transformed, so structural keys and all evidence fields
// retain their existing byte-for-byte meaning after decode.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_NEURA_JOINT_SCHEDULING_GRAPH_FACTS_IO_H
#define AMOEBA_NEURA_JOINT_SCHEDULING_GRAPH_FACTS_IO_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compression.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>

namespace mlir::amoeba::neura::joint_scheduling::graph_facts_io {

using llvm::ArrayRef;
using llvm::SmallVectorImpl;
using llvm::StringRef;

// The magic includes the storage format version.  Do not reuse this prefix
// for another binary artifact: the reader treats it as a compressed-facts
// record and fails closed on malformed headers or payloads.
inline constexpr char kCompressedFactsMagic[] = "ORBITGF1";
inline constexpr size_t kCompressedFactsMagicSize = 8;
inline constexpr size_t kCompressedFactsHeaderSize = 32;

// Header layout, all integer fields little-endian:
//   [0, 8)   kCompressedFactsMagic
//   [8]      codec (1 = zlib, 2 = zstd)
//   [9]      flags (currently zero)
//   [10,16) reserved (currently zero)
//   [16,24) uncompressed JSON byte length
//   [24,32) compressed payload byte length
//   [32,... ) compressed payload
// The explicit payload length rejects trailing bytes as well as truncation.
enum class FactsCompression : uint8_t {
  LegacyJson = 0,
  Zlib = 1,
  Zstd = 2,
};

inline StringRef compressionName(FactsCompression compression) {
  switch (compression) {
  case FactsCompression::LegacyJson:
    return "legacy-json";
  case FactsCompression::Zlib:
    return "zlib";
  case FactsCompression::Zstd:
    return "zstd";
  }
  return "unknown";
}

inline bool hasMagicPrefix(ArrayRef<uint8_t> bytes) {
  const size_t prefixSize = bytes.size() < kCompressedFactsMagicSize
                                ? bytes.size()
                                : kCompressedFactsMagicSize;
  return prefixSize != 0 &&
         std::memcmp(bytes.data(), kCompressedFactsMagic, prefixSize) == 0;
}

inline bool isCompressedFacts(ArrayRef<uint8_t> bytes) {
  return bytes.size() >= kCompressedFactsMagicSize &&
         std::memcmp(bytes.data(), kCompressedFactsMagic,
                     kCompressedFactsMagicSize) == 0;
}

inline bool encodeGraphFacts(StringRef jsonText, FactsCompression compression,
                             SmallVectorImpl<uint8_t> &output,
                             std::string &error) {
  output.clear();
  if (compression == FactsCompression::LegacyJson) {
    output.append(reinterpret_cast<const uint8_t *>(jsonText.data()),
                  reinterpret_cast<const uint8_t *>(jsonText.data()) +
                      jsonText.size());
    return true;
  }

  llvm::SmallVector<uint8_t, 0> compressed;
  switch (compression) {
  case FactsCompression::Zlib:
    if (!llvm::compression::zlib::isAvailable()) {
      error = "LLVM zlib support is unavailable";
      return false;
    }
    llvm::compression::zlib::compress(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(jsonText.data()),
                          jsonText.size()),
        compressed);
    break;
  case FactsCompression::Zstd:
    if (!llvm::compression::zstd::isAvailable()) {
      error = "LLVM zstd support is unavailable";
      return false;
    }
    llvm::compression::zstd::compress(
        ArrayRef<uint8_t>(reinterpret_cast<const uint8_t *>(jsonText.data()),
                          jsonText.size()),
        compressed);
    break;
  case FactsCompression::LegacyJson:
    llvm_unreachable("legacy JSON handled above");
  }

  if (jsonText.size() > std::numeric_limits<uint64_t>::max() ||
      compressed.size() > std::numeric_limits<uint64_t>::max()) {
    error = "graph facts payload exceeds binary header size";
    return false;
  }

  output.resize(kCompressedFactsHeaderSize);
  std::memcpy(output.data(), kCompressedFactsMagic,
              kCompressedFactsMagicSize);
  output[8] = static_cast<uint8_t>(compression);
  output[9] = 0;
  std::memset(output.data() + 10, 0, 6);
  llvm::support::endian::write64le(output.data() + 16,
                                   static_cast<uint64_t>(jsonText.size()));
  llvm::support::endian::write64le(output.data() + 24,
                                   static_cast<uint64_t>(compressed.size()));
  output.append(compressed.begin(), compressed.end());
  return true;
}

inline bool decodeGraphFacts(StringRef bytes, std::string &jsonText,
                             std::string &error,
                             FactsCompression *detectedCompression = nullptr) {
  const auto *data = reinterpret_cast<const uint8_t *>(bytes.data());
  ArrayRef<uint8_t> encoded(data, bytes.size());
  if (!isCompressedFacts(encoded)) {
    if (hasMagicPrefix(encoded)) {
      error = "truncated graph facts compression magic";
      return false;
    }
    // Preserve legacy JSON bytes exactly.  JSON parsing remains the caller's
    // responsibility, just as it was before compressed storage existed.
    jsonText.assign(bytes.data(), bytes.size());
    if (detectedCompression)
      *detectedCompression = FactsCompression::LegacyJson;
    return true;
  }

  if (encoded.size() < kCompressedFactsHeaderSize) {
    error = "compressed graph facts header is truncated";
    return false;
  }
  if (encoded[9] != 0 ||
      std::memcmp(encoded.data() + 10, "\0\0\0\0\0\0", 6) != 0) {
    error = "compressed graph facts header has unknown flags";
    return false;
  }

  FactsCompression compression;
  switch (encoded[8]) {
  case static_cast<uint8_t>(FactsCompression::Zlib):
    compression = FactsCompression::Zlib;
    break;
  case static_cast<uint8_t>(FactsCompression::Zstd):
    compression = FactsCompression::Zstd;
    break;
  default:
    error = "compressed graph facts header has unknown codec";
    return false;
  }

  const uint64_t uncompressedSize =
      llvm::support::endian::read64le(encoded.data() + 16);
  const uint64_t compressedSize =
      llvm::support::endian::read64le(encoded.data() + 24);
  if (uncompressedSize > std::numeric_limits<size_t>::max() ||
      compressedSize > std::numeric_limits<size_t>::max()) {
    error = "compressed graph facts length exceeds host size_t";
    return false;
  }
  if (compressedSize != encoded.size() - kCompressedFactsHeaderSize) {
    error = "compressed graph facts payload length does not match header";
    return false;
  }

  ArrayRef<uint8_t> payload(encoded.data() + kCompressedFactsHeaderSize,
                            static_cast<size_t>(compressedSize));
  llvm::SmallVector<uint8_t, 0> decoded;
  llvm::Error decompressError = [&]() -> llvm::Error {
    if (compression == FactsCompression::Zlib) {
      if (!llvm::compression::zlib::isAvailable()) {
        error = "LLVM zlib support is unavailable";
        return llvm::make_error<llvm::StringError>(
            error, llvm::inconvertibleErrorCode());
      }
      return llvm::compression::zlib::decompress(
          payload, decoded, static_cast<size_t>(uncompressedSize));
    }
    if (!llvm::compression::zstd::isAvailable()) {
      error = "LLVM zstd support is unavailable";
      return llvm::make_error<llvm::StringError>(
          error, llvm::inconvertibleErrorCode());
    }
    return llvm::compression::zstd::decompress(
        payload, decoded, static_cast<size_t>(uncompressedSize));
  }();
  if (decompressError) {
    error = "compressed graph facts decompression failed: " +
            llvm::toString(std::move(decompressError));
    return false;
  }
  if (decoded.size() != static_cast<size_t>(uncompressedSize)) {
    error = "compressed graph facts decoded length does not match header";
    return false;
  }
  jsonText.assign(reinterpret_cast<const char *>(decoded.data()),
                  decoded.size());
  if (detectedCompression)
    *detectedCompression = compression;
  return true;
}

inline bool readGraphFactsFile(StringRef path, std::string &jsonText,
                               std::string &error,
                               FactsCompression *detectedCompression = nullptr) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read graph facts file " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  StringRef bytes = (*buffer)->getBuffer();
  return decodeGraphFacts(bytes, jsonText, error, detectedCompression);
}

} // namespace mlir::amoeba::neura::joint_scheduling::graph_facts_io

#endif // AMOEBA_NEURA_JOINT_SCHEDULING_GRAPH_FACTS_IO_H
