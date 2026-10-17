// export_dataset.cpp — Live2D autogrid entry A: UV-AABB crop per ArtMesh → PNG+JSON
// Built against Live2DReforge libs (read-only); lives under live2d-autogrid/_tools.
//
// usage:
//   export_dataset <in.cmo3> <out_dir> [--pad N] [--list N]
//
#include "core/Cmo3Document.h"
#include "io/PngCodec.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace {

bool fieldIs(const l2d::XsNode& n, const std::string& f) {
    if (n.fieldName == f) return true;
    for (const auto& a : n.attrs) {
        if (a.first == "xs.n" && a.second == f) return true;
    }
    return false;
}

const l2d::XsNode* directField(const l2d::XsNode& n, const std::string& f) {
    for (const auto& c : n.children) {
        if (fieldIs(c, f)) return &c;
    }
    return nullptr;
}

void parseFloats(const std::string& t, std::vector<float>& out) {
    const char* p = t.c_str();
    while (*p != 0) {
        while (*p != 0 && static_cast<unsigned char>(*p) <= 32) ++p;
        if (*p == 0) break;
        char* e = nullptr;
        const float v = std::strtof(p, &e);
        if (e == p) break;
        out.push_back(v);
        p = e;
    }
}

void parseInts(const std::string& t, std::vector<int>& out) {
    const char* p = t.c_str();
    while (*p != 0) {
        while (*p != 0 && static_cast<unsigned char>(*p) <= 32) ++p;
        if (*p == 0) break;
        char* e = nullptr;
        const long v = std::strtol(p, &e, 10);
        if (e == p) break;
        out.push_back(static_cast<int>(v));
        p = e;
    }
}

bool readFloatField(const l2d::XsDocument& doc, const l2d::XsNode& holder, const std::string& f,
                    std::vector<float>& out) {
    const l2d::XsNode* arr = directField(holder, f);
    if (arr == nullptr) return false;
    const l2d::XsNode* def = arr;
    if (arr->ref >= 0) {
        def = doc.resolve(arr->ref);
        if (def == nullptr) return false;
    }
    parseFloats(def->text, out);
    return true;
}

bool readIntField(const l2d::XsDocument& doc, const l2d::XsNode& holder, const std::string& f,
                  std::vector<int>& out) {
    const l2d::XsNode* arr = directField(holder, f);
    if (arr == nullptr) return false;
    const l2d::XsNode* def = arr;
    if (arr->ref >= 0) {
        def = doc.resolve(arr->ref);
        if (def == nullptr) return false;
    }
    parseInts(def->text, out);
    return true;
}

std::string jsonEscape(const std::string& s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o.push_back('\\'); o.push_back(static_cast<char>(c)); }
        else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            o += buf;
        } else {
            o.push_back(static_cast<char>(c));
        }
    }
    return o;
}

std::string toLowerAscii(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Best-effort part vocabulary from mesh display name.
std::string guessPart(const std::string& name) {
    const std::string n = toLowerAscii(name);
    auto has = [&](const char* k) { return n.find(k) != std::string::npos; };
    // hair
    if (has("hair") || has("bangs") || has("bang") || has("ahoge") || has("twin") ||
        has("ponytail") || has("\xe9\xab\xaa") /*髪*/ || has("\xe5\x8f\x91") /*发*/ ||
        has("\xe3\x81\xbe\xe3\x81\xa4\xe6\xaf\x9b") || has("eyelash") || has("brow") ||
        has("eyebrow") || has("\xe7\x9c\x89") /*眉*/)
        return "hair";
    // face
    if (has("face") || has("eye") || has("mouth") || has("nose") || has("cheek") ||
        has("lip") || has("tooth") || has("tongue") || has("\xe9\xa1\x94") /*顔*/ ||
        has("\xe7\x9b\xae") /*目*/ || has("\xe5\x8f\xa3") /*口*/ || has("\xe9\xbc\xbb") /*鼻*/ ||
        has("head") || has("\xe9\xa0\xad"))
        return "face";
    // cloth
    if (has("cloth") || has("dress") || has("skirt") || has("shirt") || has("pants") ||
        has("sleeve") || has("collar") || has("uniform") || has("\xe6\x9c\x8d") /*服*/ ||
        has("\xe8\xa3\x99") /*裙*/ || has("jacket") || has("coat"))
        return "cloth";
    // body
    if (has("body") || has("arm") || has("leg") || has("hand") || has("foot") || has("neck") ||
        has("shoulder") || has("chest") || has("waist") || has("\xe4\xbd\x93") /*体*/ ||
        has("\xe8\x85\xb9") /*腕*/ || has("\xe8\xb6\xb3") /*足*/ || has("\xe6\x89\x8b") /*手*/ ||
        has("\xe9\xa0\x96") /*首*/)
        return "body";
    // acc
    if (has("acc") || has("ribbon") || has("hat") || has("cap") || has("ear") || has("horn") ||
        has("wing") || has("tail") || has("glass") || has("bow") || has("jewel") ||
        has("\xe9\xa3\xbe") /*飾*/ || has("\xe5\xb8\xbd") /*帽*/ || has("\xe8\x80\xb3") /*耳*/)
        return "acc";
    return "";
}

std::string sanitizeId(const std::string& name, int xsId) {
    std::string s;
    s.reserve(name.size() + 16);
    for (unsigned char c : name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
            c == '_' || c == '-') {
            s.push_back(static_cast<char>(c));
        } else if (c >= 0x80) {
            // keep UTF-8 bytes as hex-escaped short form via underscore skip;
            // for filesystem safety on Windows, drop non-ascii from filename
            // and rely on xsId uniqueness.
            continue;
        } else if (c == ' ' || c == '.') {
            s.push_back('_');
        }
    }
    while (!s.empty() && (s.back() == '_' || s.back() == '-')) s.pop_back();
    if (s.empty()) s = "mesh";
    if (s.size() > 48) s.resize(48);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "_%d", xsId);
    return s + buf;
}

bool writeFileBytes(const fs::path& path, const std::vector<uint8_t>& bytes, std::string& err) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { err = "cannot open for write: " + path.string(); return false; }
    f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!f) { err = "write failed: " + path.string(); return false; }
    return true;
}

void usage() {
    std::printf("usage:\n");
    std::printf("  export_dataset <in.cmo3> <out_dir> [--pad N] [--list N]\n");
    std::printf("    --pad N   UV AABB padding in pixels (default 2)\n");
    std::printf("    --list N  export only first N meshes (0 = all, default 0)\n");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { usage(); return 2; }
    const std::string inPath = argv[1];
    const fs::path outDir = argv[2];
    int pad = 2;
    int listN = 0;
    for (int i = 3; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pad") == 0 && i + 1 < argc) pad = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--list") == 0 && i + 1 < argc) listN = std::atoi(argv[++i]);
        else { std::printf("unknown arg: %s\n", argv[i]); usage(); return 2; }
    }
    if (pad < 0) pad = 0;

    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        std::printf("ERROR: create_directories: %s\n", ec.message().c_str());
        return 1;
    }

    l2d::Cmo3Document doc;
    std::string err;
    if (!doc.load(inPath, err)) {
        std::printf("ERROR: load: %s\n", err.c_str());
        return 1;
    }
    const l2d::XsDocument& xml = doc.xml();
    const auto& texs = doc.textures();

    int cachedTex = -1;
    std::vector<uint8_t> rgba;
    int cachedW = 0, cachedH = 0;

    int written = 0, skipped = 0;
    std::unordered_set<std::string> usedIds;

    for (const l2d::ArtMesh& am : doc.artMeshes()) {
        if (listN > 0 && written >= listN) break;
        if (am.textureIndex < 0 || am.sourceXsId < 0) { ++skipped; continue; }
        if (am.textureIndex >= static_cast<int>(texs.size())) { ++skipped; continue; }

        const l2d::XsNode* mesh = xml.resolve(am.sourceXsId);
        if (mesh == nullptr) { ++skipped; continue; }

        std::vector<float> uvs;
        if (!readFloatField(xml, *mesh, "uvs", uvs) || uvs.size() < 6) { ++skipped; continue; }

        if (am.textureIndex != cachedTex) {
            std::vector<uint8_t> pageBytes;
            if (!doc.extractTexture(am.textureIndex, pageBytes, err)) {
                std::printf("WARN: extractTexture[%d]: %s\n", am.textureIndex, err.c_str());
                ++skipped;
                continue;
            }
            l2d::PngDecodeResult dec;
            rgba.clear();
            if (!l2d::pngDecodeToRgba(pageBytes.data(), pageBytes.size(), rgba, &dec, err)) {
                std::printf("WARN: png decode tex[%d]: %s\n", am.textureIndex, err.c_str());
                ++skipped;
                continue;
            }
            cachedTex = am.textureIndex;
            cachedW = dec.width;
            cachedH = dec.height;
        }

        double minU = 10.0, maxU = -10.0, minV = 10.0, maxV = -10.0;
        for (size_t i = 0; i + 1 < uvs.size(); i += 2) {
            minU = std::min(minU, static_cast<double>(uvs[i]));
            maxU = std::max(maxU, static_cast<double>(uvs[i]));
            minV = std::min(minV, static_cast<double>(uvs[i + 1]));
            maxV = std::max(maxV, static_cast<double>(uvs[i + 1]));
        }

        const int rx0 = std::max(0, static_cast<int>(std::floor(minU * cachedW)) - pad);
        const int ry0 = std::max(0, static_cast<int>(std::floor(minV * cachedH)) - pad);
        const int rx1 = std::min(cachedW, static_cast<int>(std::ceil(maxU * cachedW)) + pad);
        const int ry1 = std::min(cachedH, static_cast<int>(std::ceil(maxV * cachedH)) + pad);
        if (rx1 <= rx0 || ry1 <= ry0) { ++skipped; continue; }

        const int cw = rx1 - rx0;
        const int ch = ry1 - ry0;
        std::vector<uint8_t> crop(static_cast<size_t>(cw) * static_cast<size_t>(ch) * 4);
        for (int y = 0; y < ch; ++y) {
            const uint8_t* src = rgba.data() + (static_cast<size_t>(ry0 + y) * cachedW + rx0) * 4;
            uint8_t* dst = crop.data() + static_cast<size_t>(y) * cw * 4;
            std::memcpy(dst, src, static_cast<size_t>(cw) * 4);
        }

        l2d::PngEncodeOptions encOpt;
        encOpt.level = 6;
        std::vector<uint8_t> pngOut;
        l2d::PngEncodeResult encRes;
        if (!l2d::pngEncodeRgba(crop.data(), cw, ch, encOpt, pngOut, &encRes, err)) {
            std::printf("WARN: png encode mesh#%d: %s\n", am.sourceXsId, err.c_str());
            ++skipped;
            continue;
        }

        const std::string name = am.displayName();
        std::string partId = sanitizeId(name, am.sourceXsId);
        if (usedIds.count(partId)) {
            char buf[48];
            std::snprintf(buf, sizeof(buf), "_dup%d", written);
            partId += buf;
        }
        usedIds.insert(partId);

        // vertices in local PNG coords (origin top-left)
        std::vector<double> localVerts;
        localVerts.reserve(uvs.size());
        for (size_t i = 0; i + 1 < uvs.size(); i += 2) {
            const double lx = static_cast<double>(uvs[i]) * cachedW - rx0;
            const double ly = static_cast<double>(uvs[i + 1]) * cachedH - ry0;
            localVerts.push_back(lx);
            localVerts.push_back(ly);
        }

        std::vector<int> indices;
        readIntField(xml, *mesh, "indices", indices);

        const std::string part = guessPart(name);
        const std::string texName = (am.textureIndex >= 0 && am.textureIndex < static_cast<int>(texs.size()))
                                        ? texs[static_cast<size_t>(am.textureIndex)].atlasName
                                        : "";
        char texLabel[64];
        if (!texName.empty())
            std::snprintf(texLabel, sizeof(texLabel), "%s", texName.c_str());
        else
            std::snprintf(texLabel, sizeof(texLabel), "tex_%d", am.textureIndex);

        // JSON
        std::string json;
        json.reserve(256 + localVerts.size() * 12 + indices.size() * 8);
        json += "{\n";
        json += "  \"id\": \"" + jsonEscape(partId) + "\",\n";
        json += "  \"texture\": \"" + jsonEscape(texLabel) + "\",\n";
        json += "  \"size\": [" + std::to_string(cw) + ", " + std::to_string(ch) + "],\n";
        json += "  \"part\": \"" + jsonEscape(part) + "\",\n";
        json += "  \"vertices\": [";
        for (size_t i = 0; i < localVerts.size(); ++i) {
            if (i) json += ", ";
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%.4f", localVerts[i]);
            json += buf;
        }
        json += "],\n";
        json += "  \"triangles\": [";
        for (size_t i = 0; i < indices.size(); ++i) {
            if (i) json += ", ";
            json += std::to_string(indices[i]);
        }
        json += "],\n";
        json += "  \"notes\": {";
        json += "\"meshXsId\": " + std::to_string(am.sourceXsId);
        json += ", \"name\": \"" + jsonEscape(name) + "\"";
        json += ", \"page\": [" + std::to_string(cachedW) + ", " + std::to_string(cachedH) + "]";
        json += ", \"aabb\": [" + std::to_string(rx0) + ", " + std::to_string(ry0) + ", " +
                std::to_string(rx1) + ", " + std::to_string(ry1) + "]";
        json += ", \"pad\": " + std::to_string(pad);
        json += ", \"entry\": \"A\"";
        json += "}\n";
        json += "}\n";

        const fs::path pngPath = outDir / (partId + ".png");
        const fs::path jsonPath = outDir / (partId + ".json");
        if (!writeFileBytes(pngPath, pngOut, err)) {
            std::printf("ERROR: %s\n", err.c_str());
            return 1;
        }
        {
            std::ofstream jf(jsonPath, std::ios::binary);
            if (!jf) {
                std::printf("ERROR: cannot write %s\n", jsonPath.string().c_str());
                return 1;
            }
            jf.write(json.data(), static_cast<std::streamsize>(json.size()));
        }

        ++written;
        if (written <= 5 || (written % 50) == 0) {
            std::printf("wrote %s  %dx%d  verts=%zu  part=%s\n", partId.c_str(), cw, ch,
                        localVerts.size() / 2, part.c_str());
        }
    }

    std::printf("=== export_dataset done ===\n");
    std::printf("written: %d  skipped: %d  artMeshes: %zu  out: %s\n", written, skipped,
                doc.artMeshes().size(), outDir.string().c_str());
    return written > 0 ? 0 : 1;
}
