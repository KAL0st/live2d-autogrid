// probe_mesh.cpp v3 — dump ArtMesh uvs/positions，读法完全照抄 export_dataset.cpp
#include "core/Cmo3Document.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

bool fieldIs(const l2d::XsNode& n, const std::string& f) {
    if (n.fieldName == f) return true;
    for (const auto& a : n.attrs) if (a.first == "xs.n" && a.second == f) return true;
    return false;
}

const l2d::XsNode* directField(const l2d::XsNode& n, const std::string& f) {
    for (const auto& c : n.children) if (fieldIs(c, f)) return &c;
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
    return !out.empty();
}

struct BB {
    double x0 = 1e18, y0 = 1e18, x1 = -1e18, y1 = -1e18; bool any = false;
    void add(double x, double y) {
        x0 = std::min(x0, x); y0 = std::min(y0, y);
        x1 = std::max(x1, x); y1 = std::max(y1, y); any = true;
    }
    double w() const { return x1 - x0; }
    double h() const { return y1 - y0; }
};

std::FILE* g_log = nullptr;
void note(const char* fmt, ...) {
    if (g_log == nullptr) return;
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_log, fmt, ap);
    va_end(ap);
    std::fflush(g_log);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: probe_mesh <in.cmo3> <out.csv> [--limit N]\n"); return 2; }
    const std::string inPath = argv[1];
    const std::string outPath = argv[2];
    int limit = 0;
    for (int i = 3; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--limit") == 0) limit = std::atoi(argv[i + 1]);
    }

    l2d::Cmo3Document doc;
    std::string err;
    if (!doc.load(inPath, err)) { std::printf("ERROR: load: %s\n", err.c_str()); return 1; }
    const l2d::XsDocument& xml = doc.xml();
    const auto& texs = doc.textures();

    g_log = std::fopen("_tmp/probe_log.txt", "wb");
    note("LOADED artMeshes=%zu texCount=%zu\n", doc.artMeshes().size(), texs.size());

    std::FILE* f = std::fopen(outPath.c_str(), "wb");
    if (f == nullptr) { std::printf("ERROR: cannot open %s\n", outPath.c_str()); return 1; }
    std::fprintf(f, "xsId,name,texIdx,nUv,nPos,idxCount,hasPos,expandPos,"
                    "u0,u1,v0,v1,p0x,p1x,p0y,p1y,aspectUV,aspectPos,uvOnGrid64,texCount,texName\n");
    std::fflush(f);

    int n = 0, skipped = 0, noPos = 0;
    size_t idx = 0;
    for (const l2d::ArtMesh& am : doc.artMeshes()) {
        if (limit > 0 && n >= limit) break;
        note("TRY[%zu] xsId=%d texIdx=%d source=%p\n", idx, am.sourceXsId, am.textureIndex,
             static_cast<const void*>(am.source));
        ++idx;
        if (am.sourceXsId < 0) { ++skipped; continue; }
        const l2d::XsNode* mesh = am.source;
        if (mesh == nullptr) mesh = xml.resolve(am.sourceXsId);
        if (mesh == nullptr) { note("  -> null mesh\n"); ++skipped; continue; }
        note("  mesh=%p text=%zu\n", static_cast<const void*>(mesh), mesh->text.size());

        std::vector<float> uvs, pos;
        const bool hasUv = readFloatField(xml, *mesh, "uvs", uvs);
        note("  uvs=%d n=%zu\n", hasUv ? 1 : 0, uvs.size());
        if (!hasUv || uvs.size() < 6) { ++skipped; continue; }
        const bool hasPos = readFloatField(xml, *mesh, "positions", pos);
        note("  pos=%d n=%zu\n", hasPos ? 1 : 0, pos.size());
        if (!hasPos) ++noPos;

        BB bu, bp;
        for (size_t i = 0; i + 1 < uvs.size(); i += 2) bu.add(uvs[i], uvs[i + 1]);
        for (size_t i = 0; i + 1 < pos.size(); i += 2) bp.add(pos[i], pos[i + 1]);

        int onGrid = 0, total = 0;
        for (size_t i = 0; i < uvs.size(); ++i) {
            const double v = static_cast<double>(uvs[i]) * 64.0;
            if (std::fabs(v - std::floor(v + 0.5)) < 0.02) ++onGrid;
            ++total;
        }

        note("  name...\n");
        std::string name = am.displayName();
        note("  name=%s texname...\n", name.c_str());
        std::string esc;
        for (char c : name) { if (c == '"' || c == ',' || c == '\n' || c == '\r') esc.push_back('\''); else esc.push_back(c); }
        std::string tesc;
        if (am.textureIndex >= 0 && am.textureIndex < static_cast<int>(texs.size())) {
            const std::string& tn = texs[static_cast<size_t>(am.textureIndex)].atlasName;
            for (char c : tn) { if (c == '"' || c == ',' || c == '\n' || c == '\r') tesc.push_back('\''); else tesc.push_back(c); }
        }
        note("  writing...\n");

        std::fprintf(f, "%d,\"%s\",%d,%zu,%zu,%d,%d,%d,"
                        "%.6f,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%d,%zu,\"%s\"\n",
                     am.sourceXsId, esc.c_str(), am.textureIndex,
                     uvs.size() / 2, pos.size() / 2, am.indexCount,
                     hasPos ? 1 : 0,
                     (hasPos && pos.size() == uvs.size()) ? 1 : 0,
                     bu.any ? bu.x0 : 0.0, bu.any ? bu.x1 : 0.0,
                     bu.any ? bu.y0 : 0.0, bu.any ? bu.y1 : 0.0,
                     bp.any ? bp.x0 : 0.0, bp.any ? bp.x1 : 0.0,
                     bp.any ? bp.y0 : 0.0, bp.any ? bp.y1 : 0.0,
                     bu.any && bu.h() > 1e-9 ? bu.w() / bu.h() : 0.0,
                     bp.any && bp.h() > 1e-9 ? bp.w() / bp.h() : 0.0,
                     total > 0 ? (onGrid * 100 / total) : -1,
                     texs.size(), tesc.c_str());
        std::fflush(f);
        note("  OK[%d]\n", n);
        ++n;
    }

    std::fclose(f);
    note("=== done artMeshes=%zu written=%d skipped=%d noPos=%d texCount=%zu\n",
         doc.artMeshes().size(), n, skipped, noPos, texs.size());
    std::printf("done written=%d skipped=%d noPos=%d\n", n, skipped, noPos);
    return n > 0 ? 0 : 1;
}
