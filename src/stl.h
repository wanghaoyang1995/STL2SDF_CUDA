//
// Created by wangh on 2025/4/8.
//

#ifndef STLREADER_H
#define STLREADER_H

#include <cstdint>
#include <vector>
#include "define.h"

namespace STL {
    struct Vec3 {
        float x, y, z;

        Vec3(float x = 0.0f, float y = 0.0f, float z = 0.0f) : x(x), y(y), z(z) {}

        Vec3 operator-(const Vec3& other) const {
            return {x - other.x, y - other.y, z - other.z};
        }
    };

    struct Triangle {
        Vec3 normal;
        Vec3 vertices[3];
        uint16_t attribute_byte_count;
    };

    struct STLHeader {
        char header[80];
        uint32_t triangle_count;
    };

    struct STLObject {
        STLHeader header;
        std::vector<Triangle> triangles;
    };

    bool read_binary_stl(const char* filename, STLObject& stl_obj);
    bool read_ascii_stl(const char* filename, STLObject& stl_obj);

    bool read_binary_stl(const char* filename, std::vector<::Triangle>& triangles, Vertice& min_bound, Vertice& max_bound);
    bool read_ascii_stl(const char* filename, std::vector<::Triangle>& triangles, Vertice& min_bound, Vertice& max_bound);

    bool read_binary_stl(const char* filename, std::vector<::Triangle>& triangles);
    bool read_ascii_stl(const char* filename, std::vector<::Triangle>& triangles);

    void write_ascii_stl(const char* filename, const std::vector<::Triangle>& triangles);
}

#endif //STLREADER_H
