//
// Created by wangh on 2025/4/8.
//

#include <cstdint>
#include <cstdio>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>

#include "stl.h"

namespace {
    bool parse_vertex(const std::string& line, STL::Vec3& vertex) {
        std::istringstream iss(line);
        std::string token;
        float x, y, z;

        iss >> token;

        if (!(iss >> x >> y >> z)) {
            return false;
        }

        vertex = STL::Vec3(x, y, z);
        return true;
    }

    bool parse_vertex(const std::string& line, ::Vertice& vertex) {
        std::istringstream iss(line);
        std::string token;
        float x, y, z;

        iss >> token;

        if (!(iss >> x >> y >> z)) {
            return false;
        }

        vertex = Vertice(x, y, z);
        return true;
    }

    bool parse_facet_normal(const std::string& line, STL::Vec3& vertex) {
        std::istringstream iss(line);
        std::string token;
        float x, y, z;
        iss >> token;
        iss >> token;

        if (!(iss >> x >> y >> z)) {
            return false;
        }

        vertex = STL::Vec3(x, y, z);
        return true;
    }

    std::string trim(const std::string& s) {
        size_t start = s.find_first_not_of(" \t\n\r");
        size_t end = s.find_last_not_of(" \t\n\r");
        return (start == std::string::npos) ? "" : s.substr(start, end - start + 1);
    }
}

bool STL::read_binary_stl(const char* filename, STLObject& stl_obj) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    file.read(reinterpret_cast<char*>(&stl_obj.header), sizeof(STLHeader));
    if (!file) {
        printf("Cannot read header of file %s!\n", filename);
        return false;
    }

    stl_obj.triangles.resize(stl_obj.header.triangle_count);

    for (int i = 0; i < stl_obj.header.triangle_count; i++) {
        STL::Triangle& tri = stl_obj.triangles[i];
        file.read(reinterpret_cast<char*>(&tri.normal), sizeof(Vec3));
        file.read(reinterpret_cast<char*>(&tri.vertices), sizeof(Vec3) * 3);
        file.read(reinterpret_cast<char*>(&tri.attribute_byte_count), sizeof(uint16_t));
        if (!file) {
            printf("Failed to read triangle %d\n", i + 1);
            stl_obj.triangles.resize(i);
            return false;
        }
    }

    return true;
}

/*
ASCII格式如下:

solid
 facet normal -0.000000e+00  0.000000e+00 -1.000000e+00
   outer loop
     vertex  4.000000e+03  1.000000e+03  0.000000e+00
     vertex  8.326673e-14  3.000000e+02  0.000000e+00
     vertex  8.326673e-14  1.000000e+03  0.000000e+00
   endloop
 endfacet
 ...
endsolid

*/

bool STL::read_ascii_stl(const char* filename, STL::STLObject& stl_obj) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    std::string line;
    bool inside_solid = false;
    int triangle_count = 0;

    while (std::getline(file, line)) {
        line = trim(line);

        // TODO: string::empty is an MSVC-specific extension; to ensure cross-platform compatibility, use the standard library function instead
        if (line.empty()) {
            continue;
        }

        if (line.compare(0, 5, "solid") == 0) {
            inside_solid = true;
            continue;
        }

        if (inside_solid) {
            if (line.compare(0, 6, "endsolid") == 0) {
                break;
            }

            if (line.compare(0, 5, "facet") == 0) {
                STL::Triangle tri;
                if (!parse_facet_normal(line, tri.normal)) {
                    printf("Parse facet section error!\n");
                    return false;
                }
                std::getline(file, line);
                for (int v = 0; v < 3; ++v) {
                    std::getline(file, line);
                    if (!parse_vertex(line, tri.vertices[v])) {
                        printf("Parse vertex section error!\n");
                        return false;
                    }
                }
                std::getline(file, line);
                std::getline(file, line);

                stl_obj.triangles.push_back(tri);
                triangle_count++;
            }
        }
    }

    file.close();

    stl_obj.header.header[0] = '\0';
    stl_obj.header.triangle_count = triangle_count;

    printf("Read finished. Triangles count: %d\n", triangle_count);
    return true;
}


bool STL::read_binary_stl(const char* filename, std::vector<::Triangle>& triangles, Vertice& min_bound, Vertice& max_bound) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    min_bound = Vertice(FLT_MAX, FLT_MAX, FLT_MAX);
    max_bound = Vertice(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    STL::STLHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(STLHeader));
    if (!file) {
        printf("Cannot read header of file %s!\n", filename);
        return false;
    }

    triangles.resize(header.triangle_count);

    for (int i = 0; i < header.triangle_count; i++) {
        ::Triangle& tri = triangles[i];
        file.seekg(12, std::ios::cur);
        file.read(reinterpret_cast<char*>(&tri.vertices), 36);
        file.seekg(2, std::ios::cur);
        if (!file) {
            printf("Failed to read triangle %d\n", i + 1);
            triangles.resize(i);
            return false;
        }


        for (const Vertice& vertex : tri.vertices) {
            if (vertex.x < min_bound.x) min_bound.x = vertex.x;
            if (vertex.y < min_bound.y) min_bound.y = vertex.y;
            if (vertex.z < min_bound.z) min_bound.z = vertex.z;

            if (vertex.x > max_bound.x) max_bound.x = vertex.x;
            if (vertex.y > max_bound.y) max_bound.y = vertex.y;
            if (vertex.z > max_bound.z) max_bound.z = vertex.z;
        }
    }

    printf("STL triangles count: %zu\n", triangles.size());
    return true;
}

bool STL::read_ascii_stl(const char* filename, std::vector<::Triangle>& triangles, Vertice& min_bound, Vertice& max_bound) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    min_bound = Vertice(FLT_MAX, FLT_MAX, FLT_MAX);
    max_bound = Vertice(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    std::string line;
    bool inside_solid = false;
    int triangle_count = 0;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        if (line.compare(0, 5, "solid") == 0) {
            inside_solid = true;
            continue;
        }

        if (inside_solid) {
            if (line.compare(0, 6, "endsolid") == 0) {
                break;
            }

            if (line.compare(0, 5, "facet") == 0) {
                ::Triangle tri;
                std::getline(file, line);
                for (int v = 0; v < 3; ++v) {
                    std::getline(file, line);
                    if (!parse_vertex(line, tri.vertices[v])) {
                        printf("Parse vertex section error!\n");
                        return false;
                    }

                    for (const Vertice& vertex : tri.vertices) {
                        if (vertex.x < min_bound.x) min_bound.x = vertex.x;
                        if (vertex.y < min_bound.y) min_bound.y = vertex.y;
                        if (vertex.z < min_bound.z) min_bound.z = vertex.z;

                        if (vertex.x > max_bound.x) max_bound.x = vertex.x;
                        if (vertex.y > max_bound.y) max_bound.y = vertex.y;
                        if (vertex.z > max_bound.z) max_bound.z = vertex.z;
                    }
                }
                std::getline(file, line);
                std::getline(file, line);

                triangles.push_back(tri);
                triangle_count++;
            }
        }
    }

    file.close();

    printf("STL triangles count: %d\n", triangle_count);
    return true;
}


bool STL::read_binary_stl(const char* filename, std::vector<::Triangle>& triangles) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    STL::STLHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(STLHeader));
    if (!file) {
        printf("Cannot read header of file %s!\n", filename);
        return false;
    }

    triangles.resize(header.triangle_count);

    for (int i = 0; i < header.triangle_count; i++) {
        ::Triangle& tri = triangles[i];
        file.seekg(12, std::ios::cur);
        file.read(reinterpret_cast<char*>(&tri.vertices), 36);
        file.seekg(2, std::ios::cur);
        if (!file) {
            printf("Failed to read triangle %d\n", i + 1);
            triangles.resize(i);
            return false;
        }
    }

    printf("STL triangles count: %zu\n", triangles.size());
    return true;
}

bool STL::read_ascii_stl(const char* filename, std::vector<::Triangle>& triangles) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        printf("Cannot open file %s!\n", filename);
        return false;
    }

    std::string line;
    bool inside_solid = false;
    int triangle_count = 0;

    while (std::getline(file, line)) {
        line = trim(line);
        if (line.empty()) {
            continue;
        }

        if (line.compare(0, 5, "solid") == 0) {
            inside_solid = true;
            continue;
        }

        if (inside_solid) {
            if (line.compare(0, 6, "endsolid") == 0) {
                break;
            }

            if (line.compare(0, 5, "facet") == 0) {
                ::Triangle tri;
                std::getline(file, line);
                for (int v = 0; v < 3; ++v) {
                    std::getline(file, line);
                    if (!parse_vertex(line, tri.vertices[v])) {
                        printf("Parse vertex section error!\n");
                        return false;
                    }
                }
                std::getline(file, line);
                std::getline(file, line);

                triangles.push_back(tri);
                triangle_count++;
            }
        }
    }

    file.close();

    printf("STL triangles count: %d\n", triangle_count);
    return true;
}

void STL::write_ascii_stl(const char* filename, const std::vector<::Triangle>& triangles) {
    std::ofstream outFile(filename);
    if (!outFile) {
        printf("Could not open file for writing.\n");
    }

    outFile << "solid STL_model\n";
    for (const ::Triangle& triangle : triangles) {
        outFile << "  facet normal -1 -1 -1\n";
        outFile << "    outer loop\n";

        outFile << "      vertex " << triangle.vertices[0].x << " " << triangle.vertices[0].y << " " << triangle.vertices[0].z << "\n";
        outFile << "      vertex " << triangle.vertices[1].x << " " << triangle.vertices[1].y << " " << triangle.vertices[1].z << "\n";
        outFile << "      vertex " << triangle.vertices[2].x << " " << triangle.vertices[2].y << " " << triangle.vertices[2].z << "\n";

        outFile << "    endloop\n";
        outFile << "  endfacet\n";
    }

    outFile << "endsolid STL_model\n";

    outFile.close();
}
