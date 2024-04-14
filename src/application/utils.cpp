#include "utils.hpp"
#include <fstream>
#include <iomanip>

__attribute__((visibility("default")))
void save_mem(const void* ptr, size_t size, const std::string& fname) {
    size_t offset   = 0;
    const size_t step     = 32;

    std::ofstream fout(fname, std::ios::out);
    fout << std::hex;

    while (offset < size) {
        if (offset % step == 0) {
            fout << "0x" << std::setfill('0') << std::setw(8) << offset
                 << ": ";
        }

        fout << std::setfill('0') << std::setw(2) << static_cast<int>(reinterpret_cast<const uint8_t*>(ptr)[offset]);

        offset ++;

        if (offset % step == 0) {
            fout << '\n';
        } else if (offset % 8 == 0) {
            fout << "  ";
        } else {
            fout << " ";
        }
    }

    fout.close();
}
