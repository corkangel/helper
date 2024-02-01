
#pragma once

#include <vector>

using column = std::vector<float>;
using matrix = std::vector<column>;

int argmax(const column& values);
