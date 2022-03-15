#include <iostream>

template<typename T> requires (T.is_enum)
const char* to_string(T e) {
  return T.enum_values == e ...? T.enum_names : "unknown <" + T.string + ">";
}

enum shape_t {
  circle, 
  triangle,
  square,
  pentagon=5,
  hexagon=6,
  octagon=8,
};

int main() {
  shape_t shapes[] {
    triangle, square, hexagon, (shape_t)9
  };

  for(shape_t shape : shapes)
    std::cout<< (int)shape<< " : "<< to_string(shape)<< "\n";
}