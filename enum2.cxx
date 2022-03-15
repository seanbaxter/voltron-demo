#include <iostream>

template<typename T> requires (T.is_enum)
const char* to_string(T e) {
  switch(e) {
    // switch over e, which is not known at copmplie time.
    
    @meta for enum(T e2 : T) {
      // Execute a compile-time loop over all enumerators of T.
      // e2 is a constant expression.
      case e2:              // case-statement works when e2 is constant.
        return e2.string;   // static reflection works when e2 is constant.
    }

    default:
      return "unknown <" + T.string + ">";
  }
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