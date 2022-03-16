#include <iostream>

@meta std::cout<< "meta printf is a compile-time printf\n";

template<typename... Ts>
struct tuple {
  @meta std::cout<< "meta for is a compile-time unrolled loop\n";
  @meta for(int i : sizeof...(Ts)) {
    @meta std::cout<< "  declaring member of type " + Ts...[i].string + "\n";
    Ts...[i] @(i);
  }
};

using my_tuple = tuple<int, long*, std::pair<char, short>>;

// It has those data members!
my_tuple tup { 1, nullptr, { 1, 2 } };

// Use reflection to print the member declarations of my_tuple.
@meta std::cout<< my_tuple.string + " members:\n";
@meta std::cout<< "  " + my_tuple.member_decl_strings + "\n" ...;

template<typename T> requires (T.is_enum)
const char* to_string(T e) {
  // switch over e, which is not known at copmplie time.
  switch(e) {
    @meta std::cout<< "Generating enum switch for type " + T.string + "\n";

    // Execute a compile-time loop over all enumerators of T.
    // e2 is a constant expression.
    @meta for enum(T e2 : T) {
      @meta std::cout<< "  Generating enum case for '" + e2.string + "'\n";
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