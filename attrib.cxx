#include <iostream>
#include <type_traits>

using title [[attribute]] = const char*;
using url   [[attribute]] = const char*;
using magic [[attribute]] = int;

enum class edge {
  left, top, right, bottom,
};
enum class corner {
  nw, ne, se, sw,
};

template<typename T>
std::string to_string(T x) {
  if constexpr(T.is_enum) {
    return T.enum_values == x ...? T.enum_names : "unknown <" + T.string + ">";

  } else if constexpr(T == bool) {
    return x ? "true" : "false";

  } else if constexpr(T.is_arithmetic) {
    return std::to_string(x);

  } else {
    static_assert((const char*) == T);
    return x;
  }
}

struct foo_t {
  [[.edge=right, .url="https://www.fake.url/"]] int x;
  [[.title="Sometimes a vowel", .corner=ne]]    int y;
  [[.magic=10101, .title="The magic number"]]   int z;
};

template<typename type_t>
void print_member_attributes() {
  std::cout<< "Member attributes for " + type_t.string + "\n";

  @meta for(int i : type_t.member_count) {
    // Loop over each member.
    std::cout<< "  " + @member_name(type_t, i) + ":\n";

    // @member_attribute_list is a type parameter pack of all attribute names.
    // Loop over them and print them out.
    @meta for typename(t : { @member_attribute_list(type_t, i)... }) {
      std::cout<< "    "<< t.string << " = "<< 
        to_string(@member_attribute(type_t, i, t))<< "\n";
    }
  }
}

int main() {
  print_member_attributes<foo_t>();
}