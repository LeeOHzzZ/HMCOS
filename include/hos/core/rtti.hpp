#pragma once

#include <fmt/core.h>
#include <glog/logging.h>

namespace hos {

/// In this project we adopt LLVM-style RTTI.
/// The base class should define its own kind enum class and abstract
/// `GetKind()` method.
/// Each derived class should implement `GetKind()` method and define static
/// constexpr field `classKind`.

template <class Derived, class Base>
inline bool Is(const std::shared_ptr<Base> &ptr) {
    return ptr->GetKind() == Derived::classKind;
}

template <class Derived, class Base>
inline std::shared_ptr<Derived> As(const std::shared_ptr<Base> &ptr) {
    if (!Is<Derived>(ptr))
        LOG(FATAL) << fmt::format("Object is not of type `{}`.",
                                  typeid(Derived).name());
    else
        return std::shared_ptr<Derived>(ptr, static_cast<Derived *>(ptr.get()));
}

}  // namespace hos
