#pragma once

#include <hos/core/hier.hpp>

namespace hos {

/// Passes that perform transformations on hierarchical graphs to enable
/// memory-aware scheduling

/// Join continunous sequences to form a larger sequence
class JoinSequencePass : public HierGraphPass {
public:
    void Run(HierGraph &graph) override;
};

}  // namespace hos