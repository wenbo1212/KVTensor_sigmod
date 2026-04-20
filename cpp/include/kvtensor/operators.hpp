#pragma once

#include "kvtensor/context.hpp"
#include "kvtensor/types.hpp"
#include <memory>
#include <string>

namespace kvtensor {

// Forward declarations
class BlockMatrix;
class InMemoryMatrix;

// Base operator class
class Operator {
public:
    virtual ~Operator() = default;
    
    // Execute operator
    virtual std::shared_ptr<InMemoryMatrix> execute(OperatorContext& ctx) = 0;
    
    const std::string& name() const { return name_; }

protected:
    std::string name_;
    std::string result_id_;
};

} // namespace kvtensor
