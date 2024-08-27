/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cudf/strings/strings_column_view.hpp>

#include <rmm/resource_ref.hpp>

#include <memory>
#include <vector>

namespace spark_rapids_jni {

/**
 * @brief The maximum supported depth that a JSON path can reach.
 */
constexpr int MAX_JSON_PATH_DEPTH = 16;

/**
 * @brief Type of instruction in a JSON path.
 */
enum class path_instruction_type : int8_t { WILDCARD, INDEX, NAMED };

/**
 * @brief Extract JSON object from a JSON string based on the specified JSON path.
 *
 * If the input JSON string is invalid, or it does not contain the object at the given path, a null
 * will be returned.
 */
std::unique_ptr<cudf::column> get_json_object(
  cudf::strings_column_view const& input,
  std::vector<std::tuple<path_instruction_type, std::string, int32_t>> const& instructions,
  bool output_as_string_verbatim    = false,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

/**
 * @brief Extract multiple JSON objects from a JSON string based on the specified JSON paths.
 *
 * This function processes all the JSON paths in parallel, which may be faster than calling
 * to `get_json_object` on the individual JSON paths. However, it may consume much more GPU
 * memory, proportional to the number of JSON paths.
 * @param input the input string column to parse JSON from
 * @param json_paths the path operations to read extract
 * @param memory_budget_bytes a memory budget for temporary memory usage if > 0
 * @param parallel_override if this value is greater than 0 then it specifies the
 *        number of paths to process in parallel (this will cause the
 *        `memory_budget_bytes` paramemter to be ignored)
 * @param output_as_string_verbatim Output strings or numbers exactly as given in the input
 *        (if the input values are given as quoted strings then the output will also be quoted)
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
std::vector<std::unique_ptr<cudf::column>> get_json_object_multiple_paths(
  cudf::strings_column_view const& input,
  std::vector<std::vector<std::tuple<path_instruction_type, std::string, int32_t>>> const&
    json_paths,
  int64_t memory_budget_bytes,
  int32_t parallel_override,
  bool output_as_string_verbatim    = false,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = rmm::mr::get_current_device_resource());

}  // namespace spark_rapids_jni
