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

#include "json_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_memcpy.cuh>
#include <cuda/functional>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tabulate.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

namespace spark_rapids_jni {

namespace detail {

namespace {

constexpr bool not_whitespace(cudf::char_utf8 ch)
{
  return ch != ' ' && ch != '\r' && ch != '\n' && ch != '\t';
}

constexpr bool can_be_delimiter(char c)
{
  // The character list below is from `json_reader_options.set_delimiter`.
  switch (c) {
    case '{':
    case '[':
    case '}':
    case ']':
    case ',':
    case ':':
    case '"':
    case '\'':
    case '\\':
    case ' ':
    case '\t':
    case '\r': return false;
    default: return true;
  }
}

}  // namespace

std::tuple<std::unique_ptr<cudf::column>, std::unique_ptr<rmm::device_buffer>, char> concat_json(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const d_input_ptr = cudf::column_device_view::create(input.parent(), stream);
  auto const default_mr  = rmm::mr::get_current_device_resource();

  // Check if the input rows are either null, equal to `null` string literal, or empty.
  // This will be used for masking out the input when doing string concatenation.
  rmm::device_uvector<bool> is_valid_input(input.size(), stream, default_mr);

  // Check if the input rows are either null or empty.
  // This will be returned to the caller.
  rmm::device_uvector<bool> is_null_or_empty(input.size(), stream, mr);

  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(0L),
    thrust::make_counting_iterator(input.size() * static_cast<int64_t>(cudf::detail::warp_size)),
    [input  = *d_input_ptr,
     output = thrust::make_zip_iterator(thrust::make_tuple(
       is_valid_input.begin(), is_null_or_empty.begin()))] __device__(int64_t tidx) {
      // Execute one warp per row to minimize thread divergence.
      if ((tidx % cudf::detail::warp_size) != 0) { return; }
      auto const idx = tidx / cudf::detail::warp_size;

      if (input.is_null(idx)) {
        output[idx] = thrust::make_tuple(false, true);
        return;
      }

      auto const d_str = input.element<cudf::string_view>(idx);
      auto const size  = d_str.size_bytes();
      int i            = 0;
      char ch;

      // Skip the very first whitespace characters.
      for (; i < size; ++i) {
        ch = d_str[i];
        if (not_whitespace(ch)) { break; }
      }

      if (i + 3 < size &&
          (d_str[i] == 'n' && d_str[i + 1] == 'u' && d_str[i + 2] == 'l' && d_str[i + 3] == 'l')) {
        i += 4;

        // Skip the very last whitespace characters.
        bool is_null_literal{true};
        for (; i < size; ++i) {
          ch = d_str[i];
          if (not_whitespace(ch)) {
            is_null_literal = false;
            break;
          }
        }

        // The current row contains only `null` string literal and not any other non-whitespace
        // characters. Such rows need to be masked out as null when doing concatenation.
        if (is_null_literal) {
          output[idx] = thrust::make_tuple(false, false);
          return;
        }
      }

      auto const not_eol = i < size;

      // If the current row is not null or empty, it should start with `{`. Otherwise, we need to
      // replace it by a null. This is necessary for libcudf's JSON reader to work.
      // Note that if we want to support ARRAY schema, we need to check for `[` instead.
      auto constexpr start_character = '{';
      if (not_eol && ch != start_character) {
        output[idx] = thrust::make_tuple(false, false);
        return;
      }

      output[idx] = thrust::make_tuple(not_eol, !not_eol);
    });

  auto constexpr num_levels  = 256;
  auto constexpr lower_level = std::numeric_limits<char>::min();
  auto constexpr upper_level = std::numeric_limits<char>::max();
  auto const num_chars       = input.chars_size(stream);

  rmm::device_uvector<uint32_t> histogram(num_levels, stream, default_mr);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), histogram.begin(), histogram.end(), 0);

  size_t temp_storage_bytes = 0;
  cub::DeviceHistogram::HistogramEven(nullptr,
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());
  rmm::device_buffer d_temp(temp_storage_bytes, stream);
  cub::DeviceHistogram::HistogramEven(d_temp.data(),
                                      temp_storage_bytes,
                                      input.chars_begin(stream),
                                      histogram.begin(),
                                      num_levels,
                                      lower_level,
                                      upper_level,
                                      num_chars,
                                      stream.value());

  auto const it             = thrust::make_counting_iterator(0);
  auto const zero_level_idx = -lower_level;  // the bin storing count for character `\0`
  auto const zero_level_it  = it + zero_level_idx;
  auto const end            = it + num_levels;

  auto const first_zero_count_pos =
    thrust::find_if(rmm::exec_policy_nosync(stream),
                    zero_level_it,  // ignore the negative characters
                    end,
                    [zero_level_idx, counts = histogram.begin()] __device__(auto idx) -> bool {
                      auto const count = counts[idx];
                      if (count > 0) { return false; }
                      auto const first_non_existing_char = static_cast<char>(idx - zero_level_idx);
                      return can_be_delimiter(first_non_existing_char);
                    });

  // This should never happen since the input should never cover the entire char range.
  if (first_zero_count_pos == end) {
    throw std::logic_error(
      "Cannot find any character suitable as delimiter during joining json strings.");
  }
  auto const delimiter = static_cast<char>(thrust::distance(zero_level_it, first_zero_count_pos));

  auto [null_mask, null_count] = cudf::detail::valid_if(
    is_valid_input.begin(), is_valid_input.end(), thrust::identity{}, stream, default_mr);
  // If the null count doesn't change, that mean we do not have any rows containing `null` string
  // literal or empty rows. In such cases, just use the input column for concatenation.
  auto const input_applied_null =
    null_count == input.null_count()
      ? cudf::column_view{}
      : cudf::column_view{cudf::data_type{cudf::type_id::STRING},
                          input.size(),
                          input.chars_begin(stream),
                          reinterpret_cast<cudf::bitmask_type const*>(null_mask.data()),
                          null_count,
                          0,
                          std::vector<cudf::column_view>{input.offsets()}};

  auto concat_strings = cudf::strings::detail::join_strings(
    null_count == input.null_count() ? input : cudf::strings_column_view{input_applied_null},
    cudf::string_scalar(std::string(1, delimiter), true, stream, default_mr),
    cudf::string_scalar("{}", true, stream, default_mr),
    stream,
    mr);

  return {std::make_unique<cudf::column>(std::move(is_null_or_empty), rmm::device_buffer{}, 0),
          std::move(concat_strings->release().data),
          delimiter};
}

std::unique_ptr<cudf::column> make_structs(std::vector<cudf::column_view> const& children,
                                           cudf::column_view const& is_null,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  if (children.size() == 0) { return nullptr; }

  auto const row_count = children.front().size();
  for (auto const& col : children) {
    CUDF_EXPECTS(col.size() == row_count, "All columns must have the same number of rows.");
  }

  auto const [null_mask, null_count] = cudf::detail::valid_if(
    is_null.begin<bool>(), is_null.end<bool>(), thrust::logical_not{}, stream, mr);

  auto const structs =
    cudf::column_view(cudf::data_type{cudf::type_id::STRUCT},
                      row_count,
                      nullptr,
                      reinterpret_cast<cudf::bitmask_type const*>(null_mask.data()),
                      null_count,
                      0,
                      children);
  return std::make_unique<cudf::column>(structs, stream, mr);
}

namespace {

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> cast_strings_to_booleans(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto output = cudf::make_fixed_width_column(
    cudf::data_type{cudf::type_id::BOOL8}, input.size(), cudf::mask_state::UNALLOCATED, stream, mr);
  auto validity = rmm::device_uvector<bool>(input.size(), stream);  // intentionally not use `mr`

  auto const d_input_ptr = cudf::column_device_view::create(input, stream);
  auto const output_it   = thrust::make_zip_iterator(
    thrust::make_tuple(output->mutable_view().begin<bool>(), validity.begin()));
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   output_it,
                   output_it + input.size(),
                   [input = *d_input_ptr] __device__(auto idx) -> thrust::tuple<bool, bool> {
                     if (input.is_valid(idx)) {
                       auto const d_str = input.element<cudf::string_view>(idx);
                       if (d_str.size_bytes() == 4 && d_str[0] == 't' && d_str[1] == 'r' &&
                           d_str[2] == 'u' && d_str[3] == 'e') {
                         return {true, true};
                       }
                       if (d_str.size_bytes() == 5 && d_str[0] == 'f' && d_str[1] == 'a' &&
                           d_str[2] == 'l' && d_str[3] == 's' && d_str[4] == 'e') {
                         return {false, true};
                       }
                     }

                     // Either null input, or the input string is neither `true` nor `false`.
                     return {false, false};
                   });

  return {std::move(output), std::move(validity)};
}

// TODO: remove this.
template <typename IndexPairIterator>
rmm::device_uvector<char> make_chars_buffer(cudf::column_view const& offsets,
                                            int64_t chars_size,
                                            IndexPairIterator begin,
                                            cudf::size_type string_count,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto chars_data      = rmm::device_uvector<char>(chars_size, stream, mr);
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets);

  auto const src_ptrs = cudf::detail::make_counting_transform_iterator(
    0u, cuda::proclaim_return_type<void*>([begin] __device__(uint32_t idx) {
      // Due to a bug in cub (https://github.com/NVIDIA/cccl/issues/586),
      // we have to use `const_cast` to remove `const` qualifier from the source pointer.
      // This should be fine as long as we only read but not write anything to the source.
      return reinterpret_cast<void*>(const_cast<char*>(begin[idx].first));
    }));
  auto const src_sizes = cudf::detail::make_counting_transform_iterator(
    0u, cuda::proclaim_return_type<cudf::size_type>([begin] __device__(uint32_t idx) {
      return begin[idx].second;
    }));
  auto const dst_ptrs = cudf::detail::make_counting_transform_iterator(
    0u,
    cuda::proclaim_return_type<char*>([offsets = d_offsets, output = chars_data.data()] __device__(
                                        uint32_t idx) { return output + offsets[idx]; }));

  size_t temp_storage_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(
    nullptr, temp_storage_bytes, src_ptrs, dst_ptrs, src_sizes, string_count, stream.value()));
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);
  CUDF_CUDA_TRY(cub::DeviceMemcpy::Batched(d_temp_storage.data(),
                                           temp_storage_bytes,
                                           src_ptrs,
                                           dst_ptrs,
                                           src_sizes,
                                           string_count,
                                           stream.value()));

  return chars_data;
}

std::pair<std::unique_ptr<cudf::column>, rmm::device_uvector<bool>> remove_quotes(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const d_input_ptr  = cudf::column_device_view::create(input, stream);
  auto const string_count = input.size();

  // Materialize the output string sizes to avoid repeated computation when being used multiple
  // times later on.
  auto output_sizes = rmm::device_uvector<cudf::size_type>(string_count, stream);
  thrust::tabulate(rmm::exec_policy_nosync(stream),
                   output_sizes.begin(),
                   output_sizes.end(),
                   [input = *d_input_ptr] __device__(cudf::size_type idx) -> cudf::size_type {
                     if (input.is_null(idx)) { return 0; }

                     auto const d_str = input.element<cudf::string_view>(idx);
                     auto const size  = d_str.size_bytes();

                     // Need to check for size, since the input string may contain just a single
                     // character `"`. Such input should not be considered as quoted.
                     auto const is_quoted = size > 1 && d_str[0] == '"' && d_str[size - 1] == '"';
                     return is_quoted ? size - 2 : size;
                   });

  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  auto const input_sv = cudf::strings_column_view{input};
  auto const d_input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input_sv.offsets());
  auto const index_pair_fn = cuda::proclaim_return_type<thrust::pair<const char*, cudf::size_type>>(
    [chars         = input_sv.chars_begin(stream),
     input_offsets = d_input_offsets,
     output_sizes  = output_sizes.begin()] __device__(cudf::size_type idx) {
      auto const start_offset = input_offsets[idx];
      auto const end_offset   = input_offsets[idx + 1];
      auto const input_size   = end_offset - start_offset;
      auto const output_size  = output_sizes[idx];

      return thrust::pair{chars + start_offset + (input_size == output_size ? 0 : 1), output_size};
    });
  auto const index_pair_it = cudf::detail::make_counting_transform_iterator(0, index_pair_fn);
  auto chars_data          = /*cudf::strings::detail::*/ make_chars_buffer(
    offsets_column->view(), bytes, index_pair_it, string_count, stream, mr);

  auto output = cudf::make_strings_column(string_count,
                                          std::move(offsets_column),
                                          chars_data.release(),
                                          input.null_count(),
                                          cudf::detail::copy_bitmask(input, stream, mr));
  return {std::move(output), rmm::device_uvector<bool>{0, stream}};
}

std::unique_ptr<cudf::column> convert_column_type(cudf::column_view const& input,
                                                  json_schema_element const& schema,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  return nullptr;
}

}  // namespace

std::unique_ptr<cudf::column> convert_types(
  cudf::table_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_columns = input.num_columns();
  CUDF_EXPECTS(static_cast<std::size_t>(num_columns) == schema.size(),
               "Numbers of columns in the input table is different from schema size.");

  std::vector<std::unique_ptr<cudf::column>> converted_cols(num_columns);
  for (int i = 0; i < num_columns; ++i) {
    converted_cols[i] = convert_column_type(input.column(i), schema[i].second, stream, mr);
  }

  return nullptr;
}

}  // namespace detail

std::tuple<std::unique_ptr<cudf::column>, std::unique_ptr<rmm::device_buffer>, char> concat_json(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concat_json(input, stream, mr);
}

std::unique_ptr<cudf::column> make_structs(std::vector<cudf::column_view> const& children,
                                           cudf::column_view const& is_null,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::make_structs(children, is_null, stream, mr);
}

std::unique_ptr<cudf::column> convert_types(
  cudf::table_view const& input,
  std::vector<std::pair<std::string, json_schema_element>> const& schema,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::convert_types(input, schema, stream, mr);
}

std::unique_ptr<cudf::column> cast_strings_to_booleans(cudf::column_view const& input,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] = detail::cast_strings_to_booleans(input, stream, mr);
  auto [null_mask, null_count] =
    cudf::detail::valid_if(validity.begin(), validity.end(), thrust::identity{}, stream, mr);
  if (null_count > 0) {
    output->set_null_mask(std::move(null_mask), null_count);
  } else {
    output->set_null_mask(rmm::device_buffer{}, 0);
  }
  return std::move(output);
}

std::unique_ptr<cudf::column> remove_quotes(cudf::column_view const& input,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output, validity] = detail::remove_quotes(input, stream, mr);
  return std::move(output);
}

}  // namespace spark_rapids_jni
