# Copyright (c) 2025, Oak Ridge National Laboratory.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tenseal as ts

# Create a TenSEAL context with CKKS scheme (for real numbers)
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)

# Set global scale and enable encryption parameters to be serialized
context.global_scale = 2 ** 40
context.generate_galois_keys()

# Encrypt two vectors
vec1 = [1.0, 2.0, 3.0]
vec2 = [4.0, 5.0, 6.0]

enc_vec1 = ts.ckks_vector(context, vec1)
enc_vec2 = ts.ckks_vector(context, vec2)

# Perform homomorphic addition and multiplication
enc_sum = enc_vec1 + enc_vec2
enc_product = enc_vec1 * enc_vec2

# Decrypt the results
print("Decrypted sum:     ", enc_sum.decrypt())
print("Decrypted product: ", enc_product.decrypt())
