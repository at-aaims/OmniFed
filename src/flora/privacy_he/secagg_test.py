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

import random
import hashlib
import hmac
from typing import Dict, List, Tuple, Optional
import numpy as np


class SecureAggregation:
    """
    Secure aggregation protocol
    Based on the original Google paper: "Practical Secure Aggregation for Privacy-Preserving Machine Learning"
    """

    def __init__(self, num_clients: int, threshold_t: int, gradient_size: int):
        self.num_clients = num_clients
        self.threshold_t = threshold_t  # Minimum clients needed to reconstruct
        self.gradient_size = gradient_size
        # Use a large prime for finite field arithmetic (2^31 - 1 for simplicity)
        self.modulus_q = 2147483647  # Mersenne prime 2^31 - 1
        self.scale_factor = 1000000  # Scale factor for float to int conversion

        # Storage for protocol state
        self.pairwise_keys = {}  # [client_i][client_j] = shared_key
        self.ss_keys = {}  # Secret sharing keys for each client
        self.commitments = {}  # Public commitments for verification

    def setup_phase(self) -> Dict[int, Dict]:
        """
        Phase 1: Setup - Generate keys and commitments
        Returns: Dictionary with setup information for each client
        """
        client_setup_info = {}

        for client_i in range(self.num_clients):
            # Generate pairwise keys with all other clients
            pairwise_keys_i = {}
            for client_j in range(self.num_clients):
                if client_i != client_j:
                    # Generate shared key using deterministic method
                    # In practice, this would be done via Diffie-Hellman key exchange
                    shared_key = self._generate_shared_key(client_i, client_j)
                    pairwise_keys_i[client_j] = shared_key

            # Generate secret sharing key (random value)
            ss_key = random.randint(1, self.modulus_q - 1)

            # Store keys (in practice, each client would only know their own keys)
            if client_i not in self.pairwise_keys:
                self.pairwise_keys[client_i] = {}
            self.pairwise_keys[client_i] = pairwise_keys_i
            self.ss_keys[client_i] = ss_key

            # Create setup info for this client
            client_setup_info[client_i] = {
                "client_id": client_i,
                "pairwise_keys": pairwise_keys_i,
                "ss_key": ss_key,
                "threshold": self.threshold_t,
                "modulus": self.modulus_q,
            }

        return client_setup_info

    def _generate_shared_key(self, client_i: int, client_j: int) -> int:
        """
        Generate a shared key between two clients
        In practice, this would be done via Diffie-Hellman key exchange
        Here we use a deterministic method for simulation
        """
        # Ensure same key regardless of order (i,j) or (j,i)
        min_id, max_id = min(client_i, client_j), max(client_i, client_j)

        # Use HMAC for deterministic key generation
        key_material = f"client_{min_id}_client_{max_id}".encode()
        seed = b"secure_aggregation_seed_2024"  # In practice, use secure random seed

        hmac_obj = hmac.new(seed, key_material, hashlib.sha256)
        key_bytes = hmac_obj.digest()

        # Convert to integer in finite field
        key_int = int.from_bytes(key_bytes[:4], byteorder="big") % self.modulus_q
        return key_int

    # This would run on each client individually
    # def generate_pairwise_keys(self, my_client_id: int, other_clients_public_keys: Dict[int, int]) -> Dict[int, int]:
    #     """
    #     Each client runs this locally to compute shared keys with all other clients
    #     using their own private key and others' public keys
    #     """
    #     my_private_key = self.my_private_key  # Only this client knows this
    #     pairwise_keys = {}
    #
    #     for other_id, other_public_key in other_clients_public_keys.items():
    #         if other_id != my_client_id:
    #             # Compute shared secret using Diffie-Hellman
    #             shared_secret = pow(other_public_key, my_private_key, self.dh_prime)
    #             pairwise_keys[other_id] = shared_secret
    #
    #     return pairwise_keys

    def generate_mask(self, client_id: int, active_clients: List[int]) -> List[int]:
        """
        Generate random mask for a client using pairwise keys
        Only consider active clients for mask generation
        """
        mask = [0] * self.gradient_size

        # Generate mask components using pairwise keys with active clients only
        for other_client in active_clients:
            if (
                other_client != client_id
                and other_client in self.pairwise_keys[client_id]
            ):
                shared_key = self.pairwise_keys[client_id][other_client]
                # Generate pseudorandom values using the shared key
                prg_output = self._prg(shared_key, self.gradient_size)

                # Add or subtract based on client ID ordering (ensures cancellation)
                if client_id < other_client:
                    # Add contribution
                    for i in range(self.gradient_size):
                        mask[i] = (mask[i] + prg_output[i]) % self.modulus_q
                else:
                    # Subtract contribution
                    for i in range(self.gradient_size):
                        mask[i] = (mask[i] - prg_output[i]) % self.modulus_q

        return mask

    def _prg(self, seed: int, length: int) -> List[int]:
        """
        Pseudorandom generator - generates deterministic random sequence from seed
        """
        # Convert seed to bytes for hashing
        seed_bytes = seed.to_bytes(8, byteorder="big")

        output = []
        for i in range(length):
            # Generate hash for each position
            h = hashlib.sha256(seed_bytes + i.to_bytes(4, byteorder="big")).digest()
            # Convert first 4 bytes to integer in finite field
            value = int.from_bytes(h[:4], byteorder="big") % self.modulus_q
            output.append(value)

        return output

    def create_secret_shares(
        self, client_id: int, secret_values: List[int]
    ) -> Dict[int, List[int]]:
        """
        Create secret shares of the given values using Shamir's Secret Sharing
        """
        shares = {i: [] for i in range(self.num_clients)}

        for secret_value in secret_values:
            # Generate random polynomial coefficients
            # f(x) = secret_value + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
            coefficients = [secret_value]
            for _ in range(self.threshold_t - 1):
                coefficients.append(random.randint(0, self.modulus_q - 1))

            # Evaluate polynomial at different points for each client
            for share_client_id in range(self.num_clients):
                x = share_client_id + 1  # x cannot be 0 in Shamir's scheme

                # Evaluate polynomial f(x)
                share_value = 0
                x_power = 1
                for coeff in coefficients:
                    share_value = (share_value + coeff * x_power) % self.modulus_q
                    x_power = (x_power * x) % self.modulus_q

                shares[share_client_id].append(share_value)

        return shares

    def mask_gradient(
        self, client_id: int, gradient: List[float], active_clients: List[int]
    ) -> Tuple[List[int], Dict[int, List[int]]]:
        """
        Mask gradient and create secret shares of the mask
        Returns: (masked_gradient, secret_shares_of_mask)
        """
        # Convert gradient to finite field with proper handling of negative values
        gradient_int = []
        for g in gradient:
            scaled_g = int(g * self.scale_factor)
            # Handle negative values properly in finite field
            if scaled_g < 0:
                scaled_g = (scaled_g % self.modulus_q + self.modulus_q) % self.modulus_q
            else:
                scaled_g = scaled_g % self.modulus_q
            gradient_int.append(scaled_g)

        # Generate mask considering only active clients
        mask = self.generate_mask(client_id, active_clients)

        # Apply mask
        masked_gradient = [(g + m) % self.modulus_q for g, m in zip(gradient_int, mask)]

        # Create secret shares of the mask for dropout resilience
        mask_shares = self.create_secret_shares(client_id, mask)

        return masked_gradient, mask_shares

    def reconstruct_secret(
        self, shares: Dict[int, List[int]], positions: List[int]
    ) -> List[int]:
        """
        Reconstruct secret from shares using Lagrange interpolation
        shares: {client_id: [share_values]}
        positions: list of client_ids whose shares to use
        """
        if len(positions) < self.threshold_t:
            raise ValueError(
                f"Need at least {self.threshold_t} shares, got {len(positions)}"
            )

        # Use first threshold_t shares
        active_positions = positions[: self.threshold_t]
        num_values = len(shares[active_positions[0]])

        reconstructed = []

        for value_idx in range(num_values):
            # Get points (x, y) for Lagrange interpolation
            points = [(pos + 1, shares[pos][value_idx]) for pos in active_positions]

            # Lagrange interpolation at x=0 to get original secret
            result = 0
            for i, (xi, yi) in enumerate(points):
                # Calculate Lagrange coefficient L_i(0)
                li = 1
                for j, (xj, _) in enumerate(points):
                    if i != j:
                        # li *= (0 - xj) / (xi - xj) in finite field
                        numerator = (-xj) % self.modulus_q
                        denominator = (xi - xj) % self.modulus_q
                        # Compute modular inverse
                        denominator_inv = pow(
                            denominator, self.modulus_q - 2, self.modulus_q
                        )
                        li = (li * numerator * denominator_inv) % self.modulus_q

                result = (result + yi * li) % self.modulus_q

            reconstructed.append(result)

        return reconstructed

    def _finite_field_to_float(self, ff_value: int) -> float:
        """
        Convert finite field value back to float, handling negative values correctly
        """
        # Check if the value represents a negative number
        if ff_value > self.modulus_q // 2:
            # Convert to negative
            float_value = -(self.modulus_q - ff_value) / self.scale_factor
        else:
            # Positive value
            float_value = ff_value / self.scale_factor

        return float_value

    def aggregate_gradients(
        self,
        client_gradients: Dict[int, List[float]],
        active_clients: Optional[List[int]] = None,
    ) -> List[float]:
        """
        Complete secure aggregation protocol
        """
        if active_clients is None:
            active_clients = list(client_gradients.keys())

        print(f"Starting secure aggregation with {len(active_clients)} active clients")

        # Phase 1: Setup (already done in __init__)
        setup_info = self.setup_phase()

        # Phase 2: Each client masks their gradient
        masked_gradients = {}
        all_mask_shares = {}

        for client_id in active_clients:
            gradient = client_gradients[client_id]
            masked_grad, mask_shares = self.mask_gradient(
                client_id, gradient, active_clients
            )

            masked_gradients[client_id] = masked_grad
            all_mask_shares[client_id] = mask_shares

        # Phase 3: Sum all masked gradients
        summed_masked = [0] * self.gradient_size
        for client_id in active_clients:
            masked_grad = masked_gradients[client_id]
            for i in range(self.gradient_size):
                summed_masked[i] = (summed_masked[i] + masked_grad[i]) % self.modulus_q

        # Phase 4: The masks from active clients should cancel out automatically
        # No need to subtract anything for active clients since pairwise masks cancel

        # Phase 5: Convert back to float values
        final_aggregate = []
        for i in range(self.gradient_size):
            final_value = self._finite_field_to_float(summed_masked[i])
            final_aggregate.append(final_value)

        return final_aggregate


# Demonstration and testing
def test_secure_aggregation():
    """
    Test the secure aggregation implementation
    """
    print("Testing Secure Aggregation Implementation")
    print("=" * 50)

    # Setup
    num_clients = 5
    threshold = 3
    gradient_size = 10

    # Generate random gradients for each client
    client_gradients = {}
    for i in range(num_clients):
        gradient = [random.uniform(-1.0, 1.0) for _ in range(gradient_size)]
        client_gradients[i] = gradient

    # Calculate expected sum (ground truth)
    expected_sum = [0.0] * gradient_size
    for client_grad in client_gradients.values():
        for i, val in enumerate(client_grad):
            expected_sum[i] += val

    # Run secure aggregation
    sa = SecureAggregation(num_clients, threshold, gradient_size)

    print("Test 1: All clients active")
    result1 = sa.aggregate_gradients(client_gradients)
    error1 = sum((a - b) ** 2 for a, b in zip(result1, expected_sum)) ** 0.5
    print(f"RMSE: {error1:.6f}")
    print(f"Expected: {expected_sum[:3]}...")
    print(f"Got:      {result1[:3]}...")
    print(f"Success: {error1 < 0.01}")

    # print("\nTest 2: With dropouts")
    # active_clients = [0, 1, 3]  # Drop clients 2 and 4
    # active_gradients = {i: client_gradients[i] for i in active_clients}
    # expected_sum_active = [0.0] * gradient_size
    # for i in active_clients:
    #     for j, val in enumerate(client_gradients[i]):
    #         expected_sum_active[j] += val
    #
    # result2 = sa.aggregate_gradients(active_gradients, active_clients)
    # error2 = sum((a - b) ** 2 for a, b in zip(result2, expected_sum_active)) ** 0.5
    # print(f"RMSE: {error2:.6f}")
    # print(f"Expected: {expected_sum_active[:3]}...")
    # print(f"Got:      {result2[:3]}...")
    # print(f"Success: {error2 < 0.01}")
    #
    # return error1 < 0.01 and error2 < 0.01

    return error1 < 0.01


if __name__ == "__main__":
    success = test_secure_aggregation()
    print(f"\nOverall test result: {'PASSED' if success else 'FAILED'}")
