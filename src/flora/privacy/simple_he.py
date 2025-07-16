import numpy as np
import torch
import seal


def setup_seal_parameters():
    parms = seal.EncryptionParameters(seal.scheme_type.BFV)
    poly_modulus_degree = 4096
    parms.set_poly_modulus_degree(poly_modulus_degree)

    # Set the plain modulus
    plain_modulus = seal.Plaintext(1 << 10)
    parms.set_plain_modulus(plain_modulus)

    return parms


def encrypt_tensor(tensor, encryptor):
    # Convert tensor to numpy array
    tensor_np = tensor.detach().cpu().numpy()

    # Encrypt the array
    encrypted_array = []
    for value in tensor_np.flatten():
        plain = seal.Plaintext(str(value))
        encrypted = seal.Ciphertext()
        encryptor.encrypt(plain, encrypted)
        encrypted_array.append(encrypted)

    return encrypted_array


def decrypt_tensor(encrypted_array, decryptor):
    decrypted_array = []
    for encrypted in encrypted_array:
        plain = seal.Plaintext()
        decryptor.decrypt(encrypted, plain)
        decrypted_value = float(plain.to_string())
        decrypted_array.append(decrypted_value)

    return torch.tensor(decrypted_array).reshape(-1)


def homomorphic_addition(encrypted_tensor1, encrypted_tensor2, evaluator):
    # Assuming both tensors are the same length and same shape
    encrypted_result = []
    for enc1, enc2 in zip(encrypted_tensor1, encrypted_tensor2):
        encrypted_enc = seal.Ciphertext()
        evaluator.add(enc1, enc2, encrypted_enc)
        encrypted_result.append(encrypted_enc)

    return encrypted_result


def main():
    # Setting up SEAL parameters
    parms = setup_seal_parameters()

    # Create SEAL Context
    context = seal.SEALContext.Create(parms)
    keygen = seal.KeyGenerator(context)
    encryptor = seal.Encryptor(context, keygen.public_key())
    decryptor = seal.Decryptor(context, keygen.secret_key())
    evaluator = seal.Evaluator(context)

    # Example tensors
    tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Encrypt tensors
    encrypted_tensor1 = encrypt_tensor(tensor1, encryptor)
    encrypted_tensor2 = encrypt_tensor(tensor2, encryptor)

    # Perform homomorphic addition on encrypted tensors
    encrypted_result = homomorphic_addition(encrypted_tensor1, encrypted_tensor2, evaluator)

    # Decrypt the result
    decrypted_result = decrypt_tensor(encrypted_result, decryptor)

    # Print out the original and the decrypted tensor
    print("Original Tensor 1:\n", tensor1)
    print("Original Tensor 2:\n", tensor2)
    print("Decrypted Result:\n", decrypted_result)


if __name__ == "__main__":
    main()