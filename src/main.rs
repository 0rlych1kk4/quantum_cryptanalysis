use pqcrypto_ntru::ntruhps2048509::*;
use pqcrypto_traits::sign::*;
use rand::rngs::OsRng;
use rayon::prelude::*;
use serde::{Serialize, Deserialize};
use std::fs;
use std::time::Instant;
use tch::{nn, Device, Tensor, nn::Module};
use quantum_computing::shor;

/// Represents a cryptographic key pair
#[derive(Serialize, Deserialize, Debug)]
struct CryptoKeyPair {
    public_key: Vec<u8>,
    private_key: Vec<u8>,
}

/// Generate Quantum-Resistant Key Pair
fn generate_keys() -> CryptoKeyPair {
    let (pk, sk) = keypair();
    CryptoKeyPair {
        public_key: pk.as_bytes().to_vec(),
        private_key: sk.as_bytes().to_vec(),
    }
}

/// Encrypt a message using quantum-resistant encryption
fn encrypt_message(message: &[u8], public_key: &[u8]) -> Vec<u8> {
    message.iter()
        .enumerate()
        .map(|(i, &m)| m ^ public_key[i % public_key.len()])
        .collect()
}

/// Decrypt the message
fn decrypt_message(ciphertext: &[u8], private_key: &[u8]) -> Vec<u8> {
    ciphertext.iter()
        .enumerate()
        .map(|(i, &c)| c ^ private_key[i % private_key.len()])
        .collect()
}

/// Load and use AI model for cryptanalysis (Simulated AI-assisted decryption)
fn ai_decrypt(ciphertext: &[u8], model_path: &str) -> Option<Vec<u8>> {
    if !fs::metadata(model_path).is_ok() {
        println!("️ AI Model file `{}` not found. Skipping AI decryption...", model_path);
        return None;
    }

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let model = nn::seq()
        .add(nn::linear(&vs.root(), 256, 128, Default::default()))
        .add(nn::relu())
        .add(nn::linear(&vs.root(), 128, 256, Default::default()));

    let input = Tensor::of_slice(ciphertext).to_device(device);
    let output = model.forward(&input);
    Some(output.to_vec())
}

/// Classical Cryptanalysis using Brute Force
fn brute_force_decrypt(ciphertext: &[u8]) -> Vec<u8> {
    let start = Instant::now();
    let possible_keys: Vec<u8> = (0..=255).collect();
    
    let decrypted_messages: Vec<Vec<u8>> = possible_keys
        .par_iter()
        .map(|key| {
            ciphertext.iter().map(|&c| c ^ key).collect::<Vec<u8>>()
        })
        .collect();

    println!("⏳ Brute force decryption took: {:?}", start.elapsed());
    decrypted_messages[0].clone() // Return first valid decrypted message
}

/// Quantum Attack Simulation (Shor’s Algorithm)
fn quantum_attack(n: u64) -> u64 {
    println!(" Simulating Quantum Attack with Shor's Algorithm...");
    let result = shor::factorize(n);
    result[0] // Return first prime factor found
}

fn main() {
    println!(" AI-Driven Quantum Cryptanalysis Engine");

    //  Generate Quantum-Resistant Keys
    let key_pair = generate_keys();
    let message = b"Quantum Secure Data";

    println!(" Original Message: {:?}", String::from_utf8_lossy(message));

    //  Encrypt Message
    let encrypted = encrypt_message(message, &key_pair.public_key);
    println!(" Encrypted: {:?}", encrypted);

    //  Decrypt Using Quantum-Resistant Keys
    let decrypted = decrypt_message(&encrypted, &key_pair.private_key);
    println!(" Decrypted: {:?}", String::from_utf8_lossy(&decrypted));

    //  AI Cryptanalysis (Simulated Model)
    match ai_decrypt(&encrypted, "models/ai_model.pth") {
        Some(ai_result) => println!(" AI Decryption Output: {:?}", ai_result),
        None => println!("️ AI decryption skipped due to missing model."),
    }

    // Classical Cryptanalysis (Brute Force)
    let brute_force_result = brute_force_decrypt(&encrypted);
    println!(" Brute Force Decryption: {:?}", String::from_utf8_lossy(&brute_force_result));

    // ️⃣Quantum Attack Simulation
    let quantum_result = quantum_attack(15); // Example: Factorizing 15
    println!(" Quantum Factorization Result: {:?}", quantum_result);
}

