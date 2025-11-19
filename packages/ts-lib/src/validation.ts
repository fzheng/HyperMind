/**
 * Validation utilities for input data
 */

export function isValidEthereumAddress(address: string): boolean {
  if (typeof address !== 'string') return false;
  // Ethereum address: 0x followed by 40 hex characters
  return /^0x[0-9a-fA-F]{40}$/.test(address);
}

export function validateEthereumAddress(address: string): string {
  if (!isValidEthereumAddress(address)) {
    throw new Error(`Invalid Ethereum address: ${address}`);
  }
  return address.toLowerCase();
}

export function validateAddressArray(addresses: unknown): string[] {
  if (!Array.isArray(addresses)) {
    throw new Error('Addresses must be an array');
  }
  if (addresses.length === 0) {
    throw new Error('Addresses array cannot be empty');
  }
  if (addresses.length > 1000) {
    throw new Error('Addresses array too large (max 1000)');
  }

  return addresses.map((addr, idx) => {
    if (typeof addr !== 'string') {
      throw new Error(`Address at index ${idx} must be a string`);
    }
    if (!isValidEthereumAddress(addr)) {
      throw new Error(`Invalid Ethereum address at index ${idx}: ${addr}`);
    }
    return addr.toLowerCase();
  });
}

export function sanitizeNickname(nickname: unknown): string | null {
  if (nickname == null || nickname === '') return null;
  if (typeof nickname !== 'string') {
    throw new Error('Nickname must be a string');
  }
  const trimmed = nickname.trim();
  if (trimmed.length > 100) {
    throw new Error('Nickname too long (max 100 characters)');
  }
  // Remove potentially dangerous characters
  return trimmed.replace(/[<>\"']/g, '');
}
