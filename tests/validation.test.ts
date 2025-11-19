import {
  isValidEthereumAddress,
  validateEthereumAddress,
  validateAddressArray,
  sanitizeNickname
} from '../packages/ts-lib/src/validation';

describe('isValidEthereumAddress', () => {
  test('accepts valid lowercase address', () => {
    expect(isValidEthereumAddress('0x1234567890abcdef1234567890abcdef12345678')).toBe(true);
  });

  test('accepts valid uppercase address', () => {
    expect(isValidEthereumAddress('0x1234567890ABCDEF1234567890ABCDEF12345678')).toBe(true);
  });

  test('accepts valid mixed-case address', () => {
    expect(isValidEthereumAddress('0x1234567890aBcDeF1234567890AbCdEf12345678')).toBe(true);
  });

  test('rejects address without 0x prefix', () => {
    expect(isValidEthereumAddress('1234567890abcdef1234567890abcdef12345678')).toBe(false);
  });

  test('rejects address with wrong length', () => {
    expect(isValidEthereumAddress('0x123')).toBe(false);
    expect(isValidEthereumAddress('0x1234567890abcdef1234567890abcdef123456789')).toBe(false);
  });

  test('rejects address with invalid characters', () => {
    expect(isValidEthereumAddress('0x123456789gabcdef1234567890abcdef12345678')).toBe(false);
    expect(isValidEthereumAddress('0x1234567890abcdef1234567890abcdef1234567g')).toBe(false);
  });

  test('rejects non-string values', () => {
    expect(isValidEthereumAddress(null as any)).toBe(false);
    expect(isValidEthereumAddress(undefined as any)).toBe(false);
    expect(isValidEthereumAddress(123 as any)).toBe(false);
    expect(isValidEthereumAddress({} as any)).toBe(false);
  });
});

describe('validateEthereumAddress', () => {
  test('returns normalized lowercase address', () => {
    expect(validateEthereumAddress('0x1234567890ABCDEF1234567890ABCDEF12345678'))
      .toBe('0x1234567890abcdef1234567890abcdef12345678');
  });

  test('throws error for invalid address', () => {
    expect(() => validateEthereumAddress('invalid'))
      .toThrow('Invalid Ethereum address');
  });

  test('throws error for address without 0x', () => {
    expect(() => validateEthereumAddress('1234567890abcdef1234567890abcdef12345678'))
      .toThrow('Invalid Ethereum address');
  });
});

describe('validateAddressArray', () => {
  test('validates and normalizes array of addresses', () => {
    const input = [
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
      '0xAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA'
    ];
    const result = validateAddressArray(input);
    expect(result).toEqual([
      '0x1111111111111111111111111111111111111111',
      '0x2222222222222222222222222222222222222222',
      '0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    ]);
  });

  test('throws error for non-array input', () => {
    expect(() => validateAddressArray('not-an-array'))
      .toThrow('Addresses must be an array');
    expect(() => validateAddressArray(null))
      .toThrow('Addresses must be an array');
    expect(() => validateAddressArray({}))
      .toThrow('Addresses must be an array');
  });

  test('throws error for empty array', () => {
    expect(() => validateAddressArray([]))
      .toThrow('Addresses array cannot be empty');
  });

  test('throws error for oversized array', () => {
    const oversized = Array(1001).fill('0x1234567890abcdef1234567890abcdef12345678');
    expect(() => validateAddressArray(oversized))
      .toThrow('Addresses array too large');
  });

  test('throws error for non-string elements', () => {
    expect(() => validateAddressArray([123, 456]))
      .toThrow('Address at index 0 must be a string');
  });

  test('throws error for invalid address in array', () => {
    const input = [
      '0x1111111111111111111111111111111111111111',
      'invalid-address'
    ];
    expect(() => validateAddressArray(input))
      .toThrow('Invalid Ethereum address at index 1');
  });

  test('accepts maximum valid array size', () => {
    const maxSized = Array(1000).fill('0x1234567890abcdef1234567890abcdef12345678');
    const result = validateAddressArray(maxSized);
    expect(result.length).toBe(1000);
  });
});

describe('sanitizeNickname', () => {
  test('returns null for null input', () => {
    expect(sanitizeNickname(null)).toBe(null);
  });

  test('returns null for undefined input', () => {
    expect(sanitizeNickname(undefined)).toBe(null);
  });

  test('returns null for empty string', () => {
    expect(sanitizeNickname('')).toBe(null);
  });

  test('trims whitespace', () => {
    expect(sanitizeNickname('  Alice  ')).toBe('Alice');
  });

  test('removes dangerous characters', () => {
    expect(sanitizeNickname('Alice<script>alert()</script>')).toBe('Alicescriptalert()/script');
    expect(sanitizeNickname('Bob"test"')).toBe('Bobtest');
    expect(sanitizeNickname("Eve's address")).toBe('Eves address');
  });

  test('allows safe characters', () => {
    expect(sanitizeNickname('Alice-Bob_123')).toBe('Alice-Bob_123');
    expect(sanitizeNickname('Trader #1')).toBe('Trader #1');
  });

  test('throws error for non-string input', () => {
    expect(() => sanitizeNickname(123))
      .toThrow('Nickname must be a string');
    expect(() => sanitizeNickname({}))
      .toThrow('Nickname must be a string');
  });

  test('throws error for nickname exceeding max length', () => {
    const tooLong = 'a'.repeat(101);
    expect(() => sanitizeNickname(tooLong))
      .toThrow('Nickname too long');
  });

  test('accepts nickname at max length', () => {
    const maxLength = 'a'.repeat(100);
    expect(sanitizeNickname(maxLength)).toBe(maxLength);
  });
});
