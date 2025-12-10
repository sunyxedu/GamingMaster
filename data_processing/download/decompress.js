#!/usr/bin/env node
/**
 * Decompress generals.io .gior replay files using LZ-string
 * 
 * Usage: node decompress.js <input.gior> <output.json>
 */

const LZString = require('lz-string');
const fs = require('fs');

if (process.argv.length < 4) {
    console.error('Usage: node decompress.js <input.gior> <output.json>');
    process.exit(1);
}

const inputFile = process.argv[2];
const outputFile = process.argv[3];

try {
    // Read the binary data
    const data = fs.readFileSync(inputFile);
    
    // Decompress using LZ-string
    const obj = JSON.parse(LZString.decompressFromUint8Array(new Uint8Array(data)));
    
    // Write the decompressed JSON
    fs.writeFileSync(outputFile, JSON.stringify(obj));
    
    process.exit(0);
} catch (error) {
    console.error('Decompression error:', error.message);
    process.exit(1);
}

