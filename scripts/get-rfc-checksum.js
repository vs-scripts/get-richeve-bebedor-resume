#!/usr/bin/env node

/**
 * @file get-rfc-checksum.js
 * @description Generates a SHA-256 checksum for CONCISE-RFC-000000 documents.
 * It handles the 'checksum-neutral' requirement by replacing existing
 * checksum values with a placeholder during calculation.
 */

const fs = require('fs');
const crypto = require('crypto');
const path = require('path');

/**
 * Calculates the checksum of an RFC document.
 *
 * @param {string} filePath - Path to the file.
 * @returns {string} The SHA-256 hash in hex format.
 */
function getRfcChecksum(filePath) {
    if (!fs.existsSync(filePath)) {
        throw new Error(`File not found: ${filePath}`);
    }

    // Read file as raw buffer to avoid encoding issues initially
    const buffer = fs.readFileSync(filePath);
    let content = buffer.toString('utf8');

    // 1. Canonical Normalization: Line Endings
    // RFC documents MUST follow line feed end of line format.
    // We normalize to LF for hashing consistency across platforms.
    content = content.replace(/\r\n/g, '\n');

    // 2. Checksum Neutralization
    // We look for the CHECKSUM field in the metadata section.
    // Pattern: Match '- CHECKSUM:' or 'CHECKSUM:' followed by any value.
    // We replace the value with the literal placeholder '<SHA-256>'.
    const checksumRegex = /^(.*?\bCHECKSUM:\s*)(.*)$/gm;
    content = content.replace(checksumRegex, '$1<SHA-256>');

    // 3. Generate SHA-256
    return crypto.createHash('sha256').update(content, 'utf8').digest('hex');
}

// Main execution
function main() {
    const args = process.argv.slice(2);
    if (args.length === 0) {
        console.log('Usage: node get-rfc-checksum.js <file-path>');
        process.exit(0);
    }

    const target = path.resolve(args[0]);
    try {
        const hash = getRfcChecksum(target);
        process.stdout.write(hash); // Output only the hash to stdout
    } catch (err) {
        process.stderr.write(`Error: ${err.message}\n`);
        process.exit(1);
    }
}

if (require.main === module) {
    main();
}
