// A key-value cache holds one key tensor and one value tensor for every block,
// for every head. The model reads and writes it but does not own it: the caller
// creates one and passes it in, so a single MiniGPT can serve many independent
// generations at once, each with its own cache.
export class KVCache {
    constructor(numBlocks, numHeads) {
        // One key tensor and one value tensor per block, per head. Each starts
        // null and becomes a growing tensor as tokens flow through.
        this.keys = [];
        this.values = [];
        for (let block = 0; block < numBlocks; ++block) {
            const blockKeys = [];
            const blockValues = [];
            for (let head = 0; head < numHeads; ++head) {
                blockKeys.push(null);
                blockValues.push(null);
            }
            this.keys.push(blockKeys);
            this.values.push(blockValues);
        }
    }

    // Tokens cached so far. Every block and head advances together, so the
    // first head's key tensor speaks for all of them.
    get length() {
        const first = this.keys[0][0];
        return first === null ? 0 : first.rows;
    }
}
